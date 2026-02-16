#include "StreamServer.hpp"

constexpr static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
constexpr static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
constexpr static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %lli.%06li\r\n\r\n";
constexpr static const char* STREAM_PART_JLS = "Content-Type: application/octet-stream\r\nContent-Length: %u\r\nX-Timestamp: %lli.%06li\r\nX-Encoding: jpegls\r\n\r\n";

static const char* STREAM_SERVER_TAG = "[STREAM_SERVER]";

// JPEG-LS encoder instance (lazily initialized, reused across frames)
static jpegls_encoder_t* s_jls_encoder = nullptr;
static uint8_t* s_jls_buffer = nullptr;
static size_t s_jls_buffer_cap = 0;

StreamServer::StreamServer(const int STREAM_PORT, StateManager* stateManager) : STREAM_SERVER_PORT(STREAM_PORT), stateManager(stateManager) {}

esp_err_t StreamHelpers::stream(httpd_req_t* req)
{
    camera_fb_t* fb = nullptr;
    struct timeval _timestamp;

    esp_err_t response = ESP_OK;
    size_t _out_buf_len = 0;
    uint8_t* _out_buf = nullptr;
    bool using_jls_buffer = false;

    // Buffer for multipart header
    char part_buf[256];
    static int64_t last_frame = 0;
    if (!last_frame)
        last_frame = esp_timer_get_time();

    // Pull event queue from user_ctx to send STREAM on/off notifications
    auto* stateManager = static_cast<StateManager*>(req->user_ctx);
    QueueHandle_t eventQueue = stateManager ? stateManager->GetEventQueue() : nullptr;
    bool stream_on_sent = false;

    // Check encoding mode
    bool use_jpegls = (deviceConfig && deviceConfig->getEncodingMode() == EncodingMode::JPEGLS);

    // Initialize JPEG-LS encoder if needed
    if (use_jpegls)
    {
        if (!s_jls_encoder)
        {
            s_jls_encoder = jpegls_encoder_create(240, 240);
            if (!s_jls_encoder)
            {
                ESP_LOGE(STREAM_SERVER_TAG, "Failed to create JPEG-LS encoder");
                use_jpegls = false;
            }
        }
        if (!s_jls_buffer)
        {
            s_jls_buffer_cap = jpegls_max_encoded_size(240, 240);
            s_jls_buffer = static_cast<uint8_t*>(malloc(s_jls_buffer_cap));
            if (!s_jls_buffer)
            {
                ESP_LOGE(STREAM_SERVER_TAG, "Failed to allocate JPEG-LS buffer");
                use_jpegls = false;
            }
        }
    }

    response = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (response != ESP_OK)
        return response;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "60");

    if (SendStreamEvent(eventQueue, StreamState_e::Stream_ON))
        stream_on_sent = true;

    while (true)
    {
        fb = esp_camera_fb_get();

        if (!fb)
        {
            ESP_LOGE(STREAM_SERVER_TAG, "Camera capture failed");
            response = ESP_FAIL;
            // Don't break immediately, try to recover
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        _timestamp.tv_sec = fb->timestamp.tv_sec;
        _timestamp.tv_usec = fb->timestamp.tv_usec;
        using_jls_buffer = false;

        if (use_jpegls && fb->format == PIXFORMAT_GRAYSCALE)
        {
            // Encode grayscale frame with JPEG-LS
            size_t encoded_len = 0;
            int64_t enc_start = esp_timer_get_time();
            int ret = jpegls_encode(s_jls_encoder,
                                    fb->buf, fb->len,
                                    s_jls_buffer, s_jls_buffer_cap,
                                    &encoded_len, 2);
            int64_t enc_us = esp_timer_get_time() - enc_start;
            if (ret == 0)
            {
                _out_buf = s_jls_buffer;
                _out_buf_len = encoded_len;
                using_jls_buffer = true;
                // Accumulate encoder timing stats
                static int64_t enc_total_us = 0;
                static int enc_count = 0;
                enc_total_us += enc_us;
                enc_count++;
                if (enc_count % 100 == 0) {
                    ESP_LOGI(STREAM_SERVER_TAG, "JPEG-LS encode: avg %lld us/frame (%lld ms), last %lld us, size %u bytes",
                             enc_total_us / enc_count, enc_total_us / enc_count / 1000, enc_us, (unsigned)encoded_len);
                    enc_total_us = 0;
                    enc_count = 0;
                }
            }
            else
            {
                ESP_LOGE(STREAM_SERVER_TAG, "JPEG-LS encoding failed (ret=%d, took %lld us), sending raw", ret, enc_us);
                _out_buf = fb->buf;
                _out_buf_len = fb->len;
            }
        }
        else
        {
            // Standard JPEG mode: camera already provides JPEG data
            _out_buf_len = fb->len;
            _out_buf = fb->buf;
        }

        if (response == ESP_OK)
            response = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (response == ESP_OK)
        {
            const char* part_fmt = (using_jls_buffer) ? STREAM_PART_JLS : STREAM_PART;
            size_t hlen = snprintf((char*)part_buf, sizeof(part_buf), part_fmt, _out_buf_len, _timestamp.tv_sec, _timestamp.tv_usec);
            response = httpd_resp_send_chunk(req, (const char*)part_buf, hlen);
        }
        if (response == ESP_OK)
            response = httpd_resp_send_chunk(req, (const char*)_out_buf, _out_buf_len);

        if (fb)
        {
            esp_camera_fb_return(fb);
            fb = NULL;
        }
        // Note: _out_buf points to either fb->buf (already returned) or s_jls_buffer (static, don't free)
        _out_buf = NULL;

        if (response != ESP_OK)
            break;

        if (esp_log_level_get(STREAM_SERVER_TAG) >= ESP_LOG_INFO)
        {
            static long last_request_time = 0;
            if (last_request_time == 0)
            {
                last_request_time = Helpers::getTimeInMillis();
            }

            // Only log every 100 frames to reduce overhead
            const int frame_window = 100;
            static int frame_count = 0;
            if (++frame_count % frame_window == 0)
            {
                long request_end = Helpers::getTimeInMillis();
                long window_ms = request_end - last_request_time;
                last_request_time = request_end;
                long fps = 0;
                if (window_ms > 0)
                {
                    fps = (frame_window * 1000) / window_ms;
                }
                ESP_LOGI(STREAM_SERVER_TAG, "%i Frames Size: %uKB, Time: %lims (%lifps)", frame_window, _out_buf_len / 1024, window_ms, fps);
            }
        }
    }
    last_frame = 0;

    if (stream_on_sent)
        SendStreamEvent(eventQueue, StreamState_e::Stream_OFF);

    return response;
}

esp_err_t StreamHelpers::ws_logs_handle(httpd_req_t* req)
{
    auto ret = webSocketLogger.register_socket_client(req);
    return ret;
}

esp_err_t StreamServer::startStreamServer()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.stack_size = 20480;
    config.max_uri_handlers = 10;
    config.server_port = STREAM_SERVER_PORT;
    config.ctrl_port = STREAM_SERVER_PORT;
    config.recv_wait_timeout = 5;    // 5 seconds for receiving
    config.send_wait_timeout = 5;    // 5 seconds for sending
    config.lru_purge_enable = true;  // Enable LRU purge for better connection handling

    httpd_uri_t stream_page = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = &StreamHelpers::stream,
        .user_ctx = this->stateManager,
    };

    httpd_uri_t logs_ws = {
        .uri = "/ws",
        .method = HTTP_GET,
        .handler = &StreamHelpers::ws_logs_handle,
        .user_ctx = nullptr,
        .is_websocket = true,
    };

    int status = httpd_start(&camera_stream, &config);

    if (status != ESP_OK)
    {
        ESP_LOGE(STREAM_SERVER_TAG, "Cannot start stream server.");
        return status;
    }

    httpd_register_uri_handler(camera_stream, &logs_ws);
    if (this->stateManager->GetCameraState() != CameraState_e::Camera_Success)
    {
        ESP_LOGE(STREAM_SERVER_TAG, "Camera not initialized. Cannot start stream server. Logs server will be running.");
        return ESP_FAIL;
    }

    httpd_register_uri_handler(camera_stream, &stream_page);

    // Initial state is OFF
    if (this->stateManager)
        SendStreamEvent(this->stateManager->GetEventQueue(), StreamState_e::Stream_OFF);

    ESP_LOGI(STREAM_SERVER_TAG, "Stream server started on port %d", STREAM_SERVER_PORT);

    return ESP_OK;
}
