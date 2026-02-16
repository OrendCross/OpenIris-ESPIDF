#include "jpegls_encoder.h"
#include "jpegls_tables.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "esp_log.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "sdkconfig.h"

// Force-inline macro for hot-path helpers
#define JPEGLS_FORCE_INLINE __attribute__((always_inline)) static inline

static const char *TAG = "[JPEGLS]";

// JPEG-LS constants for 8-bit
static constexpr int MAXVAL = 255;
static constexpr int RESET_THRESHOLD = 64;
static constexpr int LIMIT = 32;
static constexpr int NUM_CONTEXTS = 365;

// ============================================================================
// Tables in DRAM (internal SRAM, zero wait-state access)
// ============================================================================

// Mutable: rebuilt when NEAR changes (different thresholds)
DRAM_ATTR int8_t jpegls_quant_lut[511];
static int s_quant_lut_t1 = -1, s_quant_lut_t2 = -1, s_quant_lut_t3 = -1;
static int s_quant_lut_near = -1;

static void rebuild_quant_lut(int T1, int T2, int T3, int near_val)
{
    if (T1 == s_quant_lut_t1 && T2 == s_quant_lut_t2 &&
        T3 == s_quant_lut_t3 && near_val == s_quant_lut_near)
        return;  // already built for these parameters

    // LUT indexed by (delta + 255), delta in [-255..255]
    // T.87 A.3.3: for near-lossless, deadzone widens to [-NEAR, NEAR]
    for (int d = -255; d <= 255; d++) {
        int8_t q;
        if      (d <= -T3)        q = -4;
        else if (d <= -T2)        q = -3;
        else if (d <= -T1)        q = -2;
        else if (d < -near_val)   q = -1;
        else if (d <=  near_val)  q =  0;
        else if (d < T1)          q =  1;
        else if (d < T2)          q =  2;
        else if (d < T3)          q =  3;
        else                      q =  4;
        jpegls_quant_lut[d + 255] = q;
    }
    s_quant_lut_t1 = T1;
    s_quant_lut_t2 = T2;
    s_quant_lut_t3 = T3;
    s_quant_lut_near = near_val;
}

// Compute default T.87 thresholds for given NEAR value
static void compute_default_thresholds(int near_val, int *T1, int *T2, int *T3)
{
    if (near_val == 0) {
        // T.87 Table C.1 defaults for 8-bit lossless
        *T1 = 3; *T2 = 7; *T3 = 21;
    } else {
        // T.87 Annex C formula (matches CharLS)
        int factor = (MAXVAL + 128) / 256;
        if (factor < 1) factor = 1;
        int t1 = factor * (near_val + 1) + 2 + 3 * near_val;
        int t2 = factor * (near_val + 1) + 3 + 5 * near_val;
        int t3 = factor * (near_val + 1) + 4 + 7 * near_val;
        // Clamp per spec
        if (t1 < near_val + 1) t1 = near_val + 1;
        if (t1 > MAXVAL) t1 = MAXVAL;
        if (t2 < t1) t2 = t1;
        if (t2 > MAXVAL) t2 = MAXVAL;
        if (t3 < t2) t3 = t2;
        if (t3 > MAXVAL) t3 = MAXVAL;
        *T1 = t1; *T2 = t2; *T3 = t3;
    }
}

// Context+sign LUT: indexed by (q1+4)*81+(q2+4)*9+(q3+4)
// Value = (sign << 15) | context_id
// Context ID uses CharLS-compatible formula: q1*81 + q2*9 + q3 (range 0..364)
static DRAM_ATTR uint16_t s_ctx_sign_lut[729];
static bool s_ctx_lut_initialized = false;

static void init_ctx_sign_lut()
{
    if (s_ctx_lut_initialized) return;
    for (int q1 = -4; q1 <= 4; q1++) {
        for (int q2 = -4; q2 <= 4; q2++) {
            for (int q3 = -4; q3 <= 4; q3++) {
                int idx = (q1 + 4) * 81 + (q2 + 4) * 9 + (q3 + 4);
                // Apply sign-flip per T.87
                int s = 0, a = q1, b = q2, c = q3;
                if (a < 0 || (a == 0 && b < 0) || (a == 0 && b == 0 && c < 0)) {
                    a = -a; b = -b; c = -c; s = 1;
                }
                // CharLS-compatible context ID: q1*81 + q2*9 + q3
                // After normalization: a in [0..4], b in [-4..4], c in [-4..4]
                // Range: 0..364
                int ctx_id = a * 81 + b * 9 + c;
                s_ctx_sign_lut[idx] = (uint16_t)((s << 15) | ctx_id);
            }
        }
    }
    s_ctx_lut_initialized = true;
}

static DRAM_ATTR const int J[32] = {
    0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 5, 5, 6, 6, 7, 7,
    8, 9, 10, 11, 12, 13, 14, 15
};

static DRAM_ATTR const int J_pow[32] = {
    1, 1, 1, 1, 2, 2, 2, 2,
    4, 4, 4, 4, 8, 8, 8, 8,
    16, 16, 32, 32, 64, 64, 128, 128,
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768
};

// ============================================================================
// BitWriter - 64-bit accumulator with JPEG-LS bit-stuffing
// ============================================================================

struct BitWriter {
    uint8_t *buf;
    size_t cap;
    size_t byte_pos;
    uint64_t bit_buf;
    int bits_in_buf;
    int free_bits;
    bool overflow;
};

JPEGLS_FORCE_INLINE void bw_init(BitWriter *bw, uint8_t *buf, size_t cap)
{
    bw->buf = buf;
    bw->cap = cap;
    bw->byte_pos = 0;
    bw->bit_buf = 0;
    bw->bits_in_buf = 0;
    bw->free_bits = 8;
    bw->overflow = false;
}

JPEGLS_FORCE_INLINE void bw_put_raw(BitWriter *bw, uint8_t val)
{
    if (__builtin_expect(bw->byte_pos < bw->cap, 1))
        bw->buf[bw->byte_pos++] = val;
    else
        bw->overflow = true;
}

// Append bits and flush - optimized for common case (free_bits=8, 1-2 bytes flushed)
JPEGLS_FORCE_INLINE void bw_append_bits(BitWriter *bw, uint64_t value, int count)
{
    bw->bit_buf = (bw->bit_buf << count) | (value & ((1ULL << count) - 1));
    bw->bits_in_buf += count;

    // Unrolled flush: most calls produce 0-2 bytes, almost never 0xFF
    while (bw->bits_in_buf >= bw->free_bits) {
        bw->bits_in_buf -= bw->free_bits;
        uint8_t byte_val = (uint8_t)((bw->bit_buf >> bw->bits_in_buf)
                                     & ((1u << bw->free_bits) - 1));
        if (__builtin_expect(bw->byte_pos < bw->cap, 1))
            bw->buf[bw->byte_pos++] = byte_val;
        else
            bw->overflow = true;
        // 0xFF is rare in encoded data; branch predictor will learn this
        bw->free_bits = (byte_val == 0xFF) ? 7 : 8;
    }
}

JPEGLS_FORCE_INLINE void bw_append_bit(BitWriter *bw, int bit)
{
    bw_append_bits(bw, bit, 1);
}

static void bw_finalize(BitWriter *bw)
{
    if (bw->bits_in_buf > 0) {
        int pad = bw->free_bits - bw->bits_in_buf;
        uint8_t byte_val = (uint8_t)((bw->bit_buf << pad)
                                     & ((1u << bw->free_bits) - 1));
        bw_put_raw(bw, byte_val);
        bw->bits_in_buf = 0;
        bw->bit_buf = 0;
        // If last byte is 0xFF, write stuff byte per JPEG-LS spec
        if (byte_val == 0xFF) {
            bw_put_raw(bw, 0x00);
        }
        bw->free_bits = 8;
    } else if (bw->free_bits == 7) {
        // No remaining bits but last emitted byte was 0xFF - write stuff byte
        bw_put_raw(bw, 0x00);
        bw->free_bits = 8;
    }
}

// ============================================================================
// Encoder state
// ============================================================================

struct RegularContext {
    int32_t A;
    int32_t B;
    int32_t C;
    int32_t N;
};

struct RunContext {
    int32_t A;
    int32_t N;
    int32_t Nn;
};

struct jpegls_encoder {
    int width;
    int height;
    int near_val;
    int range;
    int qbpp_val;

    uint8_t *prev_line;
    uint8_t *curr_line;

    RegularContext contexts[NUM_CONTEXTS];
    RunContext run_contexts[2];
    int run_index;
    BitWriter bw;
};

static void reset_contexts(jpegls_encoder_t *enc)
{
    // T.87: A_init = max(2, floor((RANGE + 32) / 64))
    int a_init = (enc->range + 32) / 64;
    if (a_init < 2) a_init = 2;

    for (int i = 0; i < NUM_CONTEXTS; i++) {
        enc->contexts[i] = {a_init, 0, 0, 1};
    }
    enc->run_contexts[0] = {a_init, 1, 0};
    enc->run_contexts[1] = {a_init, 1, 0};
    enc->run_index = 0;
}

jpegls_encoder_t *jpegls_encoder_create(int width, int height)
{
    // Allocate in internal SRAM for zero-wait-state access
    auto *enc = static_cast<jpegls_encoder_t*>(
        heap_caps_calloc(1, sizeof(jpegls_encoder_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!enc) {
        ESP_LOGW(TAG, "Internal SRAM full (%u bytes), falling back to default heap",
                 (unsigned)sizeof(jpegls_encoder_t));
        enc = static_cast<jpegls_encoder_t*>(calloc(1, sizeof(jpegls_encoder_t)));
        if (!enc) return nullptr;
    }

    enc->width = width;
    enc->height = height;
    enc->near_val = 0;
    enc->range = 256;
    enc->qbpp_val = 8;

    size_t line_size = static_cast<size_t>(width + 2);
    size_t aligned_size = (line_size + 15) & ~(size_t)15;

    // Line buffers in internal SRAM too
    enc->prev_line = static_cast<uint8_t*>(
        heap_caps_calloc(aligned_size, 1, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    enc->curr_line = static_cast<uint8_t*>(
        heap_caps_calloc(aligned_size, 1, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

    if (!enc->prev_line || !enc->curr_line) {
        // Fallback to default heap
        if (!enc->prev_line)
            enc->prev_line = static_cast<uint8_t*>(calloc(aligned_size, 1));
        if (!enc->curr_line)
            enc->curr_line = static_cast<uint8_t*>(calloc(aligned_size, 1));
    }

    if (!enc->prev_line || !enc->curr_line) {
        jpegls_encoder_destroy(enc);
        return nullptr;
    }

    init_ctx_sign_lut();
    rebuild_quant_lut(3, 7, 21, 0);  // lossless defaults
    reset_contexts(enc);
    return enc;
}

void jpegls_encoder_destroy(jpegls_encoder_t *enc)
{
    if (!enc) return;
    free(enc->prev_line);
    free(enc->curr_line);
    heap_caps_free(enc);
}

void jpegls_encoder_reset(jpegls_encoder_t *enc)
{
    if (!enc) return;
    reset_contexts(enc);
    memset(enc->prev_line, 0, static_cast<size_t>(enc->width + 2));
    memset(enc->curr_line, 0, static_cast<size_t>(enc->width + 2));
}

size_t jpegls_max_encoded_size(int width, int height)
{
    return static_cast<size_t>(width) * height * 2 + 100;
}

// ============================================================================
// Core encoding helpers
// ============================================================================

JPEGLS_FORCE_INLINE int med_predictor(int a, int b, int c)
{
    int min_ab = (a < b) ? a : b;
    int max_ab = (a > b) ? a : b;
    if (c >= max_ab) return min_ab;
    if (c <= min_ab) return max_ab;
    return a + b - c;
}

// Golomb k parameter: shift-compare loop instead of division.
// On Xtensa, division takes ~32 cycles; this loop takes 2 cycles per iteration
// and typical k values are 0-4, so 0-8 cycles total.
JPEGLS_FORCE_INLINE int compute_k(int A, int N)
{
    int k = 0;
    for (int Nk = N; Nk < A; Nk <<= 1)
        k++;
    return k;
}

// Batched Golomb-Rice: emit entire codeword in one call
JPEGLS_FORCE_INLINE void encode_mapped_error(BitWriter *bw, int mapped_val,
                                             int k, int limit, int qbpp_val)
{
    int unary = mapped_val >> k;

    if (__builtin_expect(unary < limit - qbpp_val - 1, 1)) {
        int total_bits = unary + 1 + k;
        uint32_t codeword = (1u << k) | (mapped_val & ((1u << k) - 1));
        if (__builtin_expect(total_bits <= 48, 1)) {
            bw_append_bits(bw, codeword, total_bits);
        } else {
            int zeros = unary;
            while (zeros > 24) { bw_append_bits(bw, 0, 24); zeros -= 24; }
            bw_append_bits(bw, codeword, zeros + 1 + k);
        }
    } else {
        int escape_unary = limit - qbpp_val - 1;
        int total_bits = escape_unary + 1 + qbpp_val;
        uint32_t codeword = (1u << qbpp_val) | ((mapped_val - 1) & ((1u << qbpp_val) - 1));
        if (__builtin_expect(total_bits <= 48, 1)) {
            bw_append_bits(bw, codeword, total_bits);
        } else {
            int zeros = escape_unary;
            while (zeros > 24) { bw_append_bits(bw, 0, 24); zeros -= 24; }
            bw_append_bits(bw, codeword, zeros + 1 + qbpp_val);
        }
    }
}

JPEGLS_FORCE_INLINE int map_error_regular(int errval, int k,
                                          const RegularContext *ctx,
                                          int near_val)
{
    (void)k;
    (void)near_val;
    (void)ctx;
    // CharLS: get_error_correction(k | near_lossless) uses bit_wise_sign(param + 1)
    // which is always 0 for non-negative param, so correction is always 0.
    // No error correction applied in encoder (encoder/decoder agree).
    return (errval >= 0) ? 2 * errval : -2 * errval - 1;
}

JPEGLS_FORCE_INLINE int clamp_val(int val)
{
    if (__builtin_expect(val < 0, 0)) return 0;
    if (__builtin_expect(val > MAXVAL, 0)) return MAXVAL;
    return val;
}

// T.87 reconstruction with modular wrapping (CharLS fix_reconstructed_value)
JPEGLS_FORCE_INLINE int fix_reconstructed(int val, int near_val, int range)
{
    if (val < -near_val)
        val += range * (2 * near_val + 1);
    else if (val > MAXVAL + near_val)
        val -= range * (2 * near_val + 1);
    return clamp_val(val);
}

JPEGLS_FORCE_INLINE int reduce_error(int errval, int range)
{
    int half = range / 2;
    if (errval >= half)
        return errval - range;
    if (errval < -half)
        return errval + range;
    return errval;
}

JPEGLS_FORCE_INLINE void update_regular_context(RegularContext *ctx, int errval,
                                                 int near_x2p1)
{
    ctx->A += (errval < 0) ? -errval : errval;
    ctx->B += errval * near_x2p1;  // T.87 A.6.3: B += Errval * (2*NEAR+1)

    if (ctx->N == RESET_THRESHOLD) {
        ctx->A >>= 1;
        ctx->B >>= 1;
        ctx->N >>= 1;
    }

    ctx->N++;

    if (ctx->B + ctx->N <= 0) {
        ctx->B += ctx->N;
        if (ctx->B <= -ctx->N)
            ctx->B = -ctx->N + 1;
        if (ctx->C > -128) ctx->C--;
    } else if (ctx->B > 0) {
        ctx->B -= ctx->N;
        if (ctx->B > 0) ctx->B = 0;
        if (ctx->C < 127) ctx->C++;
    }
}

// ============================================================================
// Run-length scanning - 8 bytes at a time
// ============================================================================

JPEGLS_FORCE_INLINE int scan_run(const uint8_t *pixels, uint8_t ref, int max_count)
{
    int count = 0;

    // Align to 8-byte boundary
    while (count < max_count && ((uintptr_t)(pixels + count) & 7)) {
        if (pixels[count] != ref) return count;
        count++;
    }

    uint64_t ref8 = (uint64_t)ref * 0x0101010101010101ULL;
    while (count + 8 <= max_count) {
        uint64_t word;
        memcpy(&word, pixels + count, 8);
        if (word != ref8) break;
        count += 8;
    }

    while (count < max_count && pixels[count] == ref)
        count++;

    return count;
}

// Near-lossless: tolerance-based run scan (byte-by-byte)
JPEGLS_FORCE_INLINE int scan_run_near(const uint8_t *pixels, uint8_t ref,
                                       int near_val, int max_count)
{
    int count = 0;
    while (count < max_count) {
        int diff = (int)pixels[count] - (int)ref;
        if (diff < -near_val || diff > near_val) break;
        count++;
    }
    return count;
}

// Near-lossless error quantization (T.87 A.6.2)
JPEGLS_FORCE_INLINE int quantize_error(int errval, int near_val)
{
    if (errval > 0)
        return (errval + near_val) / (2 * near_val + 1);
    else if (errval < 0)
        return -((near_val - errval) / (2 * near_val + 1));
    return 0;
}

// ============================================================================
// Run mode encoding
// ============================================================================

JPEGLS_FORCE_INLINE void encode_run_length(BitWriter *bw, int run_count,
                                           int end_of_line, int *run_index)
{
    int ri = *run_index;

    while (run_count >= J_pow[ri]) {
        bw_append_bit(bw, 1);
        run_count -= J_pow[ri];
        if (ri < 31) ri++;
    }

    if (!end_of_line) {
        bw_append_bit(bw, 0);
        if (J[ri] > 0)
            bw_append_bits(bw, run_count, J[ri]);
        if (ri > 0) ri--;
    } else if (run_count > 0) {
        bw_append_bit(bw, 1);
    }

    *run_index = ri;
}

JPEGLS_FORCE_INLINE bool compute_run_map(int errval, int k, int Nn, int N)
{
    if (k == 0 && errval > 0 && 2 * Nn < N) return true;
    if (errval < 0 && 2 * Nn >= N) return true;
    if (errval < 0 && k != 0) return true;
    return false;
}

JPEGLS_FORCE_INLINE void encode_run_interruption(BitWriter *bw, int errval,
                                                  RunContext *ctx,
                                                  int ri_type, int run_index,
                                                  int qbpp_val)
{
    int temp = ctx->A + (ctx->N >> 1) * ri_type;
    int k = compute_k(temp, ctx->N);

    int abs_err = (errval < 0) ? -errval : errval;

    bool map = compute_run_map(errval, k, ctx->Nn, ctx->N);
    int mapped = 2 * abs_err - ri_type - (map ? 1 : 0);
    if (mapped < 0) mapped = 0;

    int limit_reduce = J[run_index] + 1;
    encode_mapped_error(bw, mapped, k, LIMIT - limit_reduce, qbpp_val);

    if (errval < 0) ctx->Nn++;
    ctx->A += (mapped + 1 - ri_type) >> 1;

    if (ctx->N == RESET_THRESHOLD) {
        ctx->A >>= 1;
        ctx->N >>= 1;
        ctx->Nn >>= 1;
    }
    ctx->N++;
}

// ============================================================================
// JPEG-LS header/trailer
// ============================================================================

static void write_header(BitWriter *bw, int width, int height, int near_val)
{
    // SOI
    bw_put_raw(bw, 0xFF); bw_put_raw(bw, 0xD8);
    // SOF55 (JPEG-LS)
    bw_put_raw(bw, 0xFF); bw_put_raw(bw, 0xF7);
    bw_put_raw(bw, 0x00); bw_put_raw(bw, 0x0B);
    bw_put_raw(bw, 8);
    bw_put_raw(bw, (uint8_t)(height >> 8)); bw_put_raw(bw, (uint8_t)(height & 0xFF));
    bw_put_raw(bw, (uint8_t)(width >> 8));  bw_put_raw(bw, (uint8_t)(width & 0xFF));
    bw_put_raw(bw, 1); bw_put_raw(bw, 1); bw_put_raw(bw, 0x11); bw_put_raw(bw, 0);

    // SOS
    bw_put_raw(bw, 0xFF); bw_put_raw(bw, 0xDA);
    bw_put_raw(bw, 0x00); bw_put_raw(bw, 0x08);
    bw_put_raw(bw, 1); bw_put_raw(bw, 1); bw_put_raw(bw, 0);
    bw_put_raw(bw, (uint8_t)near_val); bw_put_raw(bw, 0); bw_put_raw(bw, 0);
    bw->free_bits = 8;
}

static void write_eoi(BitWriter *bw)
{
    bw_finalize(bw);
    bw_put_raw(bw, 0xFF);
    bw_put_raw(bw, 0xD9);
}

// ============================================================================
// Main encoding function - proven-correct line buffer structure
//
// Optimizations applied:
// 1. DRAM_ATTR on all lookup tables (zero wait-state)
// 2. Precomputed context+sign LUT (eliminates branches)
// 3. Shift-compare compute_k (no division - Xtensa has no HW divider)
// 4. Batched Golomb-Rice codeword emission
// 5. 64-bit BitWriter accumulator
// 6. 8-byte-at-a-time run scanning
// 7. Internal SRAM allocation for encoder state
// 8. Compiled with -O3
// 9. Lossless skip: no redundant curr[] writes (memcpy already fills them)
// ============================================================================

int IRAM_ATTR jpegls_encode(jpegls_encoder_t *enc,
                  const uint8_t *src, size_t src_len,
                  uint8_t *dst, size_t dst_cap,
                  size_t *out_len,
                  int near)
{
    if (!enc || !src || !dst || !out_len) return -1;
    if ((int)src_len != enc->width * enc->height) return -1;

    const int W = enc->width;
    const int H = enc->height;

    // Compute near-lossless parameters and rebuild quant LUT if needed
    const int nv = near;  // local copy for speed
    if (nv == 0) {
        enc->near_val = 0;
        enc->range = 256;
        enc->qbpp_val = 8;
        rebuild_quant_lut(3, 7, 21, 0);
    } else {
        enc->near_val = nv;
        enc->range = (MAXVAL + 2 * nv) / (2 * nv + 1) + 1;
        // qbpp = ceil(log2(range))
        int r = enc->range, q = 0;
        for (int v = 1; v < r; v <<= 1) q++;
        enc->qbpp_val = q;
        // Rebuild quant LUT with correct thresholds and NEAR deadzone
        int T1, T2, T3;
        compute_default_thresholds(nv, &T1, &T2, &T3);
        rebuild_quant_lut(T1, T2, T3, nv);
    }
    const int range = enc->range;
    const int qbpp_v = enc->qbpp_val;
    const int near_x2p1 = 2 * nv + 1;  // precompute (2*NEAR+1)

    jpegls_encoder_reset(enc);
    bw_init(&enc->bw, dst, dst_cap);
    write_header(&enc->bw, W, H, nv);

    // Line buffers: 1-indexed, index 0 = left border, index W+1 = right border
    uint8_t *prev = enc->prev_line;
    uint8_t *curr = enc->curr_line;
    BitWriter *bw = &enc->bw;

    memset(prev, 0, static_cast<size_t>(W + 2));

    for (int y = 0; y < H; y++) {
        const uint8_t *src_row = src + y * W;

        // Set borders and copy source data into curr line buffer
        curr[0] = prev[1];  // left border = first pixel of previous row
        memcpy(curr + 1, src_row, W);
        curr[W + 1] = curr[W];  // right border = last pixel of current row

        int x = 0;
        while (x < W) {
            int xi = x + 1;  // 1-indexed position
            int Ra = curr[xi - 1];
            int Rb = prev[xi];
            int Rc = prev[xi - 1];
            int Rd = prev[xi + 1];

            int q1 = jpegls_quant_lut[(Rd - Rb) + 255];
            int q2 = jpegls_quant_lut[(Rb - Rc) + 255];
            int q3 = jpegls_quant_lut[(Rc - Ra) + 255];

            if (q1 == 0 && q2 == 0 && q3 == 0) {
                // ====== RUN MODE ======
                int run_val = Ra;
                int remaining = W - x;
                int run_count;
                if (nv == 0) {
                    run_count = scan_run(src_row + x, (uint8_t)run_val, remaining);
                } else {
                    run_count = scan_run_near(src_row + x, (uint8_t)run_val,
                                              nv, remaining);
                    // Reconstruct run pixels as Ra in curr[]
                    for (int i = 0; i < run_count; i++)
                        curr[xi + i] = (uint8_t)run_val;
                }

                bool end_of_line = (x + run_count >= W);

                encode_run_length(bw, run_count, end_of_line ? 1 : 0,
                                  &enc->run_index);

                if (!end_of_line) {
                    int ix = src_row[x + run_count];
                    int pos = x + run_count + 1;  // 1-indexed
                    int Rb_ri = prev[pos];

                    int errval, ctx_idx, ri_type;
                    int ri_sign = 1;  // sign multiplier for reconstruction

                    // T.87 A.7.2.2: use NEAR tolerance, not exact equality
                    if (((Ra - Rb_ri) >= -nv) && ((Ra - Rb_ri) <= nv)) {
                        errval = ix - Ra;
                        ctx_idx = 1;
                        ri_type = 1;
                        ri_sign = 1;
                    } else {
                        int s = (Rb_ri > Ra) ? 1 : -1;
                        errval = (ix - Rb_ri) * s;
                        ctx_idx = 0;
                        ri_type = 0;
                        ri_sign = s;
                    }

                    if (nv > 0) {
                        errval = quantize_error(errval, nv);
                        // Reconstruct with modular wrapping (T.87)
                        int pred = (ri_type == 1) ? Ra : Rb_ri;
                        int Rx = pred + ri_sign * errval * near_x2p1;
                        curr[pos] = (uint8_t)fix_reconstructed(Rx, nv, range);
                    }

                    errval = reduce_error(errval, range);

                    encode_run_interruption(bw, errval,
                                          &enc->run_contexts[ctx_idx],
                                          ri_type, enc->run_index,
                                          qbpp_v);

                    // T.87 / CharLS: decrement run_index after run interruption
                    if (enc->run_index > 0)
                        enc->run_index--;

                    run_count++;
                }

                x += run_count;
            } else {
                // ====== REGULAR MODE ======
                // Context lookup via precomputed LUT (no branches)
                int raw_idx = (q1 + 4) * 81 + (q2 + 4) * 9 + (q3 + 4);
                uint16_t packed = s_ctx_sign_lut[raw_idx];
                int sign = packed >> 15;
                int ctx_id = packed & 0x7FFF;
                RegularContext *ctx = &enc->contexts[ctx_id];

                int Px = med_predictor(Ra, Rb, Rc);
                Px = clamp_val(Px + (sign ? -ctx->C : ctx->C));

                int Ix = src_row[x];

                int errval = Ix - Px;
                if (sign) errval = -errval;

                if (nv > 0) {
                    errval = quantize_error(errval, nv);
                    // Reconstruct with modular wrapping (T.87 / CharLS)
                    int Rx = Px + (sign ? -errval : errval) * near_x2p1;
                    curr[xi] = (uint8_t)fix_reconstructed(Rx, nv, range);
                }

                errval = reduce_error(errval, range);

                int k = compute_k(ctx->A, ctx->N);
                int mapped = map_error_regular(errval, k, ctx, nv);

                encode_mapped_error(bw, mapped, k, LIMIT, qbpp_v);
                update_regular_context(ctx, errval, near_x2p1);

                x++;
            }
        }

        // Update right border with reconstructed value for next row's Rd
        if (nv > 0)
            curr[W + 1] = curr[W];

        // Swap line buffers
        uint8_t *tmp = prev;
        prev = curr;
        curr = tmp;
    }

    enc->prev_line = prev;
    enc->curr_line = curr;

    write_eoi(bw);

    if (enc->bw.overflow) {
        ESP_LOGE(TAG, "Output buffer overflow");
        return -1;
    }

    *out_len = enc->bw.byte_pos;
    return 0;
}
