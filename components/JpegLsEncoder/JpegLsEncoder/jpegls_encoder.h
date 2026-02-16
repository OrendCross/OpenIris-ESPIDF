#pragma once
#ifndef JPEGLS_ENCODER_H
#define JPEGLS_ENCODER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// JPEG-LS encoder for 8-bit grayscale images (ISO/IEC 14495-1 / ITU T.87)
// Optimized for ESP32-S3 with PIE/SIMD acceleration

// Opaque encoder handle
typedef struct jpegls_encoder jpegls_encoder_t;

// Create a JPEG-LS encoder instance
// Returns NULL on allocation failure
// The encoder is reusable across multiple frames with the same dimensions
jpegls_encoder_t *jpegls_encoder_create(int width, int height);

// Destroy the encoder and free all resources
void jpegls_encoder_destroy(jpegls_encoder_t *enc);

// Encode a grayscale 8-bit image to JPEG-LS
// Parameters:
//   enc       - encoder instance (must match width/height from create)
//   src       - source grayscale pixels, row-major, width*height bytes
//   src_len   - must equal width*height
//   dst       - output buffer for JPEG-LS bitstream
//   dst_cap   - capacity of dst buffer in bytes
//   out_len   - [out] actual encoded size in bytes
//   near      - near-lossless parameter (0 = lossless, 1-3 = near-lossless)
//
// Returns 0 on success, -1 on error (buffer too small, etc.)
int jpegls_encode(jpegls_encoder_t *enc,
                  const uint8_t *src, size_t src_len,
                  uint8_t *dst, size_t dst_cap,
                  size_t *out_len,
                  int near);

// Get the maximum possible encoded size for given dimensions
// Useful for allocating output buffers
size_t jpegls_max_encoded_size(int width, int height);

// Reset encoder state (called automatically by jpegls_encode, but
// can be called explicitly to clear context statistics)
void jpegls_encoder_reset(jpegls_encoder_t *enc);

#ifdef __cplusplus
}
#endif

#endif // JPEGLS_ENCODER_H
