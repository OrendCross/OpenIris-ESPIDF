#pragma once
#ifndef JPEGLS_TABLES_H
#define JPEGLS_TABLES_H

#include <stdint.h>
#include "esp_attr.h"

// JPEG-LS (ITU T.87 / ISO 14495-1) precomputed tables for 8-bit grayscale

// Quantization lookup table for 8-bit mode.
// Maps gradient values in range [-255..+255] to quantized values in [-4..+4].
// Index with (gradient + 255) to get the quantized bin.
// Total: 511 entries. Rebuilt at encode time based on NEAR parameter.
// DRAM_ATTR ensures zero-wait-state access from internal SRAM.
extern int8_t jpegls_quant_lut[511];

#endif // JPEGLS_TABLES_H
