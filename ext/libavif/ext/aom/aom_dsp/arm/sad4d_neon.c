/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/arm/sum_neon.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

static INLINE void sad16_neon(uint8x16_t src, uint8x16_t ref,
                              uint32x4_t *const sad_sum) {
  uint8x16_t abs_diff = vabdq_u8(src, ref);
  *sad_sum = vdotq_u32(*sad_sum, abs_diff, vdupq_n_u8(1));
}

static INLINE void sad128xhx4d_neon(const uint8_t *src, int src_stride,
                                    const uint8_t *const ref[4], int ref_stride,
                                    uint32_t res[4], int h) {
  uint32x4_t sum_lo[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };
  uint32x4_t sum_hi[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };

  int i = 0;
  do {
    const uint8x16_t s0 = vld1q_u8(src + i * src_stride);
    sad16_neon(s0, vld1q_u8(ref[0] + i * ref_stride), &sum_lo[0]);
    sad16_neon(s0, vld1q_u8(ref[1] + i * ref_stride), &sum_lo[1]);
    sad16_neon(s0, vld1q_u8(ref[2] + i * ref_stride), &sum_lo[2]);
    sad16_neon(s0, vld1q_u8(ref[3] + i * ref_stride), &sum_lo[3]);

    const uint8x16_t s1 = vld1q_u8(src + i * src_stride + 16);
    sad16_neon(s1, vld1q_u8(ref[0] + i * ref_stride + 16), &sum_hi[0]);
    sad16_neon(s1, vld1q_u8(ref[1] + i * ref_stride + 16), &sum_hi[1]);
    sad16_neon(s1, vld1q_u8(ref[2] + i * ref_stride + 16), &sum_hi[2]);
    sad16_neon(s1, vld1q_u8(ref[3] + i * ref_stride + 16), &sum_hi[3]);

    const uint8x16_t s2 = vld1q_u8(src + i * src_stride + 32);
    sad16_neon(s2, vld1q_u8(ref[0] + i * ref_stride + 32), &sum_lo[0]);
    sad16_neon(s2, vld1q_u8(ref[1] + i * ref_stride + 32), &sum_lo[1]);
    sad16_neon(s2, vld1q_u8(ref[2] + i * ref_stride + 32), &sum_lo[2]);
    sad16_neon(s2, vld1q_u8(ref[3] + i * ref_stride + 32), &sum_lo[3]);

    const uint8x16_t s3 = vld1q_u8(src + i * src_stride + 48);
    sad16_neon(s3, vld1q_u8(ref[0] + i * ref_stride + 48), &sum_hi[0]);
    sad16_neon(s3, vld1q_u8(ref[1] + i * ref_stride + 48), &sum_hi[1]);
    sad16_neon(s3, vld1q_u8(ref[2] + i * ref_stride + 48), &sum_hi[2]);
    sad16_neon(s3, vld1q_u8(ref[3] + i * ref_stride + 48), &sum_hi[3]);

    const uint8x16_t s4 = vld1q_u8(src + i * src_stride + 64);
    sad16_neon(s4, vld1q_u8(ref[0] + i * ref_stride + 64), &sum_lo[0]);
    sad16_neon(s4, vld1q_u8(ref[1] + i * ref_stride + 64), &sum_lo[1]);
    sad16_neon(s4, vld1q_u8(ref[2] + i * ref_stride + 64), &sum_lo[2]);
    sad16_neon(s4, vld1q_u8(ref[3] + i * ref_stride + 64), &sum_lo[3]);

    const uint8x16_t s5 = vld1q_u8(src + i * src_stride + 80);
    sad16_neon(s5, vld1q_u8(ref[0] + i * ref_stride + 80), &sum_hi[0]);
    sad16_neon(s5, vld1q_u8(ref[1] + i * ref_stride + 80), &sum_hi[1]);
    sad16_neon(s5, vld1q_u8(ref[2] + i * ref_stride + 80), &sum_hi[2]);
    sad16_neon(s5, vld1q_u8(ref[3] + i * ref_stride + 80), &sum_hi[3]);

    const uint8x16_t s6 = vld1q_u8(src + i * src_stride + 96);
    sad16_neon(s6, vld1q_u8(ref[0] + i * ref_stride + 96), &sum_lo[0]);
    sad16_neon(s6, vld1q_u8(ref[1] + i * ref_stride + 96), &sum_lo[1]);
    sad16_neon(s6, vld1q_u8(ref[2] + i * ref_stride + 96), &sum_lo[2]);
    sad16_neon(s6, vld1q_u8(ref[3] + i * ref_stride + 96), &sum_lo[3]);

    const uint8x16_t s7 = vld1q_u8(src + i * src_stride + 112);
    sad16_neon(s7, vld1q_u8(ref[0] + i * ref_stride + 112), &sum_hi[0]);
    sad16_neon(s7, vld1q_u8(ref[1] + i * ref_stride + 112), &sum_hi[1]);
    sad16_neon(s7, vld1q_u8(ref[2] + i * ref_stride + 112), &sum_hi[2]);
    sad16_neon(s7, vld1q_u8(ref[3] + i * ref_stride + 112), &sum_hi[3]);

    i++;
  } while (i < h);

  uint32x4_t res0 = vpaddq_u32(vaddq_u32(sum_lo[0], sum_hi[0]),
                               vaddq_u32(sum_lo[1], sum_hi[1]));
  uint32x4_t res1 = vpaddq_u32(vaddq_u32(sum_lo[2], sum_hi[2]),
                               vaddq_u32(sum_lo[3], sum_hi[3]));
  vst1q_u32(res, vpaddq_u32(res0, res1));
}

static INLINE void sad64xhx4d_neon(const uint8_t *src, int src_stride,
                                   const uint8_t *const ref[4], int ref_stride,
                                   uint32_t res[4], int h) {
  uint32x4_t sum_lo[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };
  uint32x4_t sum_hi[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };

  int i = 0;
  do {
    const uint8x16_t s0 = vld1q_u8(src + i * src_stride);
    sad16_neon(s0, vld1q_u8(ref[0] + i * ref_stride), &sum_lo[0]);
    sad16_neon(s0, vld1q_u8(ref[1] + i * ref_stride), &sum_lo[1]);
    sad16_neon(s0, vld1q_u8(ref[2] + i * ref_stride), &sum_lo[2]);
    sad16_neon(s0, vld1q_u8(ref[3] + i * ref_stride), &sum_lo[3]);

    const uint8x16_t s1 = vld1q_u8(src + i * src_stride + 16);
    sad16_neon(s1, vld1q_u8(ref[0] + i * ref_stride + 16), &sum_hi[0]);
    sad16_neon(s1, vld1q_u8(ref[1] + i * ref_stride + 16), &sum_hi[1]);
    sad16_neon(s1, vld1q_u8(ref[2] + i * ref_stride + 16), &sum_hi[2]);
    sad16_neon(s1, vld1q_u8(ref[3] + i * ref_stride + 16), &sum_hi[3]);

    const uint8x16_t s2 = vld1q_u8(src + i * src_stride + 32);
    sad16_neon(s2, vld1q_u8(ref[0] + i * ref_stride + 32), &sum_lo[0]);
    sad16_neon(s2, vld1q_u8(ref[1] + i * ref_stride + 32), &sum_lo[1]);
    sad16_neon(s2, vld1q_u8(ref[2] + i * ref_stride + 32), &sum_lo[2]);
    sad16_neon(s2, vld1q_u8(ref[3] + i * ref_stride + 32), &sum_lo[3]);

    const uint8x16_t s3 = vld1q_u8(src + i * src_stride + 48);
    sad16_neon(s3, vld1q_u8(ref[0] + i * ref_stride + 48), &sum_hi[0]);
    sad16_neon(s3, vld1q_u8(ref[1] + i * ref_stride + 48), &sum_hi[1]);
    sad16_neon(s3, vld1q_u8(ref[2] + i * ref_stride + 48), &sum_hi[2]);
    sad16_neon(s3, vld1q_u8(ref[3] + i * ref_stride + 48), &sum_hi[3]);

    i++;
  } while (i < h);

  uint32x4_t res0 = vpaddq_u32(vaddq_u32(sum_lo[0], sum_hi[0]),
                               vaddq_u32(sum_lo[1], sum_hi[1]));
  uint32x4_t res1 = vpaddq_u32(vaddq_u32(sum_lo[2], sum_hi[2]),
                               vaddq_u32(sum_lo[3], sum_hi[3]));
  vst1q_u32(res, vpaddq_u32(res0, res1));
}

static INLINE void sad32xhx4d_neon(const uint8_t *src, int src_stride,
                                   const uint8_t *const ref[4], int ref_stride,
                                   uint32_t res[4], int h) {
  uint32x4_t sum_lo[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };
  uint32x4_t sum_hi[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };

  int i = 0;
  do {
    const uint8x16_t s0 = vld1q_u8(src + i * src_stride);
    sad16_neon(s0, vld1q_u8(ref[0] + i * ref_stride), &sum_lo[0]);
    sad16_neon(s0, vld1q_u8(ref[1] + i * ref_stride), &sum_lo[1]);
    sad16_neon(s0, vld1q_u8(ref[2] + i * ref_stride), &sum_lo[2]);
    sad16_neon(s0, vld1q_u8(ref[3] + i * ref_stride), &sum_lo[3]);

    const uint8x16_t s1 = vld1q_u8(src + i * src_stride + 16);
    sad16_neon(s1, vld1q_u8(ref[0] + i * ref_stride + 16), &sum_hi[0]);
    sad16_neon(s1, vld1q_u8(ref[1] + i * ref_stride + 16), &sum_hi[1]);
    sad16_neon(s1, vld1q_u8(ref[2] + i * ref_stride + 16), &sum_hi[2]);
    sad16_neon(s1, vld1q_u8(ref[3] + i * ref_stride + 16), &sum_hi[3]);

    i++;
  } while (i < h);

  uint32x4_t res0 = vpaddq_u32(vaddq_u32(sum_lo[0], sum_hi[0]),
                               vaddq_u32(sum_lo[1], sum_hi[1]));
  uint32x4_t res1 = vpaddq_u32(vaddq_u32(sum_lo[2], sum_hi[2]),
                               vaddq_u32(sum_lo[3], sum_hi[3]));
  vst1q_u32(res, vpaddq_u32(res0, res1));
}

static INLINE void sad16xhx4d_neon(const uint8_t *src, int src_stride,
                                   const uint8_t *const ref[4], int ref_stride,
                                   uint32_t res[4], int h) {
  uint32x4_t sum[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                        vdupq_n_u32(0) };

  int i = 0;
  do {
    const uint8x16_t s = vld1q_u8(src + i * src_stride);
    sad16_neon(s, vld1q_u8(ref[0] + i * ref_stride), &sum[0]);
    sad16_neon(s, vld1q_u8(ref[1] + i * ref_stride), &sum[1]);
    sad16_neon(s, vld1q_u8(ref[2] + i * ref_stride), &sum[2]);
    sad16_neon(s, vld1q_u8(ref[3] + i * ref_stride), &sum[3]);

    i++;
  } while (i < h);

  uint32x4_t res0 = vpaddq_u32(sum[0], sum[1]);
  uint32x4_t res1 = vpaddq_u32(sum[2], sum[3]);
  vst1q_u32(res, vpaddq_u32(res0, res1));
}

#else  // !(defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD))

static INLINE void sad16_neon(uint8x16_t src, uint8x16_t ref,
                              uint16x8_t *const sad_sum) {
  uint8x16_t abs_diff = vabdq_u8(src, ref);
  *sad_sum = vpadalq_u8(*sad_sum, abs_diff);
}

static INLINE void sad128xhx4d_neon(const uint8_t *src, int src_stride,
                                    const uint8_t *const ref[4], int ref_stride,
                                    uint32_t res[4], int h) {
  vst1q_u32(res, vdupq_n_u32(0));
  int h_tmp = h > 32 ? 32 : h;

  int i = 0;
  do {
    uint16x8_t sum_lo[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                             vdupq_n_u16(0) };
    uint16x8_t sum_hi[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                             vdupq_n_u16(0) };

    do {
      const uint8x16_t s0 = vld1q_u8(src + i * src_stride);
      sad16_neon(s0, vld1q_u8(ref[0] + i * ref_stride), &sum_lo[0]);
      sad16_neon(s0, vld1q_u8(ref[1] + i * ref_stride), &sum_lo[1]);
      sad16_neon(s0, vld1q_u8(ref[2] + i * ref_stride), &sum_lo[2]);
      sad16_neon(s0, vld1q_u8(ref[3] + i * ref_stride), &sum_lo[3]);

      const uint8x16_t s1 = vld1q_u8(src + i * src_stride + 16);
      sad16_neon(s1, vld1q_u8(ref[0] + i * ref_stride + 16), &sum_hi[0]);
      sad16_neon(s1, vld1q_u8(ref[1] + i * ref_stride + 16), &sum_hi[1]);
      sad16_neon(s1, vld1q_u8(ref[2] + i * ref_stride + 16), &sum_hi[2]);
      sad16_neon(s1, vld1q_u8(ref[3] + i * ref_stride + 16), &sum_hi[3]);

      const uint8x16_t s2 = vld1q_u8(src + i * src_stride + 32);
      sad16_neon(s2, vld1q_u8(ref[0] + i * ref_stride + 32), &sum_lo[0]);
      sad16_neon(s2, vld1q_u8(ref[1] + i * ref_stride + 32), &sum_lo[1]);
      sad16_neon(s2, vld1q_u8(ref[2] + i * ref_stride + 32), &sum_lo[2]);
      sad16_neon(s2, vld1q_u8(ref[3] + i * ref_stride + 32), &sum_lo[3]);

      const uint8x16_t s3 = vld1q_u8(src + i * src_stride + 48);
      sad16_neon(s3, vld1q_u8(ref[0] + i * ref_stride + 48), &sum_hi[0]);
      sad16_neon(s3, vld1q_u8(ref[1] + i * ref_stride + 48), &sum_hi[1]);
      sad16_neon(s3, vld1q_u8(ref[2] + i * ref_stride + 48), &sum_hi[2]);
      sad16_neon(s3, vld1q_u8(ref[3] + i * ref_stride + 48), &sum_hi[3]);

      const uint8x16_t s4 = vld1q_u8(src + i * src_stride + 64);
      sad16_neon(s4, vld1q_u8(ref[0] + i * ref_stride + 64), &sum_lo[0]);
      sad16_neon(s4, vld1q_u8(ref[1] + i * ref_stride + 64), &sum_lo[1]);
      sad16_neon(s4, vld1q_u8(ref[2] + i * ref_stride + 64), &sum_lo[2]);
      sad16_neon(s4, vld1q_u8(ref[3] + i * ref_stride + 64), &sum_lo[3]);

      const uint8x16_t s5 = vld1q_u8(src + i * src_stride + 80);
      sad16_neon(s5, vld1q_u8(ref[0] + i * ref_stride + 80), &sum_hi[0]);
      sad16_neon(s5, vld1q_u8(ref[1] + i * ref_stride + 80), &sum_hi[1]);
      sad16_neon(s5, vld1q_u8(ref[2] + i * ref_stride + 80), &sum_hi[2]);
      sad16_neon(s5, vld1q_u8(ref[3] + i * ref_stride + 80), &sum_hi[3]);

      const uint8x16_t s6 = vld1q_u8(src + i * src_stride + 96);
      sad16_neon(s6, vld1q_u8(ref[0] + i * ref_stride + 96), &sum_lo[0]);
      sad16_neon(s6, vld1q_u8(ref[1] + i * ref_stride + 96), &sum_lo[1]);
      sad16_neon(s6, vld1q_u8(ref[2] + i * ref_stride + 96), &sum_lo[2]);
      sad16_neon(s6, vld1q_u8(ref[3] + i * ref_stride + 96), &sum_lo[3]);

      const uint8x16_t s7 = vld1q_u8(src + i * src_stride + 112);
      sad16_neon(s7, vld1q_u8(ref[0] + i * ref_stride + 112), &sum_hi[0]);
      sad16_neon(s7, vld1q_u8(ref[1] + i * ref_stride + 112), &sum_hi[1]);
      sad16_neon(s7, vld1q_u8(ref[2] + i * ref_stride + 112), &sum_hi[2]);
      sad16_neon(s7, vld1q_u8(ref[3] + i * ref_stride + 112), &sum_hi[3]);

      i++;
    } while (i < h_tmp);

    res[0] += horizontal_long_add_u16x8(sum_lo[0], sum_hi[0]);
    res[1] += horizontal_long_add_u16x8(sum_lo[1], sum_hi[1]);
    res[2] += horizontal_long_add_u16x8(sum_lo[2], sum_hi[2]);
    res[3] += horizontal_long_add_u16x8(sum_lo[3], sum_hi[3]);

    h_tmp += 32;
  } while (i < h);
}

static INLINE void sad64xhx4d_neon(const uint8_t *src, int src_stride,
                                   const uint8_t *const ref[4], int ref_stride,
                                   uint32_t res[4], int h) {
  vst1q_u32(res, vdupq_n_u32(0));
  int h_tmp = h > 64 ? 64 : h;

  int i = 0;
  do {
    uint16x8_t sum_lo[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                             vdupq_n_u16(0) };
    uint16x8_t sum_hi[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                             vdupq_n_u16(0) };

    do {
      const uint8x16_t s0 = vld1q_u8(src + i * src_stride);
      sad16_neon(s0, vld1q_u8(ref[0] + i * ref_stride), &sum_lo[0]);
      sad16_neon(s0, vld1q_u8(ref[1] + i * ref_stride), &sum_lo[1]);
      sad16_neon(s0, vld1q_u8(ref[2] + i * ref_stride), &sum_lo[2]);
      sad16_neon(s0, vld1q_u8(ref[3] + i * ref_stride), &sum_lo[3]);

      const uint8x16_t s1 = vld1q_u8(src + i * src_stride + 16);
      sad16_neon(s1, vld1q_u8(ref[0] + i * ref_stride + 16), &sum_hi[0]);
      sad16_neon(s1, vld1q_u8(ref[1] + i * ref_stride + 16), &sum_hi[1]);
      sad16_neon(s1, vld1q_u8(ref[2] + i * ref_stride + 16), &sum_hi[2]);
      sad16_neon(s1, vld1q_u8(ref[3] + i * ref_stride + 16), &sum_hi[3]);

      const uint8x16_t s2 = vld1q_u8(src + i * src_stride + 32);
      sad16_neon(s2, vld1q_u8(ref[0] + i * ref_stride + 32), &sum_lo[0]);
      sad16_neon(s2, vld1q_u8(ref[1] + i * ref_stride + 32), &sum_lo[1]);
      sad16_neon(s2, vld1q_u8(ref[2] + i * ref_stride + 32), &sum_lo[2]);
      sad16_neon(s2, vld1q_u8(ref[3] + i * ref_stride + 32), &sum_lo[3]);

      const uint8x16_t s3 = vld1q_u8(src + i * src_stride + 48);
      sad16_neon(s3, vld1q_u8(ref[0] + i * ref_stride + 48), &sum_hi[0]);
      sad16_neon(s3, vld1q_u8(ref[1] + i * ref_stride + 48), &sum_hi[1]);
      sad16_neon(s3, vld1q_u8(ref[2] + i * ref_stride + 48), &sum_hi[2]);
      sad16_neon(s3, vld1q_u8(ref[3] + i * ref_stride + 48), &sum_hi[3]);

      i++;
    } while (i < h_tmp);

    res[0] += horizontal_long_add_u16x8(sum_lo[0], sum_hi[0]);
    res[1] += horizontal_long_add_u16x8(sum_lo[1], sum_hi[1]);
    res[2] += horizontal_long_add_u16x8(sum_lo[2], sum_hi[2]);
    res[3] += horizontal_long_add_u16x8(sum_lo[3], sum_hi[3]);

    h_tmp += 64;
  } while (i < h);
}

static INLINE void sad32xhx4d_neon(const uint8_t *src, int src_stride,
                                   const uint8_t *const ref[4], int ref_stride,
                                   uint32_t res[4], int h) {
  uint16x8_t sum_lo[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                           vdupq_n_u16(0) };
  uint16x8_t sum_hi[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                           vdupq_n_u16(0) };

  int i = 0;
  do {
    const uint8x16_t s0 = vld1q_u8(src + i * src_stride);
    sad16_neon(s0, vld1q_u8(ref[0] + i * ref_stride), &sum_lo[0]);
    sad16_neon(s0, vld1q_u8(ref[1] + i * ref_stride), &sum_lo[1]);
    sad16_neon(s0, vld1q_u8(ref[2] + i * ref_stride), &sum_lo[2]);
    sad16_neon(s0, vld1q_u8(ref[3] + i * ref_stride), &sum_lo[3]);

    const uint8x16_t s1 = vld1q_u8(src + i * src_stride + 16);
    sad16_neon(s1, vld1q_u8(ref[0] + i * ref_stride + 16), &sum_hi[0]);
    sad16_neon(s1, vld1q_u8(ref[1] + i * ref_stride + 16), &sum_hi[1]);
    sad16_neon(s1, vld1q_u8(ref[2] + i * ref_stride + 16), &sum_hi[2]);
    sad16_neon(s1, vld1q_u8(ref[3] + i * ref_stride + 16), &sum_hi[3]);

    i++;
  } while (i < h);

  res[0] = horizontal_long_add_u16x8(sum_lo[0], sum_hi[0]);
  res[1] = horizontal_long_add_u16x8(sum_lo[1], sum_hi[1]);
  res[2] = horizontal_long_add_u16x8(sum_lo[2], sum_hi[2]);
  res[3] = horizontal_long_add_u16x8(sum_lo[3], sum_hi[3]);
}

static INLINE void sad16xhx4d_neon(const uint8_t *src, int src_stride,
                                   const uint8_t *const ref[4], int ref_stride,
                                   uint32_t res[4], int h) {
  uint16x8_t sum[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                        vdupq_n_u16(0) };

  int i = 0;
  do {
    const uint8x16_t s = vld1q_u8(src + i * src_stride);
    sad16_neon(s, vld1q_u8(ref[0] + i * ref_stride), &sum[0]);
    sad16_neon(s, vld1q_u8(ref[1] + i * ref_stride), &sum[1]);
    sad16_neon(s, vld1q_u8(ref[2] + i * ref_stride), &sum[2]);
    sad16_neon(s, vld1q_u8(ref[3] + i * ref_stride), &sum[3]);

    i++;
  } while (i < h);

  res[0] = horizontal_add_u16x8(sum[0]);
  res[1] = horizontal_add_u16x8(sum[1]);
  res[2] = horizontal_add_u16x8(sum[2]);
  res[3] = horizontal_add_u16x8(sum[3]);
}

#endif  // defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

static INLINE void sad8_neon(uint8x8_t src, uint8x8_t ref,
                             uint16x8_t *const sad_sum) {
  uint8x8_t abs_diff = vabd_u8(src, ref);
  *sad_sum = vaddw_u8(*sad_sum, abs_diff);
}

static INLINE void sad8xhx4d_neon(const uint8_t *src, int src_stride,
                                  const uint8_t *const ref[4], int ref_stride,
                                  uint32_t res[4], int h) {
  uint16x8_t sum[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                        vdupq_n_u16(0) };

  int i = 0;
  do {
    const uint8x8_t s = vld1_u8(src + i * src_stride);
    sad8_neon(s, vld1_u8(ref[0] + i * ref_stride), &sum[0]);
    sad8_neon(s, vld1_u8(ref[1] + i * ref_stride), &sum[1]);
    sad8_neon(s, vld1_u8(ref[2] + i * ref_stride), &sum[2]);
    sad8_neon(s, vld1_u8(ref[3] + i * ref_stride), &sum[3]);

    i++;
  } while (i < h);

  res[0] = horizontal_add_u16x8(sum[0]);
  res[1] = horizontal_add_u16x8(sum[1]);
  res[2] = horizontal_add_u16x8(sum[2]);
  res[3] = horizontal_add_u16x8(sum[3]);
}

static INLINE void sad4xhx4d_neon(const uint8_t *src, int src_stride,
                                  const uint8_t *const ref[4], int ref_stride,
                                  uint32_t res[4], int h) {
  uint16x8_t sum[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                        vdupq_n_u16(0) };

  int i = 0;
  do {
    uint32x2_t s, r0, r1, r2, r3;
    uint32_t s_lo, s_hi, r0_lo, r0_hi, r1_lo, r1_hi, r2_lo, r2_hi, r3_lo, r3_hi;

    memcpy(&s_lo, src + i * src_stride, 4);
    memcpy(&r0_lo, ref[0] + i * ref_stride, 4);
    memcpy(&r1_lo, ref[1] + i * ref_stride, 4);
    memcpy(&r2_lo, ref[2] + i * ref_stride, 4);
    memcpy(&r3_lo, ref[3] + i * ref_stride, 4);
    s = vdup_n_u32(s_lo);
    r0 = vdup_n_u32(r0_lo);
    r1 = vdup_n_u32(r1_lo);
    r2 = vdup_n_u32(r2_lo);
    r3 = vdup_n_u32(r3_lo);

    memcpy(&s_hi, src + (i + 1) * src_stride, 4);
    memcpy(&r0_hi, ref[0] + (i + 1) * ref_stride, 4);
    memcpy(&r1_hi, ref[1] + (i + 1) * ref_stride, 4);
    memcpy(&r2_hi, ref[2] + (i + 1) * ref_stride, 4);
    memcpy(&r3_hi, ref[3] + (i + 1) * ref_stride, 4);
    s = vset_lane_u32(s_hi, s, 1);
    r0 = vset_lane_u32(r0_hi, r0, 1);
    r1 = vset_lane_u32(r1_hi, r1, 1);
    r2 = vset_lane_u32(r2_hi, r2, 1);
    r3 = vset_lane_u32(r3_hi, r3, 1);

    sad8_neon(vreinterpret_u8_u32(s), vreinterpret_u8_u32(r0), &sum[0]);
    sad8_neon(vreinterpret_u8_u32(s), vreinterpret_u8_u32(r1), &sum[1]);
    sad8_neon(vreinterpret_u8_u32(s), vreinterpret_u8_u32(r2), &sum[2]);
    sad8_neon(vreinterpret_u8_u32(s), vreinterpret_u8_u32(r3), &sum[3]);

    i += 2;
  } while (i < h);

  res[0] = horizontal_add_u16x8(sum[0]);
  res[1] = horizontal_add_u16x8(sum[1]);
  res[2] = horizontal_add_u16x8(sum[2]);
  res[3] = horizontal_add_u16x8(sum[3]);
}

#define SAD_WXH_4D_NEON(w, h)                                                  \
  void aom_sad##w##x##h##x4d_neon(const uint8_t *src, int src_stride,          \
                                  const uint8_t *const ref[4], int ref_stride, \
                                  uint32_t res[4]) {                           \
    sad##w##xhx4d_neon(src, src_stride, ref, ref_stride, res, (h));            \
  }

SAD_WXH_4D_NEON(4, 4)
SAD_WXH_4D_NEON(4, 8)
SAD_WXH_4D_NEON(4, 16)
SAD_WXH_4D_NEON(4, 32)

SAD_WXH_4D_NEON(8, 4)
SAD_WXH_4D_NEON(8, 8)
SAD_WXH_4D_NEON(8, 16)
SAD_WXH_4D_NEON(8, 32)

SAD_WXH_4D_NEON(16, 4)
SAD_WXH_4D_NEON(16, 8)
SAD_WXH_4D_NEON(16, 16)
SAD_WXH_4D_NEON(16, 32)
SAD_WXH_4D_NEON(16, 64)

SAD_WXH_4D_NEON(32, 8)
SAD_WXH_4D_NEON(32, 16)
SAD_WXH_4D_NEON(32, 32)
SAD_WXH_4D_NEON(32, 64)

SAD_WXH_4D_NEON(64, 16)
SAD_WXH_4D_NEON(64, 32)
SAD_WXH_4D_NEON(64, 64)
SAD_WXH_4D_NEON(64, 128)

SAD_WXH_4D_NEON(128, 64)
SAD_WXH_4D_NEON(128, 128)

#undef SAD_WXH_4D_NEON

#define SAD_SKIP_WXH_4D_NEON(w, h)                                          \
  void aom_sad_skip_##w##x##h##x4d_neon(const uint8_t *src, int src_stride, \
                                        const uint8_t *const ref[4],        \
                                        int ref_stride, uint32_t res[4]) {  \
    sad##w##xhx4d_neon(src, 2 * src_stride, ref, 2 * ref_stride, res,       \
                       ((h) >> 1));                                         \
    res[0] <<= 1;                                                           \
    res[1] <<= 1;                                                           \
    res[2] <<= 1;                                                           \
    res[3] <<= 1;                                                           \
  }

SAD_SKIP_WXH_4D_NEON(4, 8)
SAD_SKIP_WXH_4D_NEON(4, 16)
SAD_SKIP_WXH_4D_NEON(4, 32)

SAD_SKIP_WXH_4D_NEON(8, 8)
SAD_SKIP_WXH_4D_NEON(8, 16)
SAD_SKIP_WXH_4D_NEON(8, 32)

SAD_SKIP_WXH_4D_NEON(16, 8)
SAD_SKIP_WXH_4D_NEON(16, 16)
SAD_SKIP_WXH_4D_NEON(16, 32)
SAD_SKIP_WXH_4D_NEON(16, 64)

SAD_SKIP_WXH_4D_NEON(32, 8)
SAD_SKIP_WXH_4D_NEON(32, 16)
SAD_SKIP_WXH_4D_NEON(32, 32)
SAD_SKIP_WXH_4D_NEON(32, 64)

SAD_SKIP_WXH_4D_NEON(64, 16)
SAD_SKIP_WXH_4D_NEON(64, 32)
SAD_SKIP_WXH_4D_NEON(64, 64)
SAD_SKIP_WXH_4D_NEON(64, 128)

SAD_SKIP_WXH_4D_NEON(128, 64)
SAD_SKIP_WXH_4D_NEON(128, 128)

#undef SAD_SKIP_WXH_4D_NEON
