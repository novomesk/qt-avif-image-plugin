/*
 *
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <arm_neon.h>

#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_dsp/arm/transpose_neon.h"
#include "aom_ports/mem.h"
#include "av1/common/convolve.h"
#include "av1/common/filter.h"
#include "av1/common/arm/convolve_neon.h"

static INLINE int16x4_t convolve8_4x4(const int16x4_t s0, const int16x4_t s1,
                                      const int16x4_t s2, const int16x4_t s3,
                                      const int16x4_t s4, const int16x4_t s5,
                                      const int16x4_t s6, const int16x4_t s7,
                                      const int16x8_t filter) {
  const int16x4_t filter_lo = vget_low_s16(filter);
  const int16x4_t filter_hi = vget_high_s16(filter);
  int16x4_t sum;

  sum = vmul_lane_s16(s0, filter_lo, 0);
  sum = vmla_lane_s16(sum, s1, filter_lo, 1);
  sum = vmla_lane_s16(sum, s2, filter_lo, 2);
  sum = vmla_lane_s16(sum, s3, filter_lo, 3);
  sum = vmla_lane_s16(sum, s4, filter_hi, 0);
  sum = vmla_lane_s16(sum, s5, filter_hi, 1);
  sum = vmla_lane_s16(sum, s6, filter_hi, 2);
  sum = vmla_lane_s16(sum, s7, filter_hi, 3);

  return sum;
}

#if !defined(__aarch64__)
static INLINE uint8x8_t convolve8_horiz_4x1(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
    const int16x4_t s6, const int16x4_t s7, const int16x8_t filter,
    const int16x4_t shift_round_0, const int16x4_t shift_by_bits) {
  const int16x4_t filter_lo = vget_low_s16(filter);
  const int16x4_t filter_hi = vget_high_s16(filter);
  int16x4_t sum;

  sum = vmul_lane_s16(s0, filter_lo, 0);
  sum = vmla_lane_s16(sum, s1, filter_lo, 1);
  sum = vmla_lane_s16(sum, s2, filter_lo, 2);
  sum = vmla_lane_s16(sum, s3, filter_lo, 3);
  sum = vmla_lane_s16(sum, s4, filter_hi, 0);
  sum = vmla_lane_s16(sum, s5, filter_hi, 1);
  sum = vmla_lane_s16(sum, s6, filter_hi, 2);
  sum = vmla_lane_s16(sum, s7, filter_hi, 3);

  sum = vqrshl_s16(sum, shift_round_0);
  sum = vqrshl_s16(sum, shift_by_bits);

  return vqmovun_s16(vcombine_s16(sum, sum));
}
#endif  // !defined(__arch64__)

static INLINE uint8x8_t convolve8_vert_8x4(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
    const int16x8_t s6, const int16x8_t s7, const int16x8_t filter) {
  const int16x4_t filter_lo = vget_low_s16(filter);
  const int16x4_t filter_hi = vget_high_s16(filter);
  int16x8_t sum;

  sum = vmulq_lane_s16(s0, filter_lo, 0);
  sum = vmlaq_lane_s16(sum, s1, filter_lo, 1);
  sum = vmlaq_lane_s16(sum, s2, filter_lo, 2);
  sum = vmlaq_lane_s16(sum, s3, filter_lo, 3);
  sum = vmlaq_lane_s16(sum, s4, filter_hi, 0);
  sum = vmlaq_lane_s16(sum, s5, filter_hi, 1);
  sum = vmlaq_lane_s16(sum, s6, filter_hi, 2);
  sum = vmlaq_lane_s16(sum, s7, filter_hi, 3);

  return vqrshrun_n_s16(sum, FILTER_BITS - 1);
}

static INLINE int16x4_t convolve8_vert_4x4_s32(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
    const int16x4_t s6, const int16x4_t s7, const int16x8_t y_filter,
    const int32x4_t round_shift_vec, const int32x4_t offset_const,
    const int32x4_t sub_const_vec) {
  const int16x4_t y_filter_lo = vget_low_s16(y_filter);
  const int16x4_t y_filter_hi = vget_high_s16(y_filter);
  int32x4_t sum;

  sum = vmull_lane_s16(s0, y_filter_lo, 0);
  sum = vmlal_lane_s16(sum, s1, y_filter_lo, 1);
  sum = vmlal_lane_s16(sum, s2, y_filter_lo, 2);
  sum = vmlal_lane_s16(sum, s3, y_filter_lo, 3);
  sum = vmlal_lane_s16(sum, s4, y_filter_hi, 0);
  sum = vmlal_lane_s16(sum, s5, y_filter_hi, 1);
  sum = vmlal_lane_s16(sum, s6, y_filter_hi, 2);
  sum = vmlal_lane_s16(sum, s7, y_filter_hi, 3);

  sum = vaddq_s32(sum, offset_const);
  sum = vqrshlq_s32(sum, round_shift_vec);
  sum = vsubq_s32(sum, sub_const_vec);

  return vmovn_s32(sum);
}

static INLINE uint8x8_t convolve8_vert_8x4_s32(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
    const int16x8_t s6, const int16x8_t s7, const int16x8_t y_filter,
    const int32x4_t round_shift_vec, const int32x4_t offset_const,
    const int32x4_t sub_const_vec, const int16x8_t vec_round_bits) {
  const int16x4_t y_filter_lo = vget_low_s16(y_filter);
  const int16x4_t y_filter_hi = vget_high_s16(y_filter);
  int32x4_t sum0, sum1;
  int16x8_t res;

  sum0 = vmull_lane_s16(vget_low_s16(s0), y_filter_lo, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s1), y_filter_lo, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s2), y_filter_lo, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s3), y_filter_lo, 3);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s4), y_filter_hi, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s5), y_filter_hi, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s6), y_filter_hi, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s7), y_filter_hi, 3);

  sum1 = vmull_lane_s16(vget_high_s16(s0), y_filter_lo, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s1), y_filter_lo, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s2), y_filter_lo, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s3), y_filter_lo, 3);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s4), y_filter_hi, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s5), y_filter_hi, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s6), y_filter_hi, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s7), y_filter_hi, 3);

  sum0 = vaddq_s32(sum0, offset_const);
  sum1 = vaddq_s32(sum1, offset_const);
  sum0 = vqrshlq_s32(sum0, round_shift_vec);
  sum1 = vqrshlq_s32(sum1, round_shift_vec);
  sum0 = vsubq_s32(sum0, sub_const_vec);
  sum1 = vsubq_s32(sum1, sub_const_vec);

  res = vcombine_s16(vmovn_s32(sum0), vmovn_s32(sum1));
  res = vqrshlq_s16(res, vec_round_bits);

  return vqmovun_s16(res);
}

static INLINE int16x4_t convolve12_vert_4x4_s32(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
    const int16x4_t s6, const int16x4_t s7, const int16x4_t s8,
    const int16x4_t s9, const int16x4_t s10, const int16x4_t s11,
    const int16x8_t y_filter_0_7, const int16x4_t y_filter_8_11,
    const int32x4_t round_shift_vec, const int32x4_t offset_const,
    const int32x4_t sub_const_vec) {
  const int16x4_t y_filter_0_3 = vget_low_s16(y_filter_0_7);
  const int16x4_t y_filter_4_7 = vget_high_s16(y_filter_0_7);
  int32x4_t sum;

  sum = vmull_lane_s16(s0, y_filter_0_3, 0);
  sum = vmlal_lane_s16(sum, s1, y_filter_0_3, 1);
  sum = vmlal_lane_s16(sum, s2, y_filter_0_3, 2);
  sum = vmlal_lane_s16(sum, s3, y_filter_0_3, 3);
  sum = vmlal_lane_s16(sum, s4, y_filter_4_7, 0);
  sum = vmlal_lane_s16(sum, s5, y_filter_4_7, 1);
  sum = vmlal_lane_s16(sum, s6, y_filter_4_7, 2);
  sum = vmlal_lane_s16(sum, s7, y_filter_4_7, 3);
  sum = vmlal_lane_s16(sum, s8, y_filter_8_11, 0);
  sum = vmlal_lane_s16(sum, s9, y_filter_8_11, 1);
  sum = vmlal_lane_s16(sum, s10, y_filter_8_11, 2);
  sum = vmlal_lane_s16(sum, s11, y_filter_8_11, 3);

  sum = vaddq_s32(sum, offset_const);
  sum = vqrshlq_s32(sum, round_shift_vec);
  sum = vsubq_s32(sum, sub_const_vec);

  return vmovn_s32(sum);
}

static INLINE uint8x8_t convolve12_vert_8x4_s32(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
    const int16x8_t s6, const int16x8_t s7, const int16x8_t s8,
    const int16x8_t s9, const int16x8_t s10, const int16x8_t s11,
    const int16x8_t y_filter_0_7, const int16x4_t y_filter_8_11,
    const int32x4_t round_shift_vec, const int32x4_t offset_const,
    const int32x4_t sub_const_vec, const int16x8_t vec_round_bits) {
  const int16x4_t y_filter_0_3 = vget_low_s16(y_filter_0_7);
  const int16x4_t y_filter_4_7 = vget_high_s16(y_filter_0_7);
  int32x4_t sum0, sum1;
  int16x8_t res;

  sum0 = vmull_lane_s16(vget_low_s16(s0), y_filter_0_3, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s1), y_filter_0_3, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s2), y_filter_0_3, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s3), y_filter_0_3, 3);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s4), y_filter_4_7, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s5), y_filter_4_7, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s6), y_filter_4_7, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s7), y_filter_4_7, 3);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s8), y_filter_8_11, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s9), y_filter_8_11, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s10), y_filter_8_11, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s11), y_filter_8_11, 3);

  sum1 = vmull_lane_s16(vget_high_s16(s0), y_filter_0_3, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s1), y_filter_0_3, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s2), y_filter_0_3, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s3), y_filter_0_3, 3);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s4), y_filter_4_7, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s5), y_filter_4_7, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s6), y_filter_4_7, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s7), y_filter_4_7, 3);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s8), y_filter_8_11, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s9), y_filter_8_11, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s10), y_filter_8_11, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s11), y_filter_8_11, 3);

  sum0 = vaddq_s32(sum0, offset_const);
  sum1 = vaddq_s32(sum1, offset_const);
  sum0 = vqrshlq_s32(sum0, round_shift_vec);
  sum1 = vqrshlq_s32(sum1, round_shift_vec);
  sum0 = vsubq_s32(sum0, sub_const_vec);
  sum1 = vsubq_s32(sum1, sub_const_vec);

  res = vcombine_s16(vmovn_s32(sum0), vmovn_s32(sum1));
  res = vqrshlq_s16(res, vec_round_bits);

  return vqmovun_s16(res);
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)

void av1_convolve_x_sr_neon(const uint8_t *src, int src_stride, uint8_t *dst,
                            int dst_stride, int w, int h,
                            const InterpFilterParams *filter_params_x,
                            const int subpel_x_qn,
                            ConvolveParams *conv_params) {
  if (filter_params_x->taps > 8) {
    av1_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h, filter_params_x,
                        subpel_x_qn, conv_params);
    return;
  }
  const uint8_t horiz_offset = filter_params_x->taps / 2 - 1;
  const int8_t bits = FILTER_BITS - conv_params->round_0;

  assert(bits >= 0);
  assert((FILTER_BITS - conv_params->round_1) >= 0 ||
         ((conv_params->round_0 + conv_params->round_1) == 2 * FILTER_BITS));

  const int16_t *x_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  // Filter values are even, so downshift by 1 to reduce intermediate precision
  // requirements.
  const int8x8_t x_filter = vshrn_n_s16(vld1q_s16(x_filter_ptr), 1);

  const int16x8_t shift_round_0 = vdupq_n_s16(-conv_params->round_0 + 1);
  const int16x8_t shift_by_bits = vdupq_n_s16(-bits);

  src -= horiz_offset;

  if (w <= 4) {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);
    uint8x16_t s0, s1, s2, s3;
    int32x4_t t0, t1, t2, t3;
    int16x8_t t01, t23;
    uint8x8_t d01, d23;

    do {
      s0 = vld1q_u8(src + 0 * src_stride);
      s1 = vld1q_u8(src + 1 * src_stride);
      s2 = vld1q_u8(src + 2 * src_stride);
      s3 = vld1q_u8(src + 3 * src_stride);

      t0 = convolve8_4_usdot(s0, x_filter, permute_tbl, vdupq_n_s32(0));
      t1 = convolve8_4_usdot(s1, x_filter, permute_tbl, vdupq_n_s32(0));
      t2 = convolve8_4_usdot(s2, x_filter, permute_tbl, vdupq_n_s32(0));
      t3 = convolve8_4_usdot(s3, x_filter, permute_tbl, vdupq_n_s32(0));

      t01 = vcombine_s16(vmovn_s32(t0), vmovn_s32(t1));
      t23 = vcombine_s16(vmovn_s32(t2), vmovn_s32(t3));

      t01 = vqrshlq_s16(t01, shift_round_0);
      t23 = vqrshlq_s16(t23, shift_round_0);

      t01 = vqrshlq_s16(t01, shift_by_bits);
      t23 = vqrshlq_s16(t23, shift_by_bits);

      d01 = vqmovun_s16(t01);
      d23 = vqmovun_s16(t23);

      if (w == 2) {
        vst1_lane_u16((uint16_t *)(dst + 0 * dst_stride),
                      vreinterpret_u16_u8(d01), 0);
        vst1_lane_u16((uint16_t *)(dst + 1 * dst_stride),
                      vreinterpret_u16_u8(d01), 2);
        if (h != 2) {
          vst1_lane_u16((uint16_t *)(dst + 2 * dst_stride),
                        vreinterpret_u16_u8(d23), 0);
          vst1_lane_u16((uint16_t *)(dst + 3 * dst_stride),
                        vreinterpret_u16_u8(d23), 2);
        }
      } else {
        vst1_lane_u32((uint32_t *)(dst + 0 * dst_stride),
                      vreinterpret_u32_u8(d01), 0);
        vst1_lane_u32((uint32_t *)(dst + 1 * dst_stride),
                      vreinterpret_u32_u8(d01), 1);
        if (h != 2) {
          vst1_lane_u32((uint32_t *)(dst + 2 * dst_stride),
                        vreinterpret_u32_u8(d23), 0);
          vst1_lane_u32((uint32_t *)(dst + 3 * dst_stride),
                        vreinterpret_u32_u8(d23), 1);
        }
      }

      h -= 4;
      src += 4 * src_stride;
      dst += 4 * dst_stride;
    } while (h > 0);

  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);
    uint8x16_t s0, s1, s2, s3;
    int16x8_t t0, t1, t2, t3;
    uint8x8_t d0, d1, d2, d3;

    do {
      int width = w;
      const uint8_t *s = src;
      uint8_t *d = dst;

      do {
        s0 = vld1q_u8(s + 0 * src_stride);
        s1 = vld1q_u8(s + 1 * src_stride);
        s2 = vld1q_u8(s + 2 * src_stride);
        s3 = vld1q_u8(s + 3 * src_stride);

        t0 = convolve8_8_usdot(s0, x_filter, permute_tbl, vdupq_n_s32(0),
                               shift_round_0);
        t1 = convolve8_8_usdot(s1, x_filter, permute_tbl, vdupq_n_s32(0),
                               shift_round_0);
        t2 = convolve8_8_usdot(s2, x_filter, permute_tbl, vdupq_n_s32(0),
                               shift_round_0);
        t3 = convolve8_8_usdot(s3, x_filter, permute_tbl, vdupq_n_s32(0),
                               shift_round_0);

        t0 = vqrshlq_s16(t0, shift_by_bits);
        t1 = vqrshlq_s16(t1, shift_by_bits);
        t2 = vqrshlq_s16(t2, shift_by_bits);
        t3 = vqrshlq_s16(t3, shift_by_bits);

        d0 = vqmovun_s16(t0);
        d1 = vqmovun_s16(t1);
        d2 = vqmovun_s16(t2);
        d3 = vqmovun_s16(t3);

        vst1_u8(d + 0 * dst_stride, d0);
        vst1_u8(d + 1 * dst_stride, d1);
        if (h != 2) {
          vst1_u8(d + 2 * dst_stride, d2);
          vst1_u8(d + 3 * dst_stride, d3);
        }

        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h > 0);
  }
}

#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

void av1_convolve_x_sr_neon(const uint8_t *src, int src_stride, uint8_t *dst,
                            int dst_stride, int w, int h,
                            const InterpFilterParams *filter_params_x,
                            const int subpel_x_qn,
                            ConvolveParams *conv_params) {
  if (filter_params_x->taps > 8) {
    av1_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h, filter_params_x,
                        subpel_x_qn, conv_params);
    return;
  }
  const uint8_t horiz_offset = filter_params_x->taps / 2 - 1;
  const int8_t bits = FILTER_BITS - conv_params->round_0;

  assert(bits >= 0);
  assert((FILTER_BITS - conv_params->round_1) >= 0 ||
         ((conv_params->round_0 + conv_params->round_1) == 2 * FILTER_BITS));

  const int16_t *x_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  // Filter values are even, so downshift by 1 to reduce intermediate precision
  // requirements.
  const int8x8_t x_filter = vshrn_n_s16(vld1q_s16(x_filter_ptr), 1);
  // Dot product constants.
  const int16x8_t correct_tmp = vshll_n_s8(x_filter, 7);
  const int32x4_t correction = vdupq_n_s32(vaddlvq_s16(correct_tmp));
  const uint8x16_t range_limit = vdupq_n_u8(128);

  const int16x8_t shift_round_0 = vdupq_n_s16(-conv_params->round_0 + 1);
  const int16x8_t shift_by_bits = vdupq_n_s16(-bits);

  src -= horiz_offset;

  if (w <= 4) {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);
    uint8x16_t s0, s1, s2, s3;
    int32x4_t t0, t1, t2, t3;
    int16x8_t t01, t23;
    uint8x8_t d01, d23;

    do {
      s0 = vld1q_u8(src + 0 * src_stride);
      s1 = vld1q_u8(src + 1 * src_stride);
      s2 = vld1q_u8(src + 2 * src_stride);
      s3 = vld1q_u8(src + 3 * src_stride);

      t0 = convolve8_4_sdot(s0, x_filter, correction, range_limit, permute_tbl);
      t1 = convolve8_4_sdot(s1, x_filter, correction, range_limit, permute_tbl);
      t2 = convolve8_4_sdot(s2, x_filter, correction, range_limit, permute_tbl);
      t3 = convolve8_4_sdot(s3, x_filter, correction, range_limit, permute_tbl);

      t01 = vcombine_s16(vmovn_s32(t0), vmovn_s32(t1));
      t23 = vcombine_s16(vmovn_s32(t2), vmovn_s32(t3));

      t01 = vqrshlq_s16(t01, shift_round_0);
      t23 = vqrshlq_s16(t23, shift_round_0);

      t01 = vqrshlq_s16(t01, shift_by_bits);
      t23 = vqrshlq_s16(t23, shift_by_bits);

      d01 = vqmovun_s16(t01);
      d23 = vqmovun_s16(t23);

      if (w == 2) {
        vst1_lane_u16((uint16_t *)(dst + 0 * dst_stride),
                      vreinterpret_u16_u8(d01), 0);
        vst1_lane_u16((uint16_t *)(dst + 1 * dst_stride),
                      vreinterpret_u16_u8(d01), 2);
        if (h != 2) {
          vst1_lane_u16((uint16_t *)(dst + 2 * dst_stride),
                        vreinterpret_u16_u8(d23), 0);
          vst1_lane_u16((uint16_t *)(dst + 3 * dst_stride),
                        vreinterpret_u16_u8(d23), 2);
        }
      } else {
        vst1_lane_u32((uint32_t *)(dst + 0 * dst_stride),
                      vreinterpret_u32_u8(d01), 0);
        vst1_lane_u32((uint32_t *)(dst + 1 * dst_stride),
                      vreinterpret_u32_u8(d01), 1);
        if (h != 2) {
          vst1_lane_u32((uint32_t *)(dst + 2 * dst_stride),
                        vreinterpret_u32_u8(d23), 0);
          vst1_lane_u32((uint32_t *)(dst + 3 * dst_stride),
                        vreinterpret_u32_u8(d23), 1);
        }
      }

      h -= 4;
      src += 4 * src_stride;
      dst += 4 * dst_stride;
    } while (h > 0);

  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);
    uint8x16_t s0, s1, s2, s3;
    int16x8_t t0, t1, t2, t3;
    uint8x8_t d0, d1, d2, d3;

    do {
      int width = w;
      const uint8_t *s = src;
      uint8_t *d = dst;

      do {
        s0 = vld1q_u8(s + 0 * src_stride);
        s1 = vld1q_u8(s + 1 * src_stride);
        s2 = vld1q_u8(s + 2 * src_stride);
        s3 = vld1q_u8(s + 3 * src_stride);

        t0 = convolve8_8_sdot(s0, x_filter, correction, range_limit,
                              permute_tbl, shift_round_0);
        t1 = convolve8_8_sdot(s1, x_filter, correction, range_limit,
                              permute_tbl, shift_round_0);
        t2 = convolve8_8_sdot(s2, x_filter, correction, range_limit,
                              permute_tbl, shift_round_0);
        t3 = convolve8_8_sdot(s3, x_filter, correction, range_limit,
                              permute_tbl, shift_round_0);

        t0 = vqrshlq_s16(t0, shift_by_bits);
        t1 = vqrshlq_s16(t1, shift_by_bits);
        t2 = vqrshlq_s16(t2, shift_by_bits);
        t3 = vqrshlq_s16(t3, shift_by_bits);

        d0 = vqmovun_s16(t0);
        d1 = vqmovun_s16(t1);
        d2 = vqmovun_s16(t2);
        d3 = vqmovun_s16(t3);

        vst1_u8(d + 0 * dst_stride, d0);
        vst1_u8(d + 1 * dst_stride, d1);
        if (h != 2) {
          vst1_u8(d + 2 * dst_stride, d2);
          vst1_u8(d + 3 * dst_stride, d3);
        }

        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);

      src += 4 * src_stride;
      dst += 4 * dst_stride;
      h -= 4;
    } while (h > 0);
  }
}

#else  // !(defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD))

static INLINE uint8x8_t convolve8_horiz_8x8(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
    const int16x8_t s6, const int16x8_t s7, const int16x8_t filter,
    const int16x8_t shift_round_0, const int16x8_t shift_by_bits) {
  const int16x4_t filter_lo = vget_low_s16(filter);
  const int16x4_t filter_hi = vget_high_s16(filter);
  int16x8_t sum;

  sum = vmulq_lane_s16(s0, filter_lo, 0);
  sum = vmlaq_lane_s16(sum, s1, filter_lo, 1);
  sum = vmlaq_lane_s16(sum, s2, filter_lo, 2);
  sum = vmlaq_lane_s16(sum, s3, filter_lo, 3);
  sum = vmlaq_lane_s16(sum, s4, filter_hi, 0);
  sum = vmlaq_lane_s16(sum, s5, filter_hi, 1);
  sum = vmlaq_lane_s16(sum, s6, filter_hi, 2);
  sum = vmlaq_lane_s16(sum, s7, filter_hi, 3);

  sum = vqrshlq_s16(sum, shift_round_0);
  sum = vqrshlq_s16(sum, shift_by_bits);

  return vqmovun_s16(sum);
}

void av1_convolve_x_sr_neon(const uint8_t *src, int src_stride, uint8_t *dst,
                            int dst_stride, int w, int h,
                            const InterpFilterParams *filter_params_x,
                            const int subpel_x_qn,
                            ConvolveParams *conv_params) {
  if (filter_params_x->taps > 8) {
    av1_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h, filter_params_x,
                        subpel_x_qn, conv_params);
    return;
  }
  const uint8_t horiz_offset = filter_params_x->taps / 2 - 1;
  const int8_t bits = FILTER_BITS - conv_params->round_0;

  uint8x8_t t0;
#if defined(__aarch64__)
  uint8x8_t t1, t2, t3;
#endif

  assert(bits >= 0);
  assert((FILTER_BITS - conv_params->round_1) >= 0 ||
         ((conv_params->round_0 + conv_params->round_1) == 2 * FILTER_BITS));

  const int16_t *x_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  // Filter values are even so downshift by 1 to reduce precision requirements.
  const int16x8_t x_filter = vshrq_n_s16(vld1q_s16(x_filter_ptr), 1);

  const int16x8_t shift_round_0 = vdupq_n_s16(-conv_params->round_0 + 1);
  const int16x8_t shift_by_bits = vdupq_n_s16(-bits);

  src -= horiz_offset;
#if defined(__aarch64__)
  if (h == 4) {
    uint8x8_t d01, d23;
    int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, d0, d1, d2, d3;
    int16x8_t d01_temp, d23_temp;

    __builtin_prefetch(src + 0 * src_stride);
    __builtin_prefetch(src + 1 * src_stride);
    __builtin_prefetch(src + 2 * src_stride);
    __builtin_prefetch(src + 3 * src_stride);

    load_u8_8x4(src, src_stride, &t0, &t1, &t2, &t3);
    transpose_u8_8x4(&t0, &t1, &t2, &t3);

    s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
    s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
    s4 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    s5 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    s6 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
    __builtin_prefetch(dst + 0 * dst_stride);
    __builtin_prefetch(dst + 1 * dst_stride);
    __builtin_prefetch(dst + 2 * dst_stride);
    __builtin_prefetch(dst + 3 * dst_stride);
    src += 7;

    do {
      load_u8_8x4(src, src_stride, &t0, &t1, &t2, &t3);
      transpose_u8_8x4(&t0, &t1, &t2, &t3);

      s7 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
      s8 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
      s9 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
      s10 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));

      d0 = convolve8_4x4(s0, s1, s2, s3, s4, s5, s6, s7, x_filter);

      d1 = convolve8_4x4(s1, s2, s3, s4, s5, s6, s7, s8, x_filter);

      d2 = convolve8_4x4(s2, s3, s4, s5, s6, s7, s8, s9, x_filter);

      d3 = convolve8_4x4(s3, s4, s5, s6, s7, s8, s9, s10, x_filter);

      d01_temp = vqrshlq_s16(vcombine_s16(d0, d1), shift_round_0);
      d23_temp = vqrshlq_s16(vcombine_s16(d2, d3), shift_round_0);

      d01_temp = vqrshlq_s16(d01_temp, shift_by_bits);
      d23_temp = vqrshlq_s16(d23_temp, shift_by_bits);

      d01 = vqmovun_s16(d01_temp);
      d23 = vqmovun_s16(d23_temp);

      transpose_u8_4x4(&d01, &d23);

      if (w != 2) {
        vst1_lane_u32((uint32_t *)(dst + 0 * dst_stride),  // 00 01 02 03
                      vreinterpret_u32_u8(d01), 0);
        vst1_lane_u32((uint32_t *)(dst + 1 * dst_stride),  // 10 11 12 13
                      vreinterpret_u32_u8(d23), 0);
        vst1_lane_u32((uint32_t *)(dst + 2 * dst_stride),  // 20 21 22 23
                      vreinterpret_u32_u8(d01), 1);
        vst1_lane_u32((uint32_t *)(dst + 3 * dst_stride),  // 30 31 32 33
                      vreinterpret_u32_u8(d23), 1);
      } else {
        vst1_lane_u16((uint16_t *)(dst + 0 * dst_stride),  // 00 01
                      vreinterpret_u16_u8(d01), 0);
        vst1_lane_u16((uint16_t *)(dst + 1 * dst_stride),  // 10 11
                      vreinterpret_u16_u8(d23), 0);
        vst1_lane_u16((uint16_t *)(dst + 2 * dst_stride),  // 20 21
                      vreinterpret_u16_u8(d01), 2);
        vst1_lane_u16((uint16_t *)(dst + 3 * dst_stride),  // 30 31
                      vreinterpret_u16_u8(d23), 2);
      }

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      src += 4;
      dst += 4;
      w -= 4;
    } while (w > 0);
  } else {
#endif
    int width;
    const uint8_t *s;
    int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;

#if defined(__aarch64__)
    int16x8_t s8, s9, s10;
    uint8x8_t t4, t5, t6, t7;
#endif

    if (w <= 4) {
#if defined(__aarch64__)
      do {
        load_u8_8x8(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
        s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
        s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
        s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
        s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
        s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
        s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

        load_u8_8x8(src + 7, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6,
                    &t7);
        src += 8 * src_stride;
        __builtin_prefetch(dst + 0 * dst_stride);
        __builtin_prefetch(dst + 1 * dst_stride);
        __builtin_prefetch(dst + 2 * dst_stride);
        __builtin_prefetch(dst + 3 * dst_stride);
        __builtin_prefetch(dst + 4 * dst_stride);
        __builtin_prefetch(dst + 5 * dst_stride);
        __builtin_prefetch(dst + 6 * dst_stride);
        __builtin_prefetch(dst + 7 * dst_stride);

        transpose_u8_4x8(&t0, &t1, &t2, &t3, t4, t5, t6, t7);

        s7 = vreinterpretq_s16_u16(vmovl_u8(t0));
        s8 = vreinterpretq_s16_u16(vmovl_u8(t1));
        s9 = vreinterpretq_s16_u16(vmovl_u8(t2));
        s10 = vreinterpretq_s16_u16(vmovl_u8(t3));

        __builtin_prefetch(src + 0 * src_stride);
        __builtin_prefetch(src + 1 * src_stride);
        __builtin_prefetch(src + 2 * src_stride);
        __builtin_prefetch(src + 3 * src_stride);
        __builtin_prefetch(src + 4 * src_stride);
        __builtin_prefetch(src + 5 * src_stride);
        __builtin_prefetch(src + 6 * src_stride);
        __builtin_prefetch(src + 7 * src_stride);
        t0 = convolve8_horiz_8x8(s0, s1, s2, s3, s4, s5, s6, s7, x_filter,
                                 shift_round_0, shift_by_bits);
        t1 = convolve8_horiz_8x8(s1, s2, s3, s4, s5, s6, s7, s8, x_filter,
                                 shift_round_0, shift_by_bits);
        t2 = convolve8_horiz_8x8(s2, s3, s4, s5, s6, s7, s8, s9, x_filter,
                                 shift_round_0, shift_by_bits);
        t3 = convolve8_horiz_8x8(s3, s4, s5, s6, s7, s8, s9, s10, x_filter,
                                 shift_round_0, shift_by_bits);

        transpose_u8_8x4(&t0, &t1, &t2, &t3);

        if ((w == 4) && (h > 4)) {
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t0),
                        0);  // 00 01 02 03
          dst += dst_stride;
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t1),
                        0);  // 10 11 12 13
          dst += dst_stride;
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t2),
                        0);  // 20 21 22 23
          dst += dst_stride;
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t3),
                        0);  // 30 31 32 33
          dst += dst_stride;
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t0),
                        1);  // 40 41 42 43
          dst += dst_stride;
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t1),
                        1);  // 50 51 52 53
          dst += dst_stride;
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t2),
                        1);  // 60 61 62 63
          dst += dst_stride;
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t3),
                        1);  // 70 71 72 73
          dst += dst_stride;
        } else if ((w == 4) && (h == 2)) {
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t0),
                        0);  // 00 01 02 03
          dst += dst_stride;
          vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t1),
                        0);  // 10 11 12 13
          dst += dst_stride;
        } else if ((w == 2) && (h > 4)) {
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t0),
                        0);  // 00 01
          dst += dst_stride;
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t1),
                        0);  // 10 11
          dst += dst_stride;
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t2),
                        0);  // 20 21
          dst += dst_stride;
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t3),
                        0);  // 30 31
          dst += dst_stride;
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t0),
                        2);  // 40 41
          dst += dst_stride;
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t1),
                        2);  // 50 51
          dst += dst_stride;
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t2),
                        2);  // 60 61
          dst += dst_stride;
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t3),
                        2);  // 70 71
          dst += dst_stride;
        } else if ((w == 2) && (h == 2)) {
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t0),
                        0);  // 00 01
          dst += dst_stride;
          vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t1),
                        0);  // 10 11
          dst += dst_stride;
        }
        h -= 8;
      } while (h > 0);
#else
    int16x8_t tt0;
    int16x4_t x0, x1, x2, x3, x4, x5, x6, x7;
    const int16x4_t shift_round_0_low = vget_low_s16(shift_round_0);
    const int16x4_t shift_by_bits_low = vget_low_s16(shift_by_bits);
    do {
      t0 = vld1_u8(src);  // a0 a1 a2 a3 a4 a5 a6 a7
      tt0 = vreinterpretq_s16_u16(vmovl_u8(t0));
      x0 = vget_low_s16(tt0);   // a0 a1 a2 a3
      x4 = vget_high_s16(tt0);  // a4 a5 a6 a7

      t0 = vld1_u8(src + 8);  // a8 a9 a10 a11 a12 a13 a14 a15
      tt0 = vreinterpretq_s16_u16(vmovl_u8(t0));
      x7 = vget_low_s16(tt0);  // a8 a9 a10 a11

      x1 = vext_s16(x0, x4, 1);  // a1 a2 a3 a4
      x2 = vext_s16(x0, x4, 2);  // a2 a3 a4 a5
      x3 = vext_s16(x0, x4, 3);  // a3 a4 a5 a6
      x5 = vext_s16(x4, x7, 1);  // a5 a6 a7 a8
      x6 = vext_s16(x4, x7, 2);  // a6 a7 a8 a9
      x7 = vext_s16(x4, x7, 3);  // a7 a8 a9 a10

      src += src_stride;

      t0 = convolve8_horiz_4x1(x0, x1, x2, x3, x4, x5, x6, x7, x_filter,
                               shift_round_0_low, shift_by_bits_low);

      if (w == 4) {
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(t0),
                      0);  // 00 01 02 03
        dst += dst_stride;
      } else if (w == 2) {
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(t0), 0);  // 00 01
        dst += dst_stride;
      }
      h -= 1;
    } while (h > 0);
#endif
    } else {
      uint8_t *d;
      int16x8_t s11;
#if defined(__aarch64__)
      int16x8_t s12, s13, s14;
      do {
        __builtin_prefetch(src + 0 * src_stride);
        __builtin_prefetch(src + 1 * src_stride);
        __builtin_prefetch(src + 2 * src_stride);
        __builtin_prefetch(src + 3 * src_stride);
        __builtin_prefetch(src + 4 * src_stride);
        __builtin_prefetch(src + 5 * src_stride);
        __builtin_prefetch(src + 6 * src_stride);
        __builtin_prefetch(src + 7 * src_stride);
        load_u8_8x8(src, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
        s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
        s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
        s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
        s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
        s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
        s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
        s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

        width = w;
        s = src + 7;
        d = dst;
        __builtin_prefetch(dst + 0 * dst_stride);
        __builtin_prefetch(dst + 1 * dst_stride);
        __builtin_prefetch(dst + 2 * dst_stride);
        __builtin_prefetch(dst + 3 * dst_stride);
        __builtin_prefetch(dst + 4 * dst_stride);
        __builtin_prefetch(dst + 5 * dst_stride);
        __builtin_prefetch(dst + 6 * dst_stride);
        __builtin_prefetch(dst + 7 * dst_stride);

        do {
          load_u8_8x8(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
          transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
          s7 = vreinterpretq_s16_u16(vmovl_u8(t0));
          s8 = vreinterpretq_s16_u16(vmovl_u8(t1));
          s9 = vreinterpretq_s16_u16(vmovl_u8(t2));
          s10 = vreinterpretq_s16_u16(vmovl_u8(t3));
          s11 = vreinterpretq_s16_u16(vmovl_u8(t4));
          s12 = vreinterpretq_s16_u16(vmovl_u8(t5));
          s13 = vreinterpretq_s16_u16(vmovl_u8(t6));
          s14 = vreinterpretq_s16_u16(vmovl_u8(t7));

          t0 = convolve8_horiz_8x8(s0, s1, s2, s3, s4, s5, s6, s7, x_filter,
                                   shift_round_0, shift_by_bits);

          t1 = convolve8_horiz_8x8(s1, s2, s3, s4, s5, s6, s7, s8, x_filter,
                                   shift_round_0, shift_by_bits);

          t2 = convolve8_horiz_8x8(s2, s3, s4, s5, s6, s7, s8, s9, x_filter,
                                   shift_round_0, shift_by_bits);

          t3 = convolve8_horiz_8x8(s3, s4, s5, s6, s7, s8, s9, s10, x_filter,
                                   shift_round_0, shift_by_bits);

          t4 = convolve8_horiz_8x8(s4, s5, s6, s7, s8, s9, s10, s11, x_filter,
                                   shift_round_0, shift_by_bits);

          t5 = convolve8_horiz_8x8(s5, s6, s7, s8, s9, s10, s11, s12, x_filter,
                                   shift_round_0, shift_by_bits);

          t6 = convolve8_horiz_8x8(s6, s7, s8, s9, s10, s11, s12, s13, x_filter,
                                   shift_round_0, shift_by_bits);

          t7 = convolve8_horiz_8x8(s7, s8, s9, s10, s11, s12, s13, s14,
                                   x_filter, shift_round_0, shift_by_bits);

          transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);
          if (h != 2) {
            store_u8_8x8(d, dst_stride, t0, t1, t2, t3, t4, t5, t6, t7);
          } else {
            store_row2_u8_8x8(d, dst_stride, t0, t1);
          }
          s0 = s8;
          s1 = s9;
          s2 = s10;
          s3 = s11;
          s4 = s12;
          s5 = s13;
          s6 = s14;
          s += 8;
          d += 8;
          width -= 8;
        } while (width > 0);
        src += 8 * src_stride;
        dst += 8 * dst_stride;
        h -= 8;
      } while (h > 0);
#else
    do {
      t0 = vld1_u8(src);  // a0 a1 a2 a3 a4 a5 a6 a7
      s0 = vreinterpretq_s16_u16(vmovl_u8(t0));

      width = w;
      s = src + 8;
      d = dst;
      __builtin_prefetch(dst);

      do {
        t0 = vld1_u8(s);  // a8 a9 a10 a11 a12 a13 a14 a15
        s7 = vreinterpretq_s16_u16(vmovl_u8(t0));
        s11 = s0;
        s0 = s7;

        s1 = vextq_s16(s11, s7, 1);  // a1 a2 a3 a4 a5 a6 a7 a8
        s2 = vextq_s16(s11, s7, 2);  // a2 a3 a4 a5 a6 a7 a8 a9
        s3 = vextq_s16(s11, s7, 3);  // a3 a4 a5 a6 a7 a8 a9 a10
        s4 = vextq_s16(s11, s7, 4);  // a4 a5 a6 a7 a8 a9 a10 a11
        s5 = vextq_s16(s11, s7, 5);  // a5 a6 a7 a8 a9 a10 a11 a12
        s6 = vextq_s16(s11, s7, 6);  // a6 a7 a8 a9 a10 a11 a12 a13
        s7 = vextq_s16(s11, s7, 7);  // a7 a8 a9 a10 a11 a12 a13 a14

        t0 = convolve8_horiz_8x8(s11, s1, s2, s3, s4, s5, s6, s7, x_filter,
                                 shift_round_0, shift_by_bits);
        vst1_u8(d, t0);

        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);
      src += src_stride;
      dst += dst_stride;
      h -= 1;
    } while (h > 0);
#endif
    }
#if defined(__aarch64__)
  }
#endif
}

#endif  // defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

void av1_convolve_y_sr_neon(const uint8_t *src, int src_stride, uint8_t *dst,
                            int dst_stride, int w, int h,
                            const InterpFilterParams *filter_params_y,
                            const int subpel_y_qn) {
  if (filter_params_y->taps > 8) {
    av1_convolve_y_sr_c(src, src_stride, dst, dst_stride, w, h, filter_params_y,
                        subpel_y_qn);
    return;
  }
  const int vert_offset = filter_params_y->taps / 2 - 1;

  src -= vert_offset * src_stride;

  const int16_t *y_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  // Filter values are even so downshift by 1 to reduce precision requirements.
  const int16x8_t y_filter = vshrq_n_s16(vld1q_s16(y_filter_ptr), 1);

  if (w <= 4) {
    uint8x8_t d01;
    int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, d0;
#if defined(__aarch64__)
    uint8x8_t d23;
    int16x4_t s8, s9, s10, d1, d2, d3;
#endif
    s0 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
    src += src_stride;
    s1 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
    src += src_stride;
    s2 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
    src += src_stride;
    s3 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
    src += src_stride;
    s4 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
    src += src_stride;
    s5 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
    src += src_stride;
    s6 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
    src += src_stride;

    do {
      s7 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
      src += src_stride;
#if defined(__aarch64__)
      s8 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
      src += src_stride;
      s9 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
      src += src_stride;
      s10 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(src))));
      src += src_stride;

      __builtin_prefetch(dst + 0 * dst_stride);
      __builtin_prefetch(dst + 1 * dst_stride);
      __builtin_prefetch(dst + 2 * dst_stride);
      __builtin_prefetch(dst + 3 * dst_stride);
      __builtin_prefetch(src + 0 * src_stride);
      __builtin_prefetch(src + 1 * src_stride);
      __builtin_prefetch(src + 2 * src_stride);
      __builtin_prefetch(src + 3 * src_stride);
      d0 = convolve8_4x4(s0, s1, s2, s3, s4, s5, s6, s7, y_filter);
      d1 = convolve8_4x4(s1, s2, s3, s4, s5, s6, s7, s8, y_filter);
      d2 = convolve8_4x4(s2, s3, s4, s5, s6, s7, s8, s9, y_filter);
      d3 = convolve8_4x4(s3, s4, s5, s6, s7, s8, s9, s10, y_filter);

      d01 = vqrshrun_n_s16(vcombine_s16(d0, d1), FILTER_BITS - 1);
      d23 = vqrshrun_n_s16(vcombine_s16(d2, d3), FILTER_BITS - 1);
      if ((w == 4) && (h != 2)) {
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d01),
                      0);  // 00 01 02 03
        dst += dst_stride;
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d01),
                      1);  // 10 11 12 13
        dst += dst_stride;
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d23),
                      0);  // 20 21 22 23
        dst += dst_stride;
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d23),
                      1);  // 30 31 32 33
        dst += dst_stride;
      } else if ((w == 4) && (h == 2)) {
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d01),
                      0);  // 00 01 02 03
        dst += dst_stride;
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d01),
                      1);  // 10 11 12 13
        dst += dst_stride;
      } else if ((w == 2) && (h != 2)) {
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(d01), 0);  // 00 01
        dst += dst_stride;
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(d01), 2);  // 10 11
        dst += dst_stride;
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(d23), 0);  // 20 21
        dst += dst_stride;
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(d23), 2);  // 30 31
        dst += dst_stride;
      } else if ((w == 2) && (h == 2)) {
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(d01), 0);  // 00 01
        dst += dst_stride;
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(d01), 2);  // 10 11
        dst += dst_stride;
      }
      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      h -= 4;
#else
      __builtin_prefetch(dst + 0 * dst_stride);
      __builtin_prefetch(src + 0 * src_stride);

      d0 = convolve8_4x4(s0, s1, s2, s3, s4, s5, s6, s7, y_filter);

      d01 = vqrshrun_n_s16(vcombine_s16(d0, d0), FILTER_BITS - 1);

      if (w == 4) {
        vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d01), 0);
        dst += dst_stride;
      } else if (w == 2) {
        vst1_lane_u16((uint16_t *)dst, vreinterpret_u16_u8(d01), 0);
        dst += dst_stride;
      }
      s0 = s1;
      s1 = s2;
      s2 = s3;
      s3 = s4;
      s4 = s5;
      s5 = s6;
      s6 = s7;
      h -= 1;
#endif
    } while (h > 0);
  } else {
    int height;
    const uint8_t *s;
    uint8_t *d;
    uint8x8_t t0;
    int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
#if defined(__aarch64__)
    uint8x8_t t1, t2, t3;
    int16x8_t s8, s9, s10;
#endif
    do {
      __builtin_prefetch(src + 0 * src_stride);
      __builtin_prefetch(src + 1 * src_stride);
      __builtin_prefetch(src + 2 * src_stride);
      __builtin_prefetch(src + 3 * src_stride);
      __builtin_prefetch(src + 4 * src_stride);
      __builtin_prefetch(src + 5 * src_stride);
      __builtin_prefetch(src + 6 * src_stride);
      s = src;
      s0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
      s += src_stride;
      s1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
      s += src_stride;
      s2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
      s += src_stride;
      s3 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
      s += src_stride;
      s4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
      s += src_stride;
      s5 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
      s += src_stride;
      s6 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
      s += src_stride;
      d = dst;
      height = h;

      do {
        s7 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
        s += src_stride;
#if defined(__aarch64__)
        s8 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
        s += src_stride;
        s9 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
        s += src_stride;
        s10 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s)));
        s += src_stride;

        __builtin_prefetch(d + 0 * dst_stride);
        __builtin_prefetch(d + 1 * dst_stride);
        __builtin_prefetch(d + 2 * dst_stride);
        __builtin_prefetch(d + 3 * dst_stride);
        __builtin_prefetch(s + 0 * src_stride);
        __builtin_prefetch(s + 1 * src_stride);
        __builtin_prefetch(s + 2 * src_stride);
        __builtin_prefetch(s + 3 * src_stride);
        t0 = convolve8_vert_8x4(s0, s1, s2, s3, s4, s5, s6, s7, y_filter);
        t1 = convolve8_vert_8x4(s1, s2, s3, s4, s5, s6, s7, s8, y_filter);
        t2 = convolve8_vert_8x4(s2, s3, s4, s5, s6, s7, s8, s9, y_filter);
        t3 = convolve8_vert_8x4(s3, s4, s5, s6, s7, s8, s9, s10, y_filter);
        if (h != 2) {
          vst1_u8(d, t0);
          d += dst_stride;
          vst1_u8(d, t1);
          d += dst_stride;
          vst1_u8(d, t2);
          d += dst_stride;
          vst1_u8(d, t3);
          d += dst_stride;
        } else {
          vst1_u8(d, t0);
          d += dst_stride;
          vst1_u8(d, t1);
          d += dst_stride;
        }
        s0 = s4;
        s1 = s5;
        s2 = s6;
        s3 = s7;
        s4 = s8;
        s5 = s9;
        s6 = s10;
        height -= 4;
#else
        __builtin_prefetch(d);
        __builtin_prefetch(s);

        t0 = convolve8_vert_8x4(s0, s1, s2, s3, s4, s5, s6, s7, y_filter);

        vst1_u8(d, t0);
        d += dst_stride;

        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s5;
        s5 = s6;
        s6 = s7;
        height -= 1;
#endif
      } while (height > 0);
      src += 8;
      dst += 8;
      w -= 8;
    } while (w > 0);
  }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)

static INLINE int16x4_t convolve12_4_usdot(uint8x16_t samples,
                                           const int8x16_t filters,
                                           const uint8x16x3_t permute_tbl,
                                           const int32x4_t horiz_const,
                                           const int32x4_t shift_round_0) {
  uint8x16_t permuted_samples[3];
  int32x4_t sum;

  /* Permute samples ready for dot product. */
  /* { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 } */
  permuted_samples[0] = vqtbl1q_u8(samples, permute_tbl.val[0]);
  /* { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 } */
  permuted_samples[1] = vqtbl1q_u8(samples, permute_tbl.val[1]);
  /* { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 } */
  permuted_samples[2] = vqtbl1q_u8(samples, permute_tbl.val[2]);

  /* First 4 output values. */
  sum = vusdotq_laneq_s32(horiz_const, permuted_samples[0], filters, 0);
  sum = vusdotq_laneq_s32(sum, permuted_samples[1], filters, 1);
  sum = vusdotq_laneq_s32(sum, permuted_samples[2], filters, 2);

  /* Narrow and re-pack. */
  sum = vqrshlq_s32(sum, shift_round_0);

  return vmovn_s32(sum);
}

static INLINE int16x8_t convolve12_8_usdot(uint8x16_t samples0,
                                           uint8x16_t samples1,
                                           const int8x16_t filters,
                                           const uint8x16x3_t permute_tbl,
                                           const int32x4_t horiz_const,
                                           const int32x4_t shift_round_0) {
  uint8x16_t permuted_samples[4];
  int32x4_t sum[2];

  /* Permute samples ready for dot product. */
  /* { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 } */
  permuted_samples[0] = vqtbl1q_u8(samples0, permute_tbl.val[0]);
  /* { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 } */
  permuted_samples[1] = vqtbl1q_u8(samples0, permute_tbl.val[1]);
  /* { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 } */
  permuted_samples[2] = vqtbl1q_u8(samples0, permute_tbl.val[2]);
  /* {12, 13, 14, 15, 13, 14, 15, 16, 14, 15, 16, 17, 15, 16, 17, 18 } */
  permuted_samples[3] = vqtbl1q_u8(samples1, permute_tbl.val[2]);

  /* First 4 output values. */
  sum[0] = vusdotq_laneq_s32(horiz_const, permuted_samples[0], filters, 0);
  sum[0] = vusdotq_laneq_s32(sum[0], permuted_samples[1], filters, 1);
  sum[0] = vusdotq_laneq_s32(sum[0], permuted_samples[2], filters, 2);
  /* Second 4 output values. */
  sum[1] = vusdotq_laneq_s32(horiz_const, permuted_samples[1], filters, 0);
  sum[1] = vusdotq_laneq_s32(sum[1], permuted_samples[2], filters, 1);
  sum[1] = vusdotq_laneq_s32(sum[1], permuted_samples[3], filters, 2);

  /* Narrow and re-pack. */
  sum[0] = vqrshlq_s32(sum[0], shift_round_0);
  sum[1] = vqrshlq_s32(sum[1], shift_round_0);

  return vcombine_s16(vmovn_s32(sum[0]), vmovn_s32(sum[1]));
}

static INLINE void av1_convolve_2d_sr_horiz_12tap_neon(
    const uint8_t *src_ptr, int src_stride, int16_t *dst_ptr,
    const int dst_stride, int w, int h, const int16x8_t x_filter_0_7,
    const int16x4_t x_filter_8_11, const int round_0) {
  const int bd = 8;

  // Special case the following no-op filter as 128 won't fit into the
  // 8-bit signed dot-product instruction:
  // { 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0 }
  if (vgetq_lane_s16(x_filter_0_7, 5) == 128) {
    const int16x8_t horiz_const = vdupq_n_s16((1 << (bd - 1)));
    const int16x8_t shift_round_0 = vdupq_n_s16(FILTER_BITS - round_0);
    // Undo the horizontal offset in the calling function.
    src_ptr += 5;

    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j += 8) {
        uint8x8_t s0 = vld1_u8(src_ptr + i * src_stride + j);
        uint16x8_t t0 = vaddw_u8(vreinterpretq_u16_s16(horiz_const), s0);
        int16x8_t d0 = vqrshlq_s16(vreinterpretq_s16_u16(t0), shift_round_0);
        if (w == 2) {
          vst1q_lane_s32((int32_t *)(dst_ptr + i * dst_stride),
                         vreinterpretq_s32_s16(d0), 0);
        } else if (w == 4) {
          vst1_s16(dst_ptr + i * dst_stride, vget_low_s16(d0));
        } else {
          vst1q_s16(dst_ptr + i * dst_stride + j, d0);
        }
      }
    }
  } else {
    // Narrow filter values to 8-bit.
    const int16x8x2_t x_filter_s16 = {
      { x_filter_0_7, vcombine_s16(x_filter_8_11, vdup_n_s16(0)) }
    };
    const int8x16_t x_filter = vcombine_s8(vmovn_s16(x_filter_s16.val[0]),
                                           vmovn_s16(x_filter_s16.val[1]));

    const int32x4_t horiz_const = vdupq_n_s32((1 << (bd + FILTER_BITS - 1)));
    const int32x4_t shift_round_0 = vdupq_n_s32(-round_0);
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);

    if (w <= 4) {
      do {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          uint8x16_t s0, s1, s2, s3;
          int16x4_t d0, d1, d2, d3;

          s0 = vld1q_u8(s + 0 * src_stride);
          s1 = vld1q_u8(s + 1 * src_stride);
          s2 = vld1q_u8(s + 2 * src_stride);
          s3 = vld1q_u8(s + 3 * src_stride);

          d0 = convolve12_4_usdot(s0, x_filter, permute_tbl, horiz_const,
                                  shift_round_0);
          d1 = convolve12_4_usdot(s1, x_filter, permute_tbl, horiz_const,
                                  shift_round_0);
          d2 = convolve12_4_usdot(s2, x_filter, permute_tbl, horiz_const,
                                  shift_round_0);
          d3 = convolve12_4_usdot(s3, x_filter, permute_tbl, horiz_const,
                                  shift_round_0);

          if (w == 2) {
            vst1_lane_s32((int32_t *)(d + 0 * dst_stride),
                          vreinterpret_s32_s16(d0), 0);
            vst1_lane_s32((int32_t *)(d + 1 * dst_stride),
                          vreinterpret_s32_s16(d1), 0);
            vst1_lane_s32((int32_t *)(d + 2 * dst_stride),
                          vreinterpret_s32_s16(d2), 0);
            vst1_lane_s32((int32_t *)(d + 3 * dst_stride),
                          vreinterpret_s32_s16(d3), 0);
          } else {
            vst1_s16(d + 0 * dst_stride, d0);
            vst1_s16(d + 1 * dst_stride, d1);
            vst1_s16(d + 2 * dst_stride, d2);
            vst1_s16(d + 3 * dst_stride, d3);
          }

          s += 4;
          d += 4;
          width -= 4;
        } while (width > 0);

        src_ptr += 4 * src_stride;
        dst_ptr += 4 * dst_stride;
        h -= 4;
      } while (h >= 4);

      for (; h > 0; h--) {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          uint8x16_t s0;
          int16x4_t d0;

          s0 = vld1q_u8(s);

          d0 = convolve12_4_usdot(s0, x_filter, permute_tbl, horiz_const,
                                  shift_round_0);

          if (w == 2) {
            vst1_lane_s32((int32_t *)d, vreinterpret_s32_s16(d0), 0);
          } else {
            vst1_s16(d, d0);
          }

          s += 4;
          d += 4;
          width -= 4;
        } while (width > 0);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
      }
    } else {
      do {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          uint8x16_t s0[2], s1[2], s2[2], s3[2];
          int16x8_t d0, d1, d2, d3;

          s0[0] = vld1q_u8(s + 0 * src_stride);
          s1[0] = vld1q_u8(s + 1 * src_stride);
          s2[0] = vld1q_u8(s + 2 * src_stride);
          s3[0] = vld1q_u8(s + 3 * src_stride);
          s0[1] = vld1q_u8(s + 0 * src_stride + 4);
          s1[1] = vld1q_u8(s + 1 * src_stride + 4);
          s2[1] = vld1q_u8(s + 2 * src_stride + 4);
          s3[1] = vld1q_u8(s + 3 * src_stride + 4);

          d0 = convolve12_8_usdot(s0[0], s0[1], x_filter, permute_tbl,
                                  horiz_const, shift_round_0);
          d1 = convolve12_8_usdot(s1[0], s1[1], x_filter, permute_tbl,
                                  horiz_const, shift_round_0);
          d2 = convolve12_8_usdot(s2[0], s2[1], x_filter, permute_tbl,
                                  horiz_const, shift_round_0);
          d3 = convolve12_8_usdot(s3[0], s3[1], x_filter, permute_tbl,
                                  horiz_const, shift_round_0);

          vst1q_s16(d + 0 * dst_stride, d0);
          vst1q_s16(d + 1 * dst_stride, d1);
          vst1q_s16(d + 2 * dst_stride, d2);
          vst1q_s16(d + 3 * dst_stride, d3);

          s += 8;
          d += 8;
          width -= 8;
        } while (width > 0);

        src_ptr += 4 * src_stride;
        dst_ptr += 4 * dst_stride;
        h -= 4;
      } while (h >= 4);

      for (; h > 0; h--) {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          uint8x16_t s0[2];
          int16x8_t d0;

          s0[0] = vld1q_u8(s);
          s0[1] = vld1q_u8(s + 4);

          d0 = convolve12_8_usdot(s0[0], s0[1], x_filter, permute_tbl,
                                  horiz_const, shift_round_0);

          vst1q_s16(d, d0);

          s += 8;
          d += 8;
          width -= 8;
        } while (width > 0);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
      }
    }
  }
}

#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

static INLINE int16x4_t convolve12_4_sdot(uint8x16_t samples,
                                          const int8x16_t filters,
                                          const int32x4_t correction,
                                          const uint8x16_t range_limit,
                                          const uint8x16x3_t permute_tbl,
                                          const int32x4_t shift_round_0) {
  int8x16_t clamped_samples, permuted_samples[3];
  int32x4_t sum;

  /* Clamp sample range to [-128, 127] for 8-bit signed dot product. */
  clamped_samples = vreinterpretq_s8_u8(vsubq_u8(samples, range_limit));

  /* Permute samples ready for dot product. */
  /* { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 } */
  permuted_samples[0] = vqtbl1q_s8(clamped_samples, permute_tbl.val[0]);
  /* { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 } */
  permuted_samples[1] = vqtbl1q_s8(clamped_samples, permute_tbl.val[1]);
  /* { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 } */
  permuted_samples[2] = vqtbl1q_s8(clamped_samples, permute_tbl.val[2]);

  /* Accumulate dot product into 'correction' to account for range clamp. */
  /* First 4 output values. */
  sum = vdotq_laneq_s32(correction, permuted_samples[0], filters, 0);
  sum = vdotq_laneq_s32(sum, permuted_samples[1], filters, 1);
  sum = vdotq_laneq_s32(sum, permuted_samples[2], filters, 2);

  /* Narrow and re-pack. */
  sum = vqrshlq_s32(sum, shift_round_0);

  return vmovn_s32(sum);
}

static INLINE int16x8_t convolve12_8_sdot(
    uint8x16_t samples0, uint8x16_t samples1, const int8x16_t filters,
    const int32x4_t correction, const uint8x16_t range_limit,
    const uint8x16x3_t permute_tbl, const int32x4_t shift_round_0) {
  int8x16_t clamped_samples[2], permuted_samples[4];
  int32x4_t sum[2];

  /* Clamp sample range to [-128, 127] for 8-bit signed dot product. */
  clamped_samples[0] = vreinterpretq_s8_u8(vsubq_u8(samples0, range_limit));
  clamped_samples[1] = vreinterpretq_s8_u8(vsubq_u8(samples1, range_limit));

  /* Permute samples ready for dot product. */
  /* { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6 } */
  permuted_samples[0] = vqtbl1q_s8(clamped_samples[0], permute_tbl.val[0]);
  /* { 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10 } */
  permuted_samples[1] = vqtbl1q_s8(clamped_samples[0], permute_tbl.val[1]);
  /* { 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14 } */
  permuted_samples[2] = vqtbl1q_s8(clamped_samples[0], permute_tbl.val[2]);
  /* {12, 13, 14, 15, 13, 14, 15, 16, 14, 15, 16, 17, 15, 16, 17, 18 } */
  permuted_samples[3] = vqtbl1q_s8(clamped_samples[1], permute_tbl.val[2]);

  /* Accumulate dot product into 'correction' to account for range clamp. */
  /* First 4 output values. */
  sum[0] = vdotq_laneq_s32(correction, permuted_samples[0], filters, 0);
  sum[0] = vdotq_laneq_s32(sum[0], permuted_samples[1], filters, 1);
  sum[0] = vdotq_laneq_s32(sum[0], permuted_samples[2], filters, 2);
  /* Second 4 output values. */
  sum[1] = vdotq_laneq_s32(correction, permuted_samples[1], filters, 0);
  sum[1] = vdotq_laneq_s32(sum[1], permuted_samples[2], filters, 1);
  sum[1] = vdotq_laneq_s32(sum[1], permuted_samples[3], filters, 2);

  /* Narrow and re-pack. */
  sum[0] = vqrshlq_s32(sum[0], shift_round_0);
  sum[1] = vqrshlq_s32(sum[1], shift_round_0);

  return vcombine_s16(vmovn_s32(sum[0]), vmovn_s32(sum[1]));
}

static INLINE void av1_convolve_2d_sr_horiz_12tap_neon(
    const uint8_t *src_ptr, int src_stride, int16_t *dst_ptr,
    const int dst_stride, int w, int h, const int16x8_t x_filter_0_7,
    const int16x4_t x_filter_8_11, const int round_0) {
  const int bd = 8;

  // Special case the following no-op filter as 128 won't fit into the
  // 8-bit signed dot-product instruction:
  // { 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0 }
  if (vgetq_lane_s16(x_filter_0_7, 5) == 128) {
    const int16x8_t horiz_const = vdupq_n_s16((1 << (bd - 1)));
    const int16x8_t shift_round_0 = vdupq_n_s16(FILTER_BITS - round_0);
    // Undo the horizontal offset in the calling function.
    src_ptr += 5;

    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j += 8) {
        uint8x8_t s0 = vld1_u8(src_ptr + i * src_stride + j);
        uint16x8_t t0 = vaddw_u8(vreinterpretq_u16_s16(horiz_const), s0);
        int16x8_t d0 = vqrshlq_s16(vreinterpretq_s16_u16(t0), shift_round_0);
        if (w == 2) {
          vst1q_lane_s32((int32_t *)(dst_ptr + i * dst_stride),
                         vreinterpretq_s32_s16(d0), 0);
        } else if (w == 4) {
          vst1_s16(dst_ptr + i * dst_stride, vget_low_s16(d0));
        } else {
          vst1q_s16(dst_ptr + i * dst_stride + j, d0);
        }
      }
    }
  } else {
    const int32x4_t shift_round_0 = vdupq_n_s32(-round_0);

    // Narrow filter values to 8-bit.
    const int16x8x2_t x_filter_s16 = {
      { x_filter_0_7, vcombine_s16(x_filter_8_11, vdup_n_s16(0)) }
    };
    const int8x16_t x_filter = vcombine_s8(vmovn_s16(x_filter_s16.val[0]),
                                           vmovn_s16(x_filter_s16.val[1]));

    // Dot product constants.
    const int32_t horiz_const = (1 << (bd + FILTER_BITS - 1));
    const int32x4_t correct_tmp =
        vaddq_s32(vpaddlq_s16(vshlq_n_s16(x_filter_s16.val[0], 7)),
                  vpaddlq_s16(vshlq_n_s16(x_filter_s16.val[1], 7)));
    const int32x4_t correction =
        vdupq_n_s32(vaddvq_s32(correct_tmp) + horiz_const);
    const uint8x16_t range_limit = vdupq_n_u8(128);
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);

    if (w <= 4) {
      do {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          uint8x16_t s0, s1, s2, s3;
          int16x4_t d0, d1, d2, d3;

          s0 = vld1q_u8(s + 0 * src_stride);
          s1 = vld1q_u8(s + 1 * src_stride);
          s2 = vld1q_u8(s + 2 * src_stride);
          s3 = vld1q_u8(s + 3 * src_stride);

          d0 = convolve12_4_sdot(s0, x_filter, correction, range_limit,
                                 permute_tbl, shift_round_0);
          d1 = convolve12_4_sdot(s1, x_filter, correction, range_limit,
                                 permute_tbl, shift_round_0);
          d2 = convolve12_4_sdot(s2, x_filter, correction, range_limit,
                                 permute_tbl, shift_round_0);
          d3 = convolve12_4_sdot(s3, x_filter, correction, range_limit,
                                 permute_tbl, shift_round_0);

          if (w == 2) {
            vst1_lane_s32((int32_t *)(d + 0 * dst_stride),
                          vreinterpret_s32_s16(d0), 0);
            vst1_lane_s32((int32_t *)(d + 1 * dst_stride),
                          vreinterpret_s32_s16(d1), 0);
            vst1_lane_s32((int32_t *)(d + 2 * dst_stride),
                          vreinterpret_s32_s16(d2), 0);
            vst1_lane_s32((int32_t *)(d + 3 * dst_stride),
                          vreinterpret_s32_s16(d3), 0);
          } else {
            vst1_s16(d + 0 * dst_stride, d0);
            vst1_s16(d + 1 * dst_stride, d1);
            vst1_s16(d + 2 * dst_stride, d2);
            vst1_s16(d + 3 * dst_stride, d3);
          }

          s += 4;
          d += 4;
          width -= 4;
        } while (width > 0);

        src_ptr += 4 * src_stride;
        dst_ptr += 4 * dst_stride;
        h -= 4;
      } while (h >= 4);

      for (; h > 0; h--) {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          uint8x16_t s0;
          int16x4_t d0;

          s0 = vld1q_u8(s);

          d0 = convolve12_4_sdot(s0, x_filter, correction, range_limit,
                                 permute_tbl, shift_round_0);

          if (w == 2) {
            vst1_lane_s32((int32_t *)d, vreinterpret_s32_s16(d0), 0);
          } else {
            vst1_s16(d, d0);
          }

          s += 4;
          d += 4;
          width -= 4;
        } while (width > 0);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
      }
    } else {
      do {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          uint8x16_t s0[2], s1[2], s2[2], s3[2];
          int16x8_t d0, d1, d2, d3;

          s0[0] = vld1q_u8(s + 0 * src_stride);
          s1[0] = vld1q_u8(s + 1 * src_stride);
          s2[0] = vld1q_u8(s + 2 * src_stride);
          s3[0] = vld1q_u8(s + 3 * src_stride);
          s0[1] = vld1q_u8(s + 0 * src_stride + 4);
          s1[1] = vld1q_u8(s + 1 * src_stride + 4);
          s2[1] = vld1q_u8(s + 2 * src_stride + 4);
          s3[1] = vld1q_u8(s + 3 * src_stride + 4);

          d0 = convolve12_8_sdot(s0[0], s0[1], x_filter, correction,
                                 range_limit, permute_tbl, shift_round_0);
          d1 = convolve12_8_sdot(s1[0], s1[1], x_filter, correction,
                                 range_limit, permute_tbl, shift_round_0);
          d2 = convolve12_8_sdot(s2[0], s2[1], x_filter, correction,
                                 range_limit, permute_tbl, shift_round_0);
          d3 = convolve12_8_sdot(s3[0], s3[1], x_filter, correction,
                                 range_limit, permute_tbl, shift_round_0);

          vst1q_s16(d + 0 * dst_stride, d0);
          vst1q_s16(d + 1 * dst_stride, d1);
          vst1q_s16(d + 2 * dst_stride, d2);
          vst1q_s16(d + 3 * dst_stride, d3);

          s += 8;
          d += 8;
          width -= 8;
        } while (width > 0);

        src_ptr += 4 * src_stride;
        dst_ptr += 4 * dst_stride;
        h -= 4;
      } while (h >= 4);

      for (; h > 0; h--) {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          uint8x16_t s0[2];
          int16x8_t d0;

          s0[0] = vld1q_u8(s);
          s0[1] = vld1q_u8(s + 4);

          d0 = convolve12_8_sdot(s0[0], s0[1], x_filter, correction,
                                 range_limit, permute_tbl, shift_round_0);

          vst1q_s16(d, d0);

          s += 8;
          d += 8;
          width -= 8;
        } while (width > 0);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
      }
    }
  }
}

#else  // !(defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD))

static INLINE int16x4_t convolve12_horiz_4x4_s16(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
    const int16x4_t s6, const int16x4_t s7, const int16x4_t s8,
    const int16x4_t s9, const int16x4_t s10, const int16x4_t s11,
    const int16x8_t x_filter_0_7, const int16x4_t x_filter_8_11,
    const int32x4_t horiz_const, const int32x4_t shift_round_0) {
  const int16x4_t x_filter_0_3 = vget_low_s16(x_filter_0_7);
  const int16x4_t x_filter_4_7 = vget_high_s16(x_filter_0_7);
  int32x4_t sum;

  sum = horiz_const;
  sum = vmlal_lane_s16(sum, s0, x_filter_0_3, 0);
  sum = vmlal_lane_s16(sum, s1, x_filter_0_3, 1);
  sum = vmlal_lane_s16(sum, s2, x_filter_0_3, 2);
  sum = vmlal_lane_s16(sum, s3, x_filter_0_3, 3);
  sum = vmlal_lane_s16(sum, s4, x_filter_4_7, 0);
  sum = vmlal_lane_s16(sum, s5, x_filter_4_7, 1);
  sum = vmlal_lane_s16(sum, s6, x_filter_4_7, 2);
  sum = vmlal_lane_s16(sum, s7, x_filter_4_7, 3);
  sum = vmlal_lane_s16(sum, s8, x_filter_8_11, 0);
  sum = vmlal_lane_s16(sum, s9, x_filter_8_11, 1);
  sum = vmlal_lane_s16(sum, s10, x_filter_8_11, 2);
  sum = vmlal_lane_s16(sum, s11, x_filter_8_11, 3);

  sum = vqrshlq_s32(sum, shift_round_0);

  return vmovn_s32(sum);
}

// 4 column per iteration horizontal filtering for 12-tap convolve_2d_sr.
// Processes one row at a time.
static INLINE void horiz_filter_12tap_w4_single_row(
    const uint8_t *src_ptr, int src_stride, int16_t *dst_ptr,
    const int dst_stride, int w, int h, const int16x8_t x_filter_0_7,
    const int16x4_t x_filter_8_11, const int32x4_t horiz_const,
    const int32x4_t shift_round_0) {
  do {
    const uint8_t *s = src_ptr;
    int16_t *d = dst_ptr;
    int width = w;

    do {
      int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, d0;
      uint8x16_t t0;
      int16x8_t tt0, tt1;

      t0 = vld1q_u8(s);
      tt0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t0)));
      tt1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t0)));

      s0 = vget_low_s16(tt0);
      s4 = vget_high_s16(tt0);
      s8 = vget_low_s16(tt1);
      s12 = vget_high_s16(tt1);

      s1 = vext_s16(s0, s4, 1);    //  a1  a2  a3  a4
      s2 = vext_s16(s0, s4, 2);    //  a2  a3  a4  a5
      s3 = vext_s16(s0, s4, 3);    //  a3  a4  a5  a6
      s5 = vext_s16(s4, s8, 1);    //  a5  a6  a7  a8
      s6 = vext_s16(s4, s8, 2);    //  a6  a7  a8  a9
      s7 = vext_s16(s4, s8, 3);    //  a7  a8  a9 a10
      s9 = vext_s16(s8, s12, 1);   //  a9 a10 a11 a12
      s10 = vext_s16(s8, s12, 2);  // a10 a11 a12 a13
      s11 = vext_s16(s8, s12, 3);  // a11 a12 a13 a14

      d0 = convolve12_horiz_4x4_s16(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                                    s11, x_filter_0_7, x_filter_8_11,
                                    horiz_const, shift_round_0);

      if (w == 2) {
        vst1_lane_s32((int32_t *)d, vreinterpret_s32_s16(d0), 0);
      } else {
        vst1_s16(d, d0);
      }

      s += 4;
      d += 4;
      width -= 4;
    } while (width > 0);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
    h--;
  } while (h > 0);
}

static INLINE void av1_convolve_2d_sr_horiz_12tap_neon(
    const uint8_t *src_ptr, int src_stride, int16_t *dst_ptr,
    const int dst_stride, int w, int h, const int16x8_t x_filter_0_7,
    const int16x4_t x_filter_8_11, const int round_0) {
  const int bd = 8;
  const int32x4_t shift_round_0 = vdupq_n_s32(-(round_0));
  const int32x4_t horiz_const = vdupq_n_s32((1 << (bd + FILTER_BITS - 1)));

#if defined(__aarch64__)
  do {
    int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;
    uint8x8_t t0, t1, t2, t3;

    const uint8_t *s = src_ptr;
    int16_t *d = dst_ptr;
    int width = w;

    load_u8_8x4(s, src_stride, &t0, &t1, &t2, &t3);
    transpose_u8_8x4(&t0, &t1, &t2, &t3);

    s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
    s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
    s4 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    s5 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    s6 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
    s7 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));

    load_u8_8x4(s + 8, src_stride, &t0, &t1, &t2, &t3);
    transpose_u8_8x4(&t0, &t1, &t2, &t3);

    s8 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    s9 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    s10 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));

    s += 11;

    do {
      int16x4_t s11, s12, s13, s14, d0, d1, d2, d3;

      load_u8_8x4(s, src_stride, &t0, &t1, &t2, &t3);
      transpose_u8_8x4(&t0, &t1, &t2, &t3);

      s11 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
      s12 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
      s13 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
      s14 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));

      d0 = convolve12_horiz_4x4_s16(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                                    s11, x_filter_0_7, x_filter_8_11,
                                    horiz_const, shift_round_0);
      d1 = convolve12_horiz_4x4_s16(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                                    s11, s12, x_filter_0_7, x_filter_8_11,
                                    horiz_const, shift_round_0);
      d2 = convolve12_horiz_4x4_s16(s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
                                    s12, s13, x_filter_0_7, x_filter_8_11,
                                    horiz_const, shift_round_0);
      d3 = convolve12_horiz_4x4_s16(s3, s4, s5, s6, s7, s8, s9, s10, s11, s12,
                                    s13, s14, x_filter_0_7, x_filter_8_11,
                                    horiz_const, shift_round_0);

      transpose_s16_4x4d(&d0, &d1, &d2, &d3);

      if (w == 2) {
        vst1_lane_s32((int32_t *)(d + 0 * dst_stride), vreinterpret_s32_s16(d0),
                      0);
        vst1_lane_s32((int32_t *)(d + 1 * dst_stride), vreinterpret_s32_s16(d1),
                      0);
        vst1_lane_s32((int32_t *)(d + 2 * dst_stride), vreinterpret_s32_s16(d2),
                      0);
        vst1_lane_s32((int32_t *)(d + 3 * dst_stride), vreinterpret_s32_s16(d3),
                      0);
      } else {
        vst1_s16((d + 0 * dst_stride), d0);
        vst1_s16((d + 1 * dst_stride), d1);
        vst1_s16((d + 2 * dst_stride), d2);
        vst1_s16((d + 3 * dst_stride), d3);
      }

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      s7 = s11;
      s8 = s12;
      s9 = s13;
      s10 = s14;

      s += 4;
      d += 4;
      width -= 4;
    } while (width > 0);

    src_ptr += 4 * src_stride;
    dst_ptr += 4 * dst_stride;
    h -= 4;
  } while (h >= 4);

  if (h) {
    horiz_filter_12tap_w4_single_row(src_ptr, src_stride, dst_ptr, dst_stride,
                                     w, h, x_filter_0_7, x_filter_8_11,
                                     horiz_const, shift_round_0);
  }
#else   // !defined(__aarch64__)
  horiz_filter_12tap_w4_single_row(src_ptr, src_stride, dst_ptr, dst_stride, w,
                                   h, x_filter_0_7, x_filter_8_11, horiz_const,
                                   shift_round_0);
#endif  // defined(__aarch64__)
}

#endif  // defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

static INLINE void av1_convolve_2d_sr_vert_12tap_neon(
    int16_t *src_ptr, int src_stride, uint8_t *dst_ptr, int dst_stride, int w,
    int h, const int16x8_t y_filter_0_7, const int16x4_t y_filter_8_11,
    ConvolveParams *conv_params) {
  const int bd = 8;
  const int16_t round_bits =
      FILTER_BITS * 2 - conv_params->round_0 - conv_params->round_1;
  const int16x8_t vec_round_bits = vdupq_n_s16(-round_bits);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int32_t sub_const = (1 << (offset_bits - conv_params->round_1)) +
                            (1 << (offset_bits - conv_params->round_1 - 1));
  const int32x4_t round_shift_vec = vdupq_n_s32(-(conv_params->round_1));
  const int32x4_t offset_const = vdupq_n_s32(1 << offset_bits);
  const int32x4_t sub_const_vec = vdupq_n_s32(sub_const);

  if (w <= 4) {
    int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14;
    int16x4_t d0, d1, d2, d3;
    int16x8_t dd01, dd23;
    uint8x8_t d01, d23;

    load_s16_4x8(src_ptr, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    src_ptr += (8 * src_stride);
    load_s16_4x4(src_ptr, src_stride, &s8, &s9, &s10, &s11);
    src_ptr += (3 * src_stride);

    do {
      load_s16_4x4(src_ptr, src_stride, &s11, &s12, &s13, &s14);
      src_ptr += 4 * src_stride;

      d0 = convolve12_vert_4x4_s32(
          s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, y_filter_0_7,
          y_filter_8_11, round_shift_vec, offset_const, sub_const_vec);
      d1 = convolve12_vert_4x4_s32(
          s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, y_filter_0_7,
          y_filter_8_11, round_shift_vec, offset_const, sub_const_vec);
      d2 = convolve12_vert_4x4_s32(
          s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, y_filter_0_7,
          y_filter_8_11, round_shift_vec, offset_const, sub_const_vec);
      d3 = convolve12_vert_4x4_s32(
          s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, y_filter_0_7,
          y_filter_8_11, round_shift_vec, offset_const, sub_const_vec);

      dd01 = vqrshlq_s16(vcombine_s16(d0, d1), vec_round_bits);
      dd23 = vqrshlq_s16(vcombine_s16(d2, d3), vec_round_bits);

      d01 = vqmovun_s16(dd01);
      d23 = vqmovun_s16(dd23);

      if (w == 2) {
        vst1_lane_u16((uint16_t *)dst_ptr, vreinterpret_u16_u8(d01), 0);
        dst_ptr += dst_stride;
        vst1_lane_u16((uint16_t *)dst_ptr, vreinterpret_u16_u8(d01), 2);
        dst_ptr += dst_stride;
        if (h != 2) {
          vst1_lane_u16((uint16_t *)dst_ptr, vreinterpret_u16_u8(d23), 0);
          dst_ptr += dst_stride;
          vst1_lane_u16((uint16_t *)dst_ptr, vreinterpret_u16_u8(d23), 2);
          dst_ptr += dst_stride;
        }
      } else {
        vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d01), 0);
        dst_ptr += dst_stride;
        vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d01), 1);
        dst_ptr += dst_stride;
        if (h != 2) {
          vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d23), 0);
          dst_ptr += dst_stride;
          vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d23), 1);
          dst_ptr += dst_stride;
        }
      }

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      s7 = s11;
      s8 = s12;
      s9 = s13;
      s10 = s14;
      h -= 4;
    } while (h > 0);

  } else {
    do {
      int16x8_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14;
      uint8x8_t d0, d1, d2, d3;

      int16_t *s = src_ptr;
      uint8_t *d = dst_ptr;

      int height = h;

      load_s16_8x8(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
      s += (8 * src_stride);
      load_s16_8x4(s, src_stride, &s8, &s9, &s10, &s11);
      s += (3 * src_stride);

      do {
        load_s16_8x4(s, src_stride, &s11, &s12, &s13, &s14);
        s += 4 * src_stride;

        d0 = convolve12_vert_8x4_s32(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9,
                                     s10, s11, y_filter_0_7, y_filter_8_11,
                                     round_shift_vec, offset_const,
                                     sub_const_vec, vec_round_bits);
        d1 = convolve12_vert_8x4_s32(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                                     s11, s12, y_filter_0_7, y_filter_8_11,
                                     round_shift_vec, offset_const,
                                     sub_const_vec, vec_round_bits);
        d2 = convolve12_vert_8x4_s32(s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
                                     s12, s13, y_filter_0_7, y_filter_8_11,
                                     round_shift_vec, offset_const,
                                     sub_const_vec, vec_round_bits);
        d3 = convolve12_vert_8x4_s32(s3, s4, s5, s6, s7, s8, s9, s10, s11, s12,
                                     s13, s14, y_filter_0_7, y_filter_8_11,
                                     round_shift_vec, offset_const,
                                     sub_const_vec, vec_round_bits);

        vst1_u8(d, d0);
        d += dst_stride;
        vst1_u8(d, d1);
        d += dst_stride;
        if (h != 2) {
          vst1_u8(d, d2);
          d += dst_stride;
          vst1_u8(d, d3);
          d += dst_stride;
        }

        s0 = s4;
        s1 = s5;
        s2 = s6;
        s3 = s7;
        s4 = s8;
        s5 = s9;
        s6 = s10;
        s7 = s11;
        s8 = s12;
        s9 = s13;
        s10 = s14;
        height -= 4;
      } while (height > 0);

      src_ptr += 8;
      dst_ptr += 8;
      w -= 8;
    } while (w > 0);
  }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)

static INLINE void av1_convolve_2d_sr_horiz_neon(
    const uint8_t *src, int src_stride, int16_t *im_block, int im_stride, int w,
    int im_h, const int16x8_t x_filter_s16, const int round_0) {
  const int bd = 8;

  const uint8_t *src_ptr = src;
  int16_t *dst_ptr = im_block;
  int dst_stride = im_stride;

  int height = im_h;

  // Filter values are even, so downshift by 1 to reduce intermediate precision
  // requirements.
  const int8x8_t x_filter = vshrn_n_s16(x_filter_s16, 1);
  const int32x4_t horiz_const = vdupq_n_s32(1 << (bd + FILTER_BITS - 2));

  assert(round_0 > 0);

  if (w <= 4) {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);
    const int16x4_t shift_round_0 = vdup_n_s16(-(round_0 - 1));
    uint8x16_t s0, s1, s2, s3;
    int32x4_t t0, t1, t2, t3;
    int16x4_t d0, d1, d2, d3;

    do {
      assert(height >= 4);

      load_u8_8x16(src_ptr, src_stride, &s0, &s1, &s2, &s3);

      t0 = convolve8_4_usdot(s0, x_filter, permute_tbl, horiz_const);
      t1 = convolve8_4_usdot(s1, x_filter, permute_tbl, horiz_const);
      t2 = convolve8_4_usdot(s2, x_filter, permute_tbl, horiz_const);
      t3 = convolve8_4_usdot(s3, x_filter, permute_tbl, horiz_const);

      d0 = vqrshl_s16(vmovn_s32(t0), shift_round_0);
      d1 = vqrshl_s16(vmovn_s32(t1), shift_round_0);
      d2 = vqrshl_s16(vmovn_s32(t2), shift_round_0);
      d3 = vqrshl_s16(vmovn_s32(t3), shift_round_0);

      if (w == 2) {
        vst1_lane_u32((uint32_t *)(dst_ptr + 0 * dst_stride),
                      vreinterpret_u32_s16(d0), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 1 * dst_stride),
                      vreinterpret_u32_s16(d1), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 2 * dst_stride),
                      vreinterpret_u32_s16(d2), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 3 * dst_stride),
                      vreinterpret_u32_s16(d3), 0);
      } else {
        vst1_s16(dst_ptr + 0 * dst_stride, d0);
        vst1_s16(dst_ptr + 1 * dst_stride, d1);
        vst1_s16(dst_ptr + 2 * dst_stride, d2);
        vst1_s16(dst_ptr + 3 * dst_stride, d3);
      }

      src_ptr += 4 * src_stride;
      dst_ptr += 4 * dst_stride;
      height -= 4;
    } while (height >= 4);

    if (height) {
      assert(height < 4);

      do {
        s0 = vld1q_u8(src_ptr);
        t0 = convolve8_4_usdot(s0, x_filter, permute_tbl, horiz_const);
        d0 = vqrshl_s16(vmovn_s32(t0), shift_round_0);

        if (w == 2) {
          vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_s16(d0), 0);
        } else {
          vst1_s16(dst_ptr, d0);
        }

        src_ptr += src_stride;
        dst_ptr += dst_stride;
        height--;
      } while (height > 0);
    }
  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);
    const int16x8_t shift_round_0 = vdupq_n_s16(-(round_0 - 1));
    uint8x16_t s0, s1, s2, s3;
    int16x8_t d0, d1, d2, d3;

    do {
      assert(height >= 4);

      const uint8_t *s = src_ptr;
      int16_t *d = dst_ptr;
      int width = w;

      do {
        s0 = vld1q_u8(s + 0 * src_stride);
        s1 = vld1q_u8(s + 1 * src_stride);
        s2 = vld1q_u8(s + 2 * src_stride);
        s3 = vld1q_u8(s + 3 * src_stride);

        d0 = convolve8_8_usdot(s0, x_filter, permute_tbl, horiz_const,
                               shift_round_0);
        d1 = convolve8_8_usdot(s1, x_filter, permute_tbl, horiz_const,
                               shift_round_0);
        d2 = convolve8_8_usdot(s2, x_filter, permute_tbl, horiz_const,
                               shift_round_0);
        d3 = convolve8_8_usdot(s3, x_filter, permute_tbl, horiz_const,
                               shift_round_0);

        vst1q_s16(d + 0 * dst_stride, d0);
        vst1q_s16(d + 1 * dst_stride, d1);
        vst1q_s16(d + 2 * dst_stride, d2);
        vst1q_s16(d + 3 * dst_stride, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);

      src_ptr += 4 * src_stride;
      dst_ptr += 4 * dst_stride;
      height -= 4;
    } while (height >= 4);

    if (height) {
      assert(height < 4);

      do {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          s0 = vld1q_u8(s);
          d0 = convolve8_8_usdot(s0, x_filter, permute_tbl, horiz_const,
                                 shift_round_0);
          vst1q_s16(d, d0);

          s += 8;
          d += 8;
          width -= 8;
        } while (width > 0);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
        height--;
      } while (height > 0);
    }
  }
}

#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

static INLINE void av1_convolve_2d_sr_horiz_neon(
    const uint8_t *src, int src_stride, int16_t *im_block, int im_stride, int w,
    int im_h, const int16x8_t x_filter_s16, const int round_0) {
  const int bd = 8;

  const uint8_t *src_ptr = src;
  int16_t *dst_ptr = im_block;
  int dst_stride = im_stride;

  int height = im_h;

  // Filter values are even, so downshift by 1 to reduce intermediate precision
  // requirements.
  const int8x8_t x_filter = vshrn_n_s16(x_filter_s16, 1);
  const int32_t horiz_const = (1 << (bd + FILTER_BITS - 2));
  // Dot product constants.
  const int16x8_t correct_tmp = vshlq_n_s16(x_filter_s16, 6);
  const int32x4_t correction =
      vdupq_n_s32(vaddlvq_s16(correct_tmp) + horiz_const);
  const uint8x16_t range_limit = vdupq_n_u8(128);

  assert(round_0 > 0);

  if (w <= 4) {
    const uint8x16x2_t permute_tbl = vld1q_u8_x2(dot_prod_permute_tbl);
    const int16x4_t shift_round_0 = vdup_n_s16(-(round_0 - 1));
    uint8x16_t s0, s1, s2, s3;
    int32x4_t t0, t1, t2, t3;
    int16x4_t d0, d1, d2, d3;

    do {
      assert(height >= 4);

      load_u8_8x16(src_ptr, src_stride, &s0, &s1, &s2, &s3);

      t0 = convolve8_4_sdot(s0, x_filter, correction, range_limit, permute_tbl);
      t1 = convolve8_4_sdot(s1, x_filter, correction, range_limit, permute_tbl);
      t2 = convolve8_4_sdot(s2, x_filter, correction, range_limit, permute_tbl);
      t3 = convolve8_4_sdot(s3, x_filter, correction, range_limit, permute_tbl);

      d0 = vqrshl_s16(vmovn_s32(t0), shift_round_0);
      d1 = vqrshl_s16(vmovn_s32(t1), shift_round_0);
      d2 = vqrshl_s16(vmovn_s32(t2), shift_round_0);
      d3 = vqrshl_s16(vmovn_s32(t3), shift_round_0);

      if (w == 2) {
        vst1_lane_u32((uint32_t *)(dst_ptr + 0 * dst_stride),
                      vreinterpret_u32_s16(d0), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 1 * dst_stride),
                      vreinterpret_u32_s16(d1), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 2 * dst_stride),
                      vreinterpret_u32_s16(d2), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 3 * dst_stride),
                      vreinterpret_u32_s16(d3), 0);
      } else {
        vst1_s16(dst_ptr + 0 * dst_stride, d0);
        vst1_s16(dst_ptr + 1 * dst_stride, d1);
        vst1_s16(dst_ptr + 2 * dst_stride, d2);
        vst1_s16(dst_ptr + 3 * dst_stride, d3);
      }

      src_ptr += 4 * src_stride;
      dst_ptr += 4 * dst_stride;
      height -= 4;
    } while (height >= 4);

    if (height) {
      assert(height < 4);

      do {
        s0 = vld1q_u8(src_ptr);
        t0 = convolve8_4_sdot(s0, x_filter, correction, range_limit,
                              permute_tbl);
        d0 = vqrshl_s16(vmovn_s32(t0), shift_round_0);

        if (w == 2) {
          vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_s16(d0), 0);
        } else {
          vst1_s16(dst_ptr, d0);
        }

        src_ptr += src_stride;
        dst_ptr += dst_stride;
        height--;
      } while (height > 0);
    }
  } else {
    const uint8x16x3_t permute_tbl = vld1q_u8_x3(dot_prod_permute_tbl);
    const int16x8_t shift_round_0 = vdupq_n_s16(-(round_0 - 1));
    uint8x16_t s0, s1, s2, s3;
    int16x8_t d0, d1, d2, d3;

    do {
      assert(height >= 4);

      const uint8_t *s = src_ptr;
      int16_t *d = dst_ptr;
      int width = w;

      do {
        s0 = vld1q_u8(s + 0 * src_stride);
        s1 = vld1q_u8(s + 1 * src_stride);
        s2 = vld1q_u8(s + 2 * src_stride);
        s3 = vld1q_u8(s + 3 * src_stride);

        d0 = convolve8_8_sdot(s0, x_filter, correction, range_limit,
                              permute_tbl, shift_round_0);
        d1 = convolve8_8_sdot(s1, x_filter, correction, range_limit,
                              permute_tbl, shift_round_0);
        d2 = convolve8_8_sdot(s2, x_filter, correction, range_limit,
                              permute_tbl, shift_round_0);
        d3 = convolve8_8_sdot(s3, x_filter, correction, range_limit,
                              permute_tbl, shift_round_0);

        vst1q_s16(d + 0 * dst_stride, d0);
        vst1q_s16(d + 1 * dst_stride, d1);
        vst1q_s16(d + 2 * dst_stride, d2);
        vst1q_s16(d + 3 * dst_stride, d3);

        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);

      src_ptr += 4 * src_stride;
      dst_ptr += 4 * dst_stride;
      height -= 4;
    } while (height >= 4);

    if (height) {
      assert(height < 4);

      do {
        const uint8_t *s = src_ptr;
        int16_t *d = dst_ptr;
        int width = w;

        do {
          s0 = vld1q_u8(s);
          d0 = convolve8_8_sdot(s0, x_filter, correction, range_limit,
                                permute_tbl, shift_round_0);
          vst1q_s16(d, d0);

          s += 8;
          d += 8;
          width -= 8;
        } while (width > 0);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
        height--;
      } while (height > 0);
    }
  }
}

#else  // !(defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD))

// Horizontal filtering for convolve_2d_sr for width multiple of 8
// Processes one row at a time
static INLINE void horiz_filter_w8_single_row(
    const uint8_t *src_ptr, int src_stride, int16_t *dst_ptr,
    const int dst_stride, int width, int height, const int16x8_t x_filter,
    const int16x8_t horiz_const, const int16x8_t shift_round_0) {
  int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
  do {
    uint8x8_t t0 = vld1_u8(src_ptr);
    s0 = vreinterpretq_s16_u16(vmovl_u8(t0));  // a0 a1 a2 a3 a4 a5 a6 a7

    int width_tmp = width;
    const uint8_t *s = src_ptr + 8;
    int16_t *dst_tmp = dst_ptr;

    __builtin_prefetch(dst_ptr);

    do {
      t0 = vld1_u8(s);  // a8 a9 a10 a11 a12 a13 a14 a15
      s7 = vreinterpretq_s16_u16(vmovl_u8(t0));
      int16x8_t sum = s0;
      s0 = s7;

      s1 = vextq_s16(sum, s7, 1);  // a1 a2 a3 a4 a5 a6 a7 a8
      s2 = vextq_s16(sum, s7, 2);  // a2 a3 a4 a5 a6 a7 a8 a9
      s3 = vextq_s16(sum, s7, 3);  // a3 a4 a5 a6 a7 a8 a9 a10
      s4 = vextq_s16(sum, s7, 4);  // a4 a5 a6 a7 a8 a9 a10 a11
      s5 = vextq_s16(sum, s7, 5);  // a5 a6 a7 a8 a9 a10 a11 a12
      s6 = vextq_s16(sum, s7, 6);  // a6 a7 a8 a9 a10 a11 a12 a13
      s7 = vextq_s16(sum, s7, 7);  // a7 a8 a9 a10 a11 a12 a13 a14

      int16x8_t res0 = convolve8_8x8_s16(sum, s1, s2, s3, s4, s5, s6, s7,
                                         x_filter, horiz_const, shift_round_0);

      vst1q_s16(dst_tmp, res0);

      s += 8;
      dst_tmp += 8;
      width_tmp -= 8;
    } while (width_tmp > 0);
    src_ptr += src_stride;
    dst_ptr += dst_stride;
    height--;
  } while (height > 0);
}

// Horizontal filtering for convolve_2d_sr for width <= 4
// Processes one row at a time
static INLINE void horiz_filter_w4_single_row(
    const uint8_t *src_ptr, int src_stride, int16_t *dst_ptr,
    const int dst_stride, int width, int height, const int16x8_t x_filter,
    const int16x4_t horiz_const, const int16x4_t shift_round_0) {
  int16x4_t s0, s1, s2, s3, s4, s5, s6, s7;
  do {
    const uint8_t *s = src_ptr;

    __builtin_prefetch(s);

    uint8x8_t t0 = vld1_u8(s);  // a0 a1 a2 a3 a4 a5 a6 a7
    int16x8_t tt0 = vreinterpretq_s16_u16(vmovl_u8(t0));
    s0 = vget_low_s16(tt0);
    s4 = vget_high_s16(tt0);

    __builtin_prefetch(dst_ptr);
    s += 8;

    t0 = vld1_u8(s);  // a8 a9 a10 a11 a12 a13 a14 a15
    s7 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));

    s1 = vext_s16(s0, s4, 1);  // a1 a2 a3 a4
    s2 = vext_s16(s0, s4, 2);  // a2 a3 a4 a5
    s3 = vext_s16(s0, s4, 3);  // a3 a4 a5 a6
    s5 = vext_s16(s4, s7, 1);  // a5 a6 a7 a8
    s6 = vext_s16(s4, s7, 2);  // a6 a7 a8 a9
    s7 = vext_s16(s4, s7, 3);  // a7 a8 a9 a10

    int16x4_t d0 = convolve8_4x4_s16(s0, s1, s2, s3, s4, s5, s6, s7, x_filter,
                                     horiz_const, shift_round_0);

    if (width == 2) {
      vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_s16(d0), 0);
    } else {
      vst1_s16(dst_ptr, d0);
    }

    dst_ptr += dst_stride;
    src_ptr += src_stride;
    height--;
  } while (height > 0);
}

static INLINE void av1_convolve_2d_sr_horiz_neon(
    const uint8_t *src, int src_stride, int16_t *im_block, int im_stride, int w,
    int im_h, const int16x8_t x_filter_s16, const int round_0) {
  const int bd = 8;

  const uint8_t *src_ptr = src;
  int16_t *dst_ptr = im_block;
  int dst_stride = im_stride;

  int height = im_h;

  // Filter values are even, so downshift by 1 to reduce intermediate precision
  // requirements.
  const int16x8_t x_filter = vshrq_n_s16(x_filter_s16, 1);

  assert(round_0 > 0);

  if (w <= 4) {
    const int16x4_t horiz_const = vdup_n_s16((1 << (bd + FILTER_BITS - 2)));
    const int16x4_t shift_round_0 = vdup_n_s16(-(round_0 - 1));

#if defined(__aarch64__)
    do {
      int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, d0, d1, d2, d3;
      uint8x8_t t0, t1, t2, t3;
      const uint8_t *s = src_ptr;

      assert(height >= 4);

      load_u8_8x4(s, src_stride, &t0, &t1, &t2, &t3);
      transpose_u8_8x4(&t0, &t1, &t2, &t3);

      s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
      s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
      s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
      s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
      s4 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
      s5 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
      s6 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));

      s += 7;

      load_u8_8x4(s, src_stride, &t0, &t1, &t2, &t3);
      transpose_u8_8x4(&t0, &t1, &t2, &t3);

      s7 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
      s8 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
      s9 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
      s10 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));

      d0 = convolve8_4x4_s16(s0, s1, s2, s3, s4, s5, s6, s7, x_filter,
                             horiz_const, shift_round_0);
      d1 = convolve8_4x4_s16(s1, s2, s3, s4, s5, s6, s7, s8, x_filter,
                             horiz_const, shift_round_0);
      d2 = convolve8_4x4_s16(s2, s3, s4, s5, s6, s7, s8, s9, x_filter,
                             horiz_const, shift_round_0);
      d3 = convolve8_4x4_s16(s3, s4, s5, s6, s7, s8, s9, s10, x_filter,
                             horiz_const, shift_round_0);

      transpose_s16_4x4d(&d0, &d1, &d2, &d3);

      if (w == 2) {
        vst1_lane_u32((uint32_t *)(dst_ptr + 0 * dst_stride),
                      vreinterpret_u32_s16(d0), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 1 * dst_stride),
                      vreinterpret_u32_s16(d1), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 2 * dst_stride),
                      vreinterpret_u32_s16(d2), 0);
        vst1_lane_u32((uint32_t *)(dst_ptr + 3 * dst_stride),
                      vreinterpret_u32_s16(d3), 0);
      } else {
        vst1_s16((dst_ptr + 0 * dst_stride), d0);
        vst1_s16((dst_ptr + 1 * dst_stride), d1);
        vst1_s16((dst_ptr + 2 * dst_stride), d2);
        vst1_s16((dst_ptr + 3 * dst_stride), d3);
      }

      src_ptr += 4 * src_stride;
      dst_ptr += 4 * dst_stride;
      height -= 4;
    } while (height >= 4);

    if (height) {
      assert(height < 4);
      horiz_filter_w4_single_row(src_ptr, src_stride, dst_ptr, dst_stride, w,
                                 height, x_filter, horiz_const, shift_round_0);
    }

#else   // !defined(__aarch64__)
    horiz_filter_w4_single_row(src_ptr, src_stride, dst_ptr, dst_stride, w,
                               height, x_filter, horiz_const, shift_round_0);
#endif  // defined(__aarch64__)

  } else {
    const int16x8_t horiz_const = vdupq_n_s16((1 << (bd + FILTER_BITS - 2)));
    const int16x8_t shift_round_0 = vdupq_n_s16(-(round_0 - 1));

#if defined(__aarch64__)

    for (; height >= 8; height -= 8) {
      int16x8_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
          d0, d1, d2, d3, d4, d5, d6, d7;
      uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7;

      const uint8_t *s = src_ptr;
      int16_t *d = dst_ptr;
      int width = w;

      load_u8_8x8(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

      transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

      s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
      s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
      s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
      s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
      s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
      s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
      s6 = vreinterpretq_s16_u16(vmovl_u8(t6));

      s += 7;

      do {
        load_u8_8x8(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        transpose_u8_8x8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        s7 = vreinterpretq_s16_u16(vmovl_u8(t0));
        s8 = vreinterpretq_s16_u16(vmovl_u8(t1));
        s9 = vreinterpretq_s16_u16(vmovl_u8(t2));
        s10 = vreinterpretq_s16_u16(vmovl_u8(t3));
        s11 = vreinterpretq_s16_u16(vmovl_u8(t4));
        s12 = vreinterpretq_s16_u16(vmovl_u8(t5));
        s13 = vreinterpretq_s16_u16(vmovl_u8(t6));
        s14 = vreinterpretq_s16_u16(vmovl_u8(t7));

        d0 = convolve8_8x8_s16(s0, s1, s2, s3, s4, s5, s6, s7, x_filter,
                               horiz_const, shift_round_0);
        d1 = convolve8_8x8_s16(s1, s2, s3, s4, s5, s6, s7, s8, x_filter,
                               horiz_const, shift_round_0);
        d2 = convolve8_8x8_s16(s2, s3, s4, s5, s6, s7, s8, s9, x_filter,
                               horiz_const, shift_round_0);
        d3 = convolve8_8x8_s16(s3, s4, s5, s6, s7, s8, s9, s10, x_filter,
                               horiz_const, shift_round_0);
        d4 = convolve8_8x8_s16(s4, s5, s6, s7, s8, s9, s10, s11, x_filter,
                               horiz_const, shift_round_0);
        d5 = convolve8_8x8_s16(s5, s6, s7, s8, s9, s10, s11, s12, x_filter,
                               horiz_const, shift_round_0);
        d6 = convolve8_8x8_s16(s6, s7, s8, s9, s10, s11, s12, s13, x_filter,
                               horiz_const, shift_round_0);
        d7 = convolve8_8x8_s16(s7, s8, s9, s10, s11, s12, s13, s14, x_filter,
                               horiz_const, shift_round_0);

        transpose_s16_8x8(&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

        store_s16_8x8(d, dst_stride, d0, d1, d2, d3, d4, d5, d6, d7);

        s0 = s8;
        s1 = s9;
        s2 = s10;
        s3 = s11;
        s4 = s12;
        s5 = s13;
        s6 = s14;
        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);

      src_ptr += 8 * src_stride;
      dst_ptr += 8 * dst_stride;
    }

    for (; height >= 4; height -= 4) {
      int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
          dd0, dd1, dd2, dd3, dd4, dd5, dd6, dd7;
      int16x8_t d0, d1, d2, d3;
      uint8x8_t t0, t1, t2, t3;

      const uint8_t *s = src_ptr;
      int16_t *d = dst_ptr;
      int width = w;

      load_u8_8x4(src_ptr, src_stride, &t0, &t1, &t2, &t3);
      transpose_u8_8x4(&t0, &t1, &t2, &t3);

      s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
      s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
      s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
      s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
      s4 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
      s5 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
      s6 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));

      s += 7;

      do {
        load_u8_8x4(s, src_stride, &t0, &t1, &t2, &t3);
        transpose_u8_8x4(&t0, &t1, &t2, &t3);

        s7 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
        s8 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
        s9 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
        s10 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
        s11 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
        s12 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
        s13 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
        s14 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));

        dd0 = convolve8_4x4(s0, s1, s2, s3, s4, s5, s6, s7, x_filter);
        dd1 = convolve8_4x4(s1, s2, s3, s4, s5, s6, s7, s8, x_filter);
        dd2 = convolve8_4x4(s2, s3, s4, s5, s6, s7, s8, s9, x_filter);
        dd3 = convolve8_4x4(s3, s4, s5, s6, s7, s8, s9, s10, x_filter);
        dd4 = convolve8_4x4(s4, s5, s6, s7, s8, s9, s10, s11, x_filter);
        dd5 = convolve8_4x4(s5, s6, s7, s8, s9, s10, s11, s12, x_filter);
        dd6 = convolve8_4x4(s6, s7, s8, s9, s10, s11, s12, s13, x_filter);
        dd7 = convolve8_4x4(s7, s8, s9, s10, s11, s12, s13, s14, x_filter);

        transpose_s16_4x8(&dd0, &dd1, &dd2, &dd3, &dd4, &dd5, &dd6, &dd7, &d0,
                          &d1, &d2, &d3);

        d0 = vaddq_s16(d0, horiz_const);
        d1 = vaddq_s16(d1, horiz_const);
        d2 = vaddq_s16(d2, horiz_const);
        d3 = vaddq_s16(d3, horiz_const);

        d0 = vqrshlq_s16(d0, shift_round_0);
        d1 = vqrshlq_s16(d1, shift_round_0);
        d2 = vqrshlq_s16(d2, shift_round_0);
        d3 = vqrshlq_s16(d3, shift_round_0);

        store_s16_8x4(d, dst_stride, d0, d1, d2, d3);

        s0 = s8;
        s1 = s9;
        s2 = s10;
        s3 = s11;
        s4 = s12;
        s5 = s13;
        s6 = s14;
        s += 8;
        d += 8;
        width -= 8;
      } while (width > 0);

      src_ptr += 4 * src_stride;
      dst_ptr += 4 * dst_stride;
    }

    if (height) {
      assert(height < 4);
      horiz_filter_w8_single_row(src_ptr, src_stride, dst_ptr, dst_stride, w,
                                 height, x_filter, horiz_const, shift_round_0);
    }

#else   // !defined(__aarch64__)
    horiz_filter_w8_single_row(src_ptr, src_stride, dst_ptr, dst_stride, w,
                               height, x_filter, horiz_const, shift_round_0);
#endif  // defined(__aarch64__)
  }
}

#endif  // defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

static INLINE void av1_convolve_2d_sr_vert_8tap_neon(
    int16_t *src_ptr, int src_stride, uint8_t *dst_ptr, int dst_stride, int w,
    int h, const int16x8_t y_filter, ConvolveParams *conv_params) {
  const int bd = 8;
  const int16_t round_bits =
      FILTER_BITS * 2 - conv_params->round_0 - conv_params->round_1;
  const int16x8_t vec_round_bits = vdupq_n_s16(-round_bits);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;

  const int32_t sub_const = (1 << (offset_bits - conv_params->round_1)) +
                            (1 << (offset_bits - conv_params->round_1 - 1));

  const int32x4_t round_shift_vec = vdupq_n_s32(-(conv_params->round_1));
  const int32x4_t offset_const = vdupq_n_s32(1 << offset_bits);
  const int32x4_t sub_const_vec = vdupq_n_s32(sub_const);

  if (w <= 4) {
    int16x4_t s0, s1, s2, s3, s4, s5, s6, s7, d0;
    int16x8_t dd0;
    uint8x8_t d01;

#if defined(__aarch64__)
    int16x4_t s8, s9, s10, d1, d2, d3;
    int16x8_t dd1;
    uint8x8_t d23;
#endif  // defined(__aarch64__)

    int16_t *s = src_ptr;
    uint8_t *d = dst_ptr;

    load_s16_4x8(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    s += (7 * src_stride);

    do {
#if defined(__aarch64__)
      load_s16_4x4(s, src_stride, &s7, &s8, &s9, &s10);
      s += (4 * src_stride);

      d0 = convolve8_vert_4x4_s32(s0, s1, s2, s3, s4, s5, s6, s7, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);
      d1 = convolve8_vert_4x4_s32(s1, s2, s3, s4, s5, s6, s7, s8, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);
      d2 = convolve8_vert_4x4_s32(s2, s3, s4, s5, s6, s7, s8, s9, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);
      d3 = convolve8_vert_4x4_s32(s3, s4, s5, s6, s7, s8, s9, s10, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);

      dd0 = vqrshlq_s16(vcombine_s16(d0, d1), vec_round_bits);
      dd1 = vqrshlq_s16(vcombine_s16(d2, d3), vec_round_bits);

      d01 = vqmovun_s16(dd0);
      d23 = vqmovun_s16(dd1);

      if (w == 4) {
        vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d01), 0);
        d += dst_stride;
        vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d01), 1);
        d += dst_stride;
        if (h != 2) {
          vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d23), 0);
          d += dst_stride;
          vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d23), 1);
          d += dst_stride;
        }
      } else {
        vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d01), 0);
        d += dst_stride;
        vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d01), 2);
        d += dst_stride;
        if (h != 2) {
          vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d23), 0);
          d += dst_stride;
          vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d23), 2);
          d += dst_stride;
        }
      }

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = s10;
      h -= 4;
#else   // !defined(__aarch64__)
      s7 = vld1_s16(s);
      s += src_stride;

      d0 = convolve8_vert_4x4_s32(s0, s1, s2, s3, s4, s5, s6, s7, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);

      dd0 = vqrshlq_s16(vcombine_s16(d0, d0), vec_round_bits);
      d01 = vqmovun_s16(dd0);

      if (w == 2) {
        vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d01), 0);
        d += dst_stride;
      } else {
        vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d01), 0);
        d += dst_stride;
      }

      s0 = s1;
      s1 = s2;
      s2 = s3;
      s3 = s4;
      s4 = s5;
      s5 = s6;
      s6 = s7;
      h--;
#endif  // defined(__aarch64__)
    } while (h > 0);
  } else {
    // if width is a multiple of 8 & height is a multiple of 4
    int16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
    uint8x8_t d0;
#if defined(__aarch64__)
    int16x8_t s8, s9, s10;
    uint8x8_t d1, d2, d3;
#endif  // defined(__aarch64__)

    do {
      int height = h;
      int16_t *s = src_ptr;
      uint8_t *d = dst_ptr;

      load_s16_8x8(s, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
      s += (7 * src_stride);

      do {
#if defined(__aarch64__)
        load_s16_8x4(s, src_stride, &s7, &s8, &s9, &s10);
        s += (4 * src_stride);

        d0 = convolve8_vert_8x4_s32(s0, s1, s2, s3, s4, s5, s6, s7, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);
        d1 = convolve8_vert_8x4_s32(s1, s2, s3, s4, s5, s6, s7, s8, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);
        d2 = convolve8_vert_8x4_s32(s2, s3, s4, s5, s6, s7, s8, s9, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);
        d3 = convolve8_vert_8x4_s32(s3, s4, s5, s6, s7, s8, s9, s10, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);

        vst1_u8(d, d0);
        d += dst_stride;
        vst1_u8(d, d1);
        d += dst_stride;
        if (h != 2) {
          vst1_u8(d, d2);
          d += dst_stride;
          vst1_u8(d, d3);
          d += dst_stride;
        }

        s0 = s4;
        s1 = s5;
        s2 = s6;
        s3 = s7;
        s4 = s8;
        s5 = s9;
        s6 = s10;
        height -= 4;
#else   // !defined(__aarch64__)
        s7 = vld1q_s16(s);
        s += src_stride;

        d0 = convolve8_vert_8x4_s32(s0, s1, s2, s3, s4, s5, s6, s7, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);

        vst1_u8(d, d0);
        d += dst_stride;

        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s5;
        s5 = s6;
        s6 = s7;
        height--;
#endif  // defined(__aarch64__)
      } while (height > 0);

      src_ptr += 8;
      dst_ptr += 8;
      w -= 8;
    } while (w > 0);
  }
}

static INLINE int16x4_t convolve6_vert_4x4_s32(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t s4, const int16x4_t s5,
    const int16x8_t y_filter, const int32x4_t round_shift_vec,
    const int32x4_t offset_const, const int32x4_t sub_const_vec) {
  const int16x4_t y_filter_lo = vget_low_s16(y_filter);
  const int16x4_t y_filter_hi = vget_high_s16(y_filter);
  int32x4_t sum;

  sum = vmull_lane_s16(s0, y_filter_lo, 1);
  sum = vmlal_lane_s16(sum, s1, y_filter_lo, 2);
  sum = vmlal_lane_s16(sum, s2, y_filter_lo, 3);
  sum = vmlal_lane_s16(sum, s3, y_filter_hi, 0);
  sum = vmlal_lane_s16(sum, s4, y_filter_hi, 1);
  sum = vmlal_lane_s16(sum, s5, y_filter_hi, 2);

  sum = vaddq_s32(sum, offset_const);
  sum = vqrshlq_s32(sum, round_shift_vec);
  sum = vsubq_s32(sum, sub_const_vec);

  return vmovn_s32(sum);
}

static INLINE uint8x8_t convolve6_vert_8x4_s32(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
    const int16x8_t y_filter, const int32x4_t round_shift_vec,
    const int32x4_t offset_const, const int32x4_t sub_const_vec,
    const int16x8_t vec_round_bits) {
  const int16x4_t y_filter_lo = vget_low_s16(y_filter);
  const int16x4_t y_filter_hi = vget_high_s16(y_filter);
  int32x4_t sum0, sum1;
  int16x8_t res;

  sum0 = vmull_lane_s16(vget_low_s16(s0), y_filter_lo, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s1), y_filter_lo, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s2), y_filter_lo, 3);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s3), y_filter_hi, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s4), y_filter_hi, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s5), y_filter_hi, 2);

  sum1 = vmull_lane_s16(vget_high_s16(s0), y_filter_lo, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s1), y_filter_lo, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s2), y_filter_lo, 3);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s3), y_filter_hi, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s4), y_filter_hi, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s5), y_filter_hi, 2);

  sum0 = vaddq_s32(sum0, offset_const);
  sum1 = vaddq_s32(sum1, offset_const);
  sum0 = vqrshlq_s32(sum0, round_shift_vec);
  sum1 = vqrshlq_s32(sum1, round_shift_vec);
  sum0 = vsubq_s32(sum0, sub_const_vec);
  sum1 = vsubq_s32(sum1, sub_const_vec);

  res = vcombine_s16(vmovn_s32(sum0), vmovn_s32(sum1));
  res = vqrshlq_s16(res, vec_round_bits);

  return vqmovun_s16(res);
}

static INLINE void av1_convolve_2d_sr_vert_6tap_neon(
    int16_t *src_ptr, int src_stride, uint8_t *dst_ptr, int dst_stride, int w,
    int h, const int16x8_t y_filter, ConvolveParams *conv_params) {
  const int bd = 8;
  const int16_t round_bits =
      FILTER_BITS * 2 - conv_params->round_0 - conv_params->round_1;
  const int16x8_t vec_round_bits = vdupq_n_s16(-round_bits);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;

  const int32_t sub_const = (1 << (offset_bits - conv_params->round_1)) +
                            (1 << (offset_bits - conv_params->round_1 - 1));

  const int32x4_t round_shift_vec = vdupq_n_s32(-(conv_params->round_1));
  const int32x4_t offset_const = vdupq_n_s32(1 << offset_bits);
  const int32x4_t sub_const_vec = vdupq_n_s32(sub_const);

  if (w <= 4) {
    int16x4_t s0, s1, s2, s3, s4, s5, d0;
    int16x8_t dd0;
    uint8x8_t d01;

#if defined(__aarch64__)
    int16x4_t s6, s7, s8, d1, d2, d3;
    int16x8_t dd1;
    uint8x8_t d23;
#endif  // defined(__aarch64__)

    int16_t *s = src_ptr;
    uint8_t *d = dst_ptr;

    s0 = vld1_s16(s + 0 * src_stride);
    s1 = vld1_s16(s + 1 * src_stride);
    s2 = vld1_s16(s + 2 * src_stride);
    s3 = vld1_s16(s + 3 * src_stride);
    s4 = vld1_s16(s + 4 * src_stride);
    s += (5 * src_stride);

    do {
#if defined(__aarch64__)
      load_s16_4x4(s, src_stride, &s5, &s6, &s7, &s8);
      s += (4 * src_stride);

      d0 = convolve6_vert_4x4_s32(s0, s1, s2, s3, s4, s5, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);
      d1 = convolve6_vert_4x4_s32(s1, s2, s3, s4, s5, s6, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);
      d2 = convolve6_vert_4x4_s32(s2, s3, s4, s5, s6, s7, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);
      d3 = convolve6_vert_4x4_s32(s3, s4, s5, s6, s7, s8, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);

      dd0 = vqrshlq_s16(vcombine_s16(d0, d1), vec_round_bits);
      dd1 = vqrshlq_s16(vcombine_s16(d2, d3), vec_round_bits);

      d01 = vqmovun_s16(dd0);
      d23 = vqmovun_s16(dd1);

      if (w == 4) {
        vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d01), 0);
        d += dst_stride;
        vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d01), 1);
        d += dst_stride;
        if (h != 2) {
          vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d23), 0);
          d += dst_stride;
          vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d23), 1);
          d += dst_stride;
        }
      } else {
        vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d01), 0);
        d += dst_stride;
        vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d01), 2);
        d += dst_stride;
        if (h != 2) {
          vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d23), 0);
          d += dst_stride;
          vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d23), 2);
          d += dst_stride;
        }
      }

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      h -= 4;
#else   // !defined(__aarch64__)
      s5 = vld1_s16(s);
      s += src_stride;

      d0 = convolve6_vert_4x4_s32(s0, s1, s2, s3, s4, s5, y_filter,
                                  round_shift_vec, offset_const, sub_const_vec);

      dd0 = vqrshlq_s16(vcombine_s16(d0, d0), vec_round_bits);
      d01 = vqmovun_s16(dd0);

      if (w == 2) {
        vst1_lane_u16((uint16_t *)d, vreinterpret_u16_u8(d01), 0);
        d += dst_stride;
      } else {
        vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(d01), 0);
        d += dst_stride;
      }

      s0 = s1;
      s1 = s2;
      s2 = s3;
      s3 = s4;
      s4 = s5;
      h--;
#endif  // defined(__aarch64__)
    } while (h > 0);
  } else {
    // if width is a multiple of 8 & height is a multiple of 4
    int16x8_t s0, s1, s2, s3, s4, s5;
    uint8x8_t d0;
#if defined(__aarch64__)
    int16x8_t s6, s7, s8;
    uint8x8_t d1, d2, d3;
#endif  // defined(__aarch64__)

    do {
      int height = h;
      int16_t *s = src_ptr;
      uint8_t *d = dst_ptr;

      s0 = vld1q_s16(s + 0 * src_stride);
      s1 = vld1q_s16(s + 1 * src_stride);
      s2 = vld1q_s16(s + 2 * src_stride);
      s3 = vld1q_s16(s + 3 * src_stride);
      s4 = vld1q_s16(s + 4 * src_stride);
      s += (5 * src_stride);

      do {
#if defined(__aarch64__)
        load_s16_8x4(s, src_stride, &s5, &s6, &s7, &s8);
        s += (4 * src_stride);

        d0 = convolve6_vert_8x4_s32(s0, s1, s2, s3, s4, s5, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);
        d1 = convolve6_vert_8x4_s32(s1, s2, s3, s4, s5, s6, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);
        d2 = convolve6_vert_8x4_s32(s2, s3, s4, s5, s6, s7, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);
        d3 = convolve6_vert_8x4_s32(s3, s4, s5, s6, s7, s8, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);

        vst1_u8(d, d0);
        d += dst_stride;
        vst1_u8(d, d1);
        d += dst_stride;
        if (h != 2) {
          vst1_u8(d, d2);
          d += dst_stride;
          vst1_u8(d, d3);
          d += dst_stride;
        }

        s0 = s4;
        s1 = s5;
        s2 = s6;
        s3 = s7;
        s4 = s8;
        height -= 4;
#else   // !defined(__aarch64__)
        s5 = vld1q_s16(s);
        s += src_stride;

        d0 = convolve6_vert_8x4_s32(s0, s1, s2, s3, s4, s5, y_filter,
                                    round_shift_vec, offset_const,
                                    sub_const_vec, vec_round_bits);

        vst1_u8(d, d0);
        d += dst_stride;

        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s5;
        height--;
#endif  // defined(__aarch64__)
      } while (height > 0);

      src_ptr += 8;
      dst_ptr += 8;
      w -= 8;
    } while (w > 0);
  }
}

void av1_convolve_2d_sr_neon(const uint8_t *src, int src_stride, uint8_t *dst,
                             int dst_stride, int w, int h,
                             const InterpFilterParams *filter_params_x,
                             const InterpFilterParams *filter_params_y,
                             const int subpel_x_qn, const int subpel_y_qn,
                             ConvolveParams *conv_params) {
  const int y_filter_taps = get_filter_tap(filter_params_y, subpel_y_qn);
  const int clamped_y_taps = y_filter_taps < 6 ? 6 : y_filter_taps;
  const int im_h = h + clamped_y_taps - 1;
  const int im_stride = MAX_SB_SIZE;
  const int vert_offset = clamped_y_taps / 2 - 1;
  const int horiz_offset = filter_params_x->taps / 2 - 1;
  const uint8_t *src_ptr = src - vert_offset * src_stride - horiz_offset;

  const int16_t *x_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  const int16_t *y_filter_ptr = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);

  if (filter_params_x->taps > 8) {
    DECLARE_ALIGNED(16, int16_t,
                    im_block[(MAX_SB_SIZE + MAX_FILTER_TAP - 1) * MAX_SB_SIZE]);

    const int16x8_t x_filter_0_7 = vld1q_s16(x_filter_ptr);
    const int16x4_t x_filter_8_11 = vld1_s16(x_filter_ptr + 8);
    const int16x8_t y_filter_0_7 = vld1q_s16(y_filter_ptr);
    const int16x4_t y_filter_8_11 = vld1_s16(y_filter_ptr + 8);

    av1_convolve_2d_sr_horiz_12tap_neon(src_ptr, src_stride, im_block,
                                        im_stride, w, im_h, x_filter_0_7,
                                        x_filter_8_11, conv_params->round_0);

    av1_convolve_2d_sr_vert_12tap_neon(im_block, im_stride, dst, dst_stride, w,
                                       h, y_filter_0_7, y_filter_8_11,
                                       conv_params);
  } else {
    DECLARE_ALIGNED(16, int16_t,
                    im_block[(MAX_SB_SIZE + HORIZ_EXTRA_ROWS) * MAX_SB_SIZE]);

    const int16x8_t x_filter = vld1q_s16(x_filter_ptr);
    const int16x8_t y_filter = vld1q_s16(y_filter_ptr);

    av1_convolve_2d_sr_horiz_neon(src_ptr, src_stride, im_block, im_stride, w,
                                  im_h, x_filter, conv_params->round_0);

    if (clamped_y_taps <= 6) {
      av1_convolve_2d_sr_vert_6tap_neon(im_block, im_stride, dst, dst_stride, w,
                                        h, y_filter, conv_params);
    } else {
      av1_convolve_2d_sr_vert_8tap_neon(im_block, im_stride, dst, dst_stride, w,
                                        h, y_filter, conv_params);
    }
  }
}

static INLINE void scaledconvolve_horiz_w4(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *dst,
    const ptrdiff_t dst_stride, const InterpKernel *const x_filters,
    const int x0_q4, const int x_step_q4, const int w, const int h) {
  DECLARE_ALIGNED(16, uint8_t, temp[4 * 4]);
  int x, y, z;

  src -= SUBPEL_TAPS / 2 - 1;

  y = h;
  do {
    int x_q4 = x0_q4;
    x = 0;
    do {
      // process 4 src_x steps
      for (z = 0; z < 4; ++z) {
        const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
        if (x_q4 & SUBPEL_MASK) {
          const int16x8_t filters = vld1q_s16(x_filters[x_q4 & SUBPEL_MASK]);
          uint8x8_t s[8], d;
          int16x8_t ss[4];
          int16x4_t t[8], tt;

          load_u8_8x4(src_x, src_stride, &s[0], &s[1], &s[2], &s[3]);
          transpose_u8_8x4(&s[0], &s[1], &s[2], &s[3]);

          ss[0] = vreinterpretq_s16_u16(vmovl_u8(s[0]));
          ss[1] = vreinterpretq_s16_u16(vmovl_u8(s[1]));
          ss[2] = vreinterpretq_s16_u16(vmovl_u8(s[2]));
          ss[3] = vreinterpretq_s16_u16(vmovl_u8(s[3]));
          t[0] = vget_low_s16(ss[0]);
          t[1] = vget_low_s16(ss[1]);
          t[2] = vget_low_s16(ss[2]);
          t[3] = vget_low_s16(ss[3]);
          t[4] = vget_high_s16(ss[0]);
          t[5] = vget_high_s16(ss[1]);
          t[6] = vget_high_s16(ss[2]);
          t[7] = vget_high_s16(ss[3]);

          tt = convolve8_4(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7],
                           filters);
          d = vqrshrun_n_s16(vcombine_s16(tt, tt), 7);
          vst1_lane_u32((uint32_t *)&temp[4 * z], vreinterpret_u32_u8(d), 0);
        } else {
          int i;
          for (i = 0; i < 4; ++i) {
            temp[z * 4 + i] = src_x[i * src_stride + 3];
          }
        }
        x_q4 += x_step_q4;
      }

      // transpose the 4x4 filters values back to dst
      {
        const uint8x8x4_t d4 = vld4_u8(temp);
        vst1_lane_u32((uint32_t *)&dst[x + 0 * dst_stride],
                      vreinterpret_u32_u8(d4.val[0]), 0);
        vst1_lane_u32((uint32_t *)&dst[x + 1 * dst_stride],
                      vreinterpret_u32_u8(d4.val[1]), 0);
        vst1_lane_u32((uint32_t *)&dst[x + 2 * dst_stride],
                      vreinterpret_u32_u8(d4.val[2]), 0);
        vst1_lane_u32((uint32_t *)&dst[x + 3 * dst_stride],
                      vreinterpret_u32_u8(d4.val[3]), 0);
      }
      x += 4;
    } while (x < w);

    src += src_stride * 4;
    dst += dst_stride * 4;
    y -= 4;
  } while (y > 0);
}

static INLINE void scaledconvolve_horiz_w8(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *dst,
    const ptrdiff_t dst_stride, const InterpKernel *const x_filters,
    const int x0_q4, const int x_step_q4, const int w, const int h) {
  DECLARE_ALIGNED(16, uint8_t, temp[8 * 8]);
  int x, y, z;
  src -= SUBPEL_TAPS / 2 - 1;

  // This function processes 8x8 areas. The intermediate height is not always
  // a multiple of 8, so force it to be a multiple of 8 here.
  y = (h + 7) & ~7;

  do {
    int x_q4 = x0_q4;
    x = 0;
    do {
      uint8x8_t d[8];
      // process 8 src_x steps
      for (z = 0; z < 8; ++z) {
        const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];

        if (x_q4 & SUBPEL_MASK) {
          const int16x8_t filters = vld1q_s16(x_filters[x_q4 & SUBPEL_MASK]);
          uint8x8_t s[8];
          load_u8_8x8(src_x, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4],
                      &s[5], &s[6], &s[7]);
          transpose_u8_8x8(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6],
                           &s[7]);
          d[0] = scale_filter_8(s, filters);
          vst1_u8(&temp[8 * z], d[0]);
        } else {
          int i;
          for (i = 0; i < 8; ++i) {
            temp[z * 8 + i] = src_x[i * src_stride + 3];
          }
        }
        x_q4 += x_step_q4;
      }

      // transpose the 8x8 filters values back to dst
      load_u8_8x8(temp, 8, &d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6],
                  &d[7]);
      transpose_u8_8x8(&d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6], &d[7]);
      vst1_u8(&dst[x + 0 * dst_stride], d[0]);
      vst1_u8(&dst[x + 1 * dst_stride], d[1]);
      vst1_u8(&dst[x + 2 * dst_stride], d[2]);
      vst1_u8(&dst[x + 3 * dst_stride], d[3]);
      vst1_u8(&dst[x + 4 * dst_stride], d[4]);
      vst1_u8(&dst[x + 5 * dst_stride], d[5]);
      vst1_u8(&dst[x + 6 * dst_stride], d[6]);
      vst1_u8(&dst[x + 7 * dst_stride], d[7]);
      x += 8;
    } while (x < w);

    src += src_stride * 8;
    dst += dst_stride * 8;
  } while (y -= 8);
}

static INLINE void scaledconvolve_vert_w4(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *dst,
    const ptrdiff_t dst_stride, const InterpKernel *const y_filters,
    const int y0_q4, const int y_step_q4, const int w, const int h) {
  int y;
  int y_q4 = y0_q4;

  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  y = h;
  do {
    const unsigned char *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];

    if (y_q4 & SUBPEL_MASK) {
      const int16x8_t filters = vld1q_s16(y_filters[y_q4 & SUBPEL_MASK]);
      uint8x8_t s[8], d;
      int16x4_t t[8], tt;

      load_u8_8x8(src_y, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5],
                  &s[6], &s[7]);
      t[0] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(s[0])));
      t[1] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(s[1])));
      t[2] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(s[2])));
      t[3] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(s[3])));
      t[4] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(s[4])));
      t[5] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(s[5])));
      t[6] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(s[6])));
      t[7] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(s[7])));

      tt = convolve8_4(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], filters);
      d = vqrshrun_n_s16(vcombine_s16(tt, tt), 7);
      vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d), 0);
    } else {
      memcpy(dst, &src_y[3 * src_stride], w);
    }

    dst += dst_stride;
    y_q4 += y_step_q4;
  } while (--y);
}

static INLINE void scaledconvolve_vert_w8(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *dst,
    const ptrdiff_t dst_stride, const InterpKernel *const y_filters,
    const int y0_q4, const int y_step_q4, const int w, const int h) {
  int y;
  int y_q4 = y0_q4;

  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  y = h;
  do {
    const unsigned char *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    if (y_q4 & SUBPEL_MASK) {
      const int16x8_t filters = vld1q_s16(y_filters[y_q4 & SUBPEL_MASK]);
      uint8x8_t s[8], d;
      load_u8_8x8(src_y, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5],
                  &s[6], &s[7]);
      d = scale_filter_8(s, filters);
      vst1_u8(dst, d);
    } else {
      memcpy(dst, &src_y[3 * src_stride], w);
    }
    dst += dst_stride;
    y_q4 += y_step_q4;
  } while (--y);
}

static INLINE void scaledconvolve_vert_w16(
    const uint8_t *src, const ptrdiff_t src_stride, uint8_t *dst,
    const ptrdiff_t dst_stride, const InterpKernel *const y_filters,
    const int y0_q4, const int y_step_q4, const int w, const int h) {
  int x, y;
  int y_q4 = y0_q4;

  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  y = h;
  do {
    const unsigned char *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    if (y_q4 & SUBPEL_MASK) {
      x = 0;
      do {
        const int16x8_t filters = vld1q_s16(y_filters[y_q4 & SUBPEL_MASK]);
        uint8x16_t ss[8];
        uint8x8_t s[8], d[2];
        load_u8_16x8(src_y, src_stride, &ss[0], &ss[1], &ss[2], &ss[3], &ss[4],
                     &ss[5], &ss[6], &ss[7]);
        s[0] = vget_low_u8(ss[0]);
        s[1] = vget_low_u8(ss[1]);
        s[2] = vget_low_u8(ss[2]);
        s[3] = vget_low_u8(ss[3]);
        s[4] = vget_low_u8(ss[4]);
        s[5] = vget_low_u8(ss[5]);
        s[6] = vget_low_u8(ss[6]);
        s[7] = vget_low_u8(ss[7]);
        d[0] = scale_filter_8(s, filters);

        s[0] = vget_high_u8(ss[0]);
        s[1] = vget_high_u8(ss[1]);
        s[2] = vget_high_u8(ss[2]);
        s[3] = vget_high_u8(ss[3]);
        s[4] = vget_high_u8(ss[4]);
        s[5] = vget_high_u8(ss[5]);
        s[6] = vget_high_u8(ss[6]);
        s[7] = vget_high_u8(ss[7]);
        d[1] = scale_filter_8(s, filters);
        vst1q_u8(&dst[x], vcombine_u8(d[0], d[1]));
        src_y += 16;
        x += 16;
      } while (x < w);
    } else {
      memcpy(dst, &src_y[3 * src_stride], w);
    }
    dst += dst_stride;
    y_q4 += y_step_q4;
  } while (--y);
}

void aom_scaled_2d_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                        ptrdiff_t dst_stride, const InterpKernel *filter,
                        int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                        int w, int h) {
  // Note: Fixed size intermediate buffer, temp, places limits on parameters.
  // 2d filtering proceeds in 2 steps:
  //   (1) Interpolate horizontally into an intermediate buffer, temp.
  //   (2) Interpolate temp vertically to derive the sub-pixel result.
  // Deriving the maximum number of rows in the temp buffer (135):
  // --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
  // --Largest block size is 64x64 pixels.
  // --64 rows in the downscaled frame span a distance of (64 - 1) * 32 in the
  //   original frame (in 1/16th pixel units).
  // --Must round-up because block may be located at sub-pixel position.
  // --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
  // --((64 - 1) * 32 + 15) >> 4 + 8 = 135.
  // --Require an additional 8 rows for the horiz_w8 transpose tail.
  // When calling in frame scaling function, the smallest scaling factor is x1/4
  // ==> y_step_q4 = 64. Since w and h are at most 16, the temp buffer is still
  // big enough.
  DECLARE_ALIGNED(16, uint8_t, temp[(135 + 8) * 64]);
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

  assert(w <= 64);
  assert(h <= 64);
  assert(y_step_q4 <= 32 || (y_step_q4 <= 64 && h <= 32));
  assert(x_step_q4 <= 64);

  if (w >= 8) {
    scaledconvolve_horiz_w8(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                            src_stride, temp, 64, filter, x0_q4, x_step_q4, w,
                            intermediate_height);
  } else {
    scaledconvolve_horiz_w4(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                            src_stride, temp, 64, filter, x0_q4, x_step_q4, w,
                            intermediate_height);
  }

  if (w >= 16) {
    scaledconvolve_vert_w16(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                            dst_stride, filter, y0_q4, y_step_q4, w, h);
  } else if (w == 8) {
    scaledconvolve_vert_w8(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                           dst_stride, filter, y0_q4, y_step_q4, w, h);
  } else {
    scaledconvolve_vert_w4(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                           dst_stride, filter, y0_q4, y_step_q4, w, h);
  }
}
