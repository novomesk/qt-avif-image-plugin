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

#include "aom_dsp/aom_simd.h"
#include "aom_dsp/arm/mem_neon.h"

#define SIMD_FUNC(name) name##_neon
#include "av1/common/cdef_block_simd.h"

void cdef_copy_rect8_8bit_to_16bit_neon(uint16_t *dst, int dstride,
                                        const uint8_t *src, int sstride,
                                        int width, int height) {
  int j;
  for (int i = 0; i < height; i++) {
    for (j = 0; j < (width & ~0x7); j += 8) {
      v64 row = v64_load_unaligned(&src[i * sstride + j]);
      v128_store_unaligned(&dst[i * dstride + j], v128_unpack_u8_s16(row));
    }
    for (; j < width; j++) {
      dst[i * dstride + j] = src[i * sstride + j];
    }
  }
}

static INLINE int16x8_t v128_from_64_neon(int64_t a, int64_t b) {
  return vreinterpretq_s16_s64(vcombine_s64(vcreate_s64(a), vcreate_s64(b)));
}

#define SHL_HIGH_NEON(n)                                                       \
  static INLINE int16x8_t v128_shl_##n##_byte_neon(int16x8_t a) {              \
    int64x2_t a_s64 = vreinterpretq_s64_s16(a);                                \
    return v128_from_64_neon(                                                  \
        0, vget_lane_u64(vshl_n_u64(vreinterpret_u64_s64(vget_low_s64(a_s64)), \
                                    (n - 8) * 8),                              \
                         0));                                                  \
  }

#define SHL_NEON(n)                                                      \
  static INLINE int16x8_t v128_shl_##n##_byte_neon(int16x8_t a) {        \
    int64x2_t a_s64 = vreinterpretq_s64_s16(a);                          \
    return v128_from_64_neon(                                            \
        0, vget_lane_u64(vreinterpret_u64_s64(vget_low_s64(a_s64)), 0)); \
  }

#define SHL_LOW_NEON(n)                                                        \
  static INLINE int16x8_t v128_shl_##n##_byte_neon(int16x8_t a) {              \
    int64x2_t a_s64 = vreinterpretq_s64_s16(a);                                \
    return v128_from_64_neon(                                                  \
        vget_lane_u64(                                                         \
            vshl_n_u64(vreinterpret_u64_s64(vget_low_s64(a_s64)), n * 8), 0),  \
        vget_lane_u64(                                                         \
            vorr_u64(                                                          \
                vshl_n_u64(vreinterpret_u64_s64(vget_high_s64(a_s64)), n * 8), \
                vshr_n_u64(vreinterpret_u64_s64(vget_low_s64(a_s64)),          \
                           (8 - n) * 8)),                                      \
            0));                                                               \
  }

SHL_HIGH_NEON(14)
SHL_HIGH_NEON(12)
SHL_HIGH_NEON(10)
SHL_NEON(8)
SHL_LOW_NEON(6)
SHL_LOW_NEON(4)
SHL_LOW_NEON(2)

#define v128_shl_n_byte_neon(a, n) v128_shl_##n##_byte_neon(a)

#define SHR_HIGH_NEON(n)                                                     \
  static INLINE int16x8_t v128_shr_##n##_byte_neon(int16x8_t a) {            \
    int64x2_t a_s64 = vreinterpretq_s64_s16(a);                              \
    return v128_from_64_neon(                                                \
        vget_lane_u64(vshr_n_u64(vreinterpret_u64_s64(vget_high_s64(a_s64)), \
                                 (n - 8) * 8),                               \
                      0),                                                    \
        0);                                                                  \
  }

#define SHR_NEON(n)                                                       \
  static INLINE int16x8_t v128_shr_##n##_byte_neon(int16x8_t a) {         \
    int64x2_t a_s64 = vreinterpretq_s64_s16(a);                           \
    return v128_from_64_neon(                                             \
        vget_lane_u64(vreinterpret_u64_s64(vget_high_s64(a_s64)), 0), 0); \
  }

#define SHR_LOW_NEON(n)                                                       \
  static INLINE int16x8_t v128_shr_##n##_byte_neon(int16x8_t a) {             \
    int64x2_t a_s64 = vreinterpretq_s64_s16(a);                               \
    return v128_from_64_neon(                                                 \
        vget_lane_u64(                                                        \
            vorr_u64(                                                         \
                vshr_n_u64(vreinterpret_u64_s64(vget_low_s64(a_s64)), n * 8), \
                vshl_n_u64(vreinterpret_u64_s64(vget_high_s64(a_s64)),        \
                           (8 - n) * 8)),                                     \
            0),                                                               \
        vget_lane_u64(                                                        \
            vshr_n_u64(vreinterpret_u64_s64(vget_high_s64(a_s64)), n * 8),    \
            0));                                                              \
  }

SHR_HIGH_NEON(14)
SHR_HIGH_NEON(12)
SHR_HIGH_NEON(10)
SHR_NEON(8)
SHR_LOW_NEON(6)
SHR_LOW_NEON(4)
SHR_LOW_NEON(2)

#define v128_shr_n_byte_neon(a, n) v128_shr_##n##_byte_neon(a)

static INLINE uint32x4_t v128_madd_s16_neon(int16x8_t a, int16x8_t b) {
  uint32x4_t t1 =
      vreinterpretq_u32_s32(vmull_s16(vget_low_s16(a), vget_low_s16(b)));
  uint32x4_t t2 =
      vreinterpretq_u32_s32(vmull_s16(vget_high_s16(a), vget_high_s16(b)));
#if AOM_ARCH_AARCH64
  return vpaddq_u32(t1, t2);
#else
  return vcombine_u32(vpadd_u32(vget_low_u32(t1), vget_high_u32(t1)),
                      vpadd_u32(vget_low_u32(t2), vget_high_u32(t2)));
#endif
}

// partial A is a 16-bit vector of the form:
// [x8 x7 x6 x5 x4 x3 x2 x1] and partial B has the form:
// [0  y1 y2 y3 y4 y5 y6 y7].
// This function computes (x1^2+y1^2)*C1 + (x2^2+y2^2)*C2 + ...
// (x7^2+y2^7)*C7 + (x8^2+0^2)*C8 where the C1..C8 constants are in const1
// and const2.
static INLINE uint32x4_t fold_mul_and_sum_neon(int16x8_t partiala,
                                               int16x8_t partialb,
                                               uint32x4_t const1,
                                               uint32x4_t const2) {
  int16x8_t tmp;
  // Reverse partial B.
  uint8x16_t pattern = vreinterpretq_u8_u64(
      vcombine_u64(vcreate_u64((uint64_t)0x07060908 << 32 | 0x0b0a0d0c),
                   vcreate_u64((uint64_t)0x0f0e0100 << 32 | 0x03020504)));

#if AOM_ARCH_AARCH64
  partialb =
      vreinterpretq_s16_s8(vqtbl1q_s8(vreinterpretq_s8_s16(partialb), pattern));
#else
  int8x8x2_t p = { { vget_low_s8(vreinterpretq_s8_s16(partialb)),
                     vget_high_s8(vreinterpretq_s8_s16(partialb)) } };
  int8x8_t shuffle_hi = vtbl2_s8(p, vget_high_s8(vreinterpretq_s8_u8(pattern)));
  int8x8_t shuffle_lo = vtbl2_s8(p, vget_low_s8(vreinterpretq_s8_u8(pattern)));
  partialb = vreinterpretq_s16_s8(vcombine_s8(shuffle_lo, shuffle_hi));
#endif

  // Interleave the x and y values of identical indices and pair x8 with 0.
  tmp = partiala;
  partiala = vzipq_s16(partiala, partialb).val[0];
  partialb = vzipq_s16(tmp, partialb).val[1];
  // Square and add the corresponding x and y values.
  uint32x4_t partiala_u32 = v128_madd_s16_neon(partiala, partiala);
  uint32x4_t partialb_u32 = v128_madd_s16_neon(partialb, partialb);

  // Multiply by constant.
  partiala_u32 = vmulq_u32(partiala_u32, const1);
  partialb_u32 = vmulq_u32(partialb_u32, const2);

  // Sum all results.
  partiala_u32 = vaddq_u32(partiala_u32, partialb_u32);
  return partiala_u32;
}

static INLINE uint64x2_t ziplo_u64(uint32x4_t a, uint32x4_t b) {
  return vcombine_u64(vget_low_u64(vreinterpretq_u64_u32(a)),
                      vget_low_u64(vreinterpretq_u64_u32(b)));
}

static INLINE uint64x2_t ziphi_u64(uint32x4_t a, uint32x4_t b) {
  return vcombine_u64(vget_high_u64(vreinterpretq_u64_u32(a)),
                      vget_high_u64(vreinterpretq_u64_u32(b)));
}

static INLINE uint32x4_t hsum4_neon(uint32x4_t x0, uint32x4_t x1, uint32x4_t x2,
                                    uint32x4_t x3) {
  uint32x4_t t0, t1, t2, t3;
  t0 = vzipq_u32(x0, x1).val[0];
  t1 = vzipq_u32(x2, x3).val[0];
  t2 = vzipq_u32(x0, x1).val[1];
  t3 = vzipq_u32(x2, x3).val[1];
  x0 = vreinterpretq_u32_u64(ziplo_u64(t0, t1));
  x1 = vreinterpretq_u32_u64(ziphi_u64(t0, t1));
  x2 = vreinterpretq_u32_u64(ziplo_u64(t2, t3));
  x3 = vreinterpretq_u32_u64(ziphi_u64(t2, t3));
  return vaddq_u32(vaddq_u32(x0, x1), vaddq_u32(x2, x3));
}

static INLINE uint32x4_t compute_directions_neon(int16x8_t lines[8],
                                                 uint32_t cost[4]) {
  int16x8_t partial4a, partial4b, partial5a, partial5b, partial6, partial7a,
      partial7b;
  int16x8_t tmp;

  // Partial sums for lines 0 and 1.
  partial4a = v128_shl_n_byte_neon(lines[0], 14);
  partial4b = v128_shr_n_byte_neon(lines[0], 2);
  partial4a = vaddq_s16(partial4a, v128_shl_n_byte_neon(lines[1], 12));
  partial4b = vaddq_s16(partial4b, v128_shr_n_byte_neon(lines[1], 4));
  tmp = vaddq_s16(lines[0], lines[1]);
  partial5a = v128_shl_n_byte_neon(tmp, 10);
  partial5b = v128_shr_n_byte_neon(tmp, 6);
  partial7a = v128_shl_n_byte_neon(tmp, 4);
  partial7b = v128_shr_n_byte_neon(tmp, 12);
  partial6 = tmp;

  // Partial sums for lines 2 and 3.
  partial4a = vaddq_s16(partial4a, v128_shl_n_byte_neon(lines[2], 10));
  partial4b = vaddq_s16(partial4b, v128_shr_n_byte_neon(lines[2], 6));
  partial4a = vaddq_s16(partial4a, v128_shl_n_byte_neon(lines[3], 8));
  partial4b = vaddq_s16(partial4b, v128_shr_n_byte_neon(lines[3], 8));
  tmp = vaddq_s16(lines[2], lines[3]);
  partial5a = vaddq_s16(partial5a, v128_shl_n_byte_neon(tmp, 8));
  partial5b = vaddq_s16(partial5b, v128_shr_n_byte_neon(tmp, 8));
  partial7a = vaddq_s16(partial7a, v128_shl_n_byte_neon(tmp, 6));
  partial7b = vaddq_s16(partial7b, v128_shr_n_byte_neon(tmp, 10));
  partial6 = vaddq_s16(partial6, tmp);

  // Partial sums for lines 4 and 5.
  partial4a = vaddq_s16(partial4a, v128_shl_n_byte_neon(lines[4], 6));
  partial4b = vaddq_s16(partial4b, v128_shr_n_byte_neon(lines[4], 10));
  partial4a = vaddq_s16(partial4a, v128_shl_n_byte_neon(lines[5], 4));
  partial4b = vaddq_s16(partial4b, v128_shr_n_byte_neon(lines[5], 12));
  tmp = vaddq_s16(lines[4], lines[5]);
  partial5a = vaddq_s16(partial5a, v128_shl_n_byte_neon(tmp, 6));
  partial5b = vaddq_s16(partial5b, v128_shr_n_byte_neon(tmp, 10));
  partial7a = vaddq_s16(partial7a, v128_shl_n_byte_neon(tmp, 8));
  partial7b = vaddq_s16(partial7b, v128_shr_n_byte_neon(tmp, 8));
  partial6 = vaddq_s16(partial6, tmp);

  // Partial sums for lines 6 and 7.
  partial4a = vaddq_s16(partial4a, v128_shl_n_byte_neon(lines[6], 2));
  partial4b = vaddq_s16(partial4b, v128_shr_n_byte_neon(lines[6], 14));
  partial4a = vaddq_s16(partial4a, lines[7]);
  tmp = vaddq_s16(lines[6], lines[7]);
  partial5a = vaddq_s16(partial5a, v128_shl_n_byte_neon(tmp, 4));
  partial5b = vaddq_s16(partial5b, v128_shr_n_byte_neon(tmp, 12));
  partial7a = vaddq_s16(partial7a, v128_shl_n_byte_neon(tmp, 10));
  partial7b = vaddq_s16(partial7b, v128_shr_n_byte_neon(tmp, 6));
  partial6 = vaddq_s16(partial6, tmp);

  uint32x4_t const0 = vreinterpretq_u32_u64(
      vcombine_u64(vcreate_u64((uint64_t)420 << 32 | 840),
                   vcreate_u64((uint64_t)210 << 32 | 280)));
  uint32x4_t const1 = vreinterpretq_u32_u64(
      vcombine_u64(vcreate_u64((uint64_t)140 << 32 | 168),
                   vcreate_u64((uint64_t)105 << 32 | 120)));
  uint32x4_t const2 = vreinterpretq_u32_u64(
      vcombine_u64(vcreate_u64(0), vcreate_u64((uint64_t)210 << 32 | 420)));
  uint32x4_t const3 = vreinterpretq_u32_u64(
      vcombine_u64(vcreate_u64((uint64_t)105 << 32 | 140),
                   vcreate_u64((uint64_t)105 << 32 | 105)));

  // Compute costs in terms of partial sums.
  uint32x4_t partial4a_u32 =
      fold_mul_and_sum_neon(partial4a, partial4b, const0, const1);
  uint32x4_t partial7a_u32 =
      fold_mul_and_sum_neon(partial7a, partial7b, const2, const3);
  uint32x4_t partial5a_u32 =
      fold_mul_and_sum_neon(partial5a, partial5b, const2, const3);
  uint32x4_t partial6_u32 = v128_madd_s16_neon(partial6, partial6);
  partial6_u32 = vmulq_u32(partial6_u32, vdupq_n_u32(105));

  partial4a_u32 =
      hsum4_neon(partial4a_u32, partial5a_u32, partial6_u32, partial7a_u32);
  vst1q_u32(cost, partial4a_u32);
  return partial4a_u32;
}

static INLINE int64x2_t ziplo_s64(int32x4_t a, int32x4_t b) {
  return vcombine_s64(vget_low_s64(vreinterpretq_s64_s32(a)),
                      vget_low_s64(vreinterpretq_s64_s32(b)));
}

static INLINE int64x2_t ziphi_s64(int32x4_t a, int32x4_t b) {
  return vcombine_s64(vget_high_s64(vreinterpretq_s64_s32(a)),
                      vget_high_s64(vreinterpretq_s64_s32(b)));
}

// Transpose and reverse the order of the lines -- equivalent to a 90-degree
// counter-clockwise rotation of the pixels.
static INLINE void array_reverse_transpose_8x8_neon(int16x8_t *in,
                                                    int16x8_t *res) {
  const int32x4_t tr0_0 = vreinterpretq_s32_s16(vzipq_s16(in[0], in[1]).val[0]);
  const int32x4_t tr0_1 = vreinterpretq_s32_s16(vzipq_s16(in[2], in[3]).val[0]);
  const int32x4_t tr0_2 = vreinterpretq_s32_s16(vzipq_s16(in[0], in[1]).val[1]);
  const int32x4_t tr0_3 = vreinterpretq_s32_s16(vzipq_s16(in[2], in[3]).val[1]);
  const int32x4_t tr0_4 = vreinterpretq_s32_s16(vzipq_s16(in[4], in[5]).val[0]);
  const int32x4_t tr0_5 = vreinterpretq_s32_s16(vzipq_s16(in[6], in[7]).val[0]);
  const int32x4_t tr0_6 = vreinterpretq_s32_s16(vzipq_s16(in[4], in[5]).val[1]);
  const int32x4_t tr0_7 = vreinterpretq_s32_s16(vzipq_s16(in[6], in[7]).val[1]);

  const int32x4_t tr1_0 = vzipq_s32(tr0_0, tr0_1).val[0];
  const int32x4_t tr1_1 = vzipq_s32(tr0_4, tr0_5).val[0];
  const int32x4_t tr1_2 = vzipq_s32(tr0_0, tr0_1).val[1];
  const int32x4_t tr1_3 = vzipq_s32(tr0_4, tr0_5).val[1];
  const int32x4_t tr1_4 = vzipq_s32(tr0_2, tr0_3).val[0];
  const int32x4_t tr1_5 = vzipq_s32(tr0_6, tr0_7).val[0];
  const int32x4_t tr1_6 = vzipq_s32(tr0_2, tr0_3).val[1];
  const int32x4_t tr1_7 = vzipq_s32(tr0_6, tr0_7).val[1];

  res[7] = vreinterpretq_s16_s64(ziplo_s64(tr1_0, tr1_1));
  res[6] = vreinterpretq_s16_s64(ziphi_s64(tr1_0, tr1_1));
  res[5] = vreinterpretq_s16_s64(ziplo_s64(tr1_2, tr1_3));
  res[4] = vreinterpretq_s16_s64(ziphi_s64(tr1_2, tr1_3));
  res[3] = vreinterpretq_s16_s64(ziplo_s64(tr1_4, tr1_5));
  res[2] = vreinterpretq_s16_s64(ziphi_s64(tr1_4, tr1_5));
  res[1] = vreinterpretq_s16_s64(ziplo_s64(tr1_6, tr1_7));
  res[0] = vreinterpretq_s16_s64(ziphi_s64(tr1_6, tr1_7));
}

static INLINE uint32_t compute_best_dir(uint8x16_t a) {
  uint8x16_t idx =
      vandq_u8(a, vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201ULL)));
#if AOM_ARCH_AARCH64
  return vaddv_u8(vget_low_u8(idx)) + (vaddv_u8(vget_high_u8(idx)) << 8);
#else
  uint64x2_t m = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(idx)));
  uint8x16_t s = vreinterpretq_u8_u64(m);
  return vget_lane_u32(
      vreinterpret_u32_u8(vzip_u8(vget_low_u8(s), vget_high_u8(s)).val[0]), 0);
#endif
}

int cdef_find_dir_neon(const uint16_t *img, int stride, int32_t *var,
                       int coeff_shift) {
  uint32_t cost[8];
  uint32_t best_cost = 0;
  int best_dir = 0;
  int16x8_t lines[8];
  for (int i = 0; i < 8; i++) {
    uint16x8_t s = vld1q_u16(&img[i * stride]);
    lines[i] = vreinterpretq_s16_u16(
        vsubq_u16(vshlq_u16(s, vdupq_n_s16(-coeff_shift)), vdupq_n_u16(128)));
  }

  // Compute "mostly vertical" directions.
  uint32x4_t cost47 = compute_directions_neon(lines, cost + 4);

  array_reverse_transpose_8x8_neon(lines, lines);

  // Compute "mostly horizontal" directions.
  uint32x4_t cost03 = compute_directions_neon(lines, cost);

  uint32x4_t max_cost = vmaxq_u32(cost03, cost47);
  max_cost = vmaxq_u32(max_cost, vextq_u32(max_cost, max_cost, 2));
  max_cost = vmaxq_u32(max_cost, vextq_u32(max_cost, max_cost, 1));
  best_cost = vgetq_lane_u32(max_cost, 0);
  uint16x8_t idx = vcombine_u16(vqmovn_u32(vceqq_u32(max_cost, cost03)),
                                vqmovn_u32(vceqq_u32(max_cost, cost47)));
  uint8x16_t idx_u8 = vcombine_u8(vqmovn_u16(idx), vqmovn_u16(idx));
  best_dir = compute_best_dir(idx_u8);
  best_dir = get_msb(best_dir ^ (best_dir - 1));  // Count trailing zeros

  // Difference between the optimal variance and the variance along the
  // orthogonal direction. Again, the sum(x^2) terms cancel out.
  *var = best_cost - cost[(best_dir + 4) & 7];
  // We'd normally divide by 840, but dividing by 1024 is close enough
  // for what we're going to do with this.
  *var >>= 10;
  return best_dir;
}

void cdef_find_dir_dual_neon(const uint16_t *img1, const uint16_t *img2,
                             int stride, int32_t *var_out_1st,
                             int32_t *var_out_2nd, int coeff_shift,
                             int *out_dir_1st_8x8, int *out_dir_2nd_8x8) {
  // Process first 8x8.
  *out_dir_1st_8x8 = cdef_find_dir(img1, stride, var_out_1st, coeff_shift);

  // Process second 8x8.
  *out_dir_2nd_8x8 = cdef_find_dir(img2, stride, var_out_2nd, coeff_shift);
}

// sign(a-b) * min(abs(a-b), max(0, threshold - (abs(a-b) >> adjdamp)))
static INLINE int16x8_t constrain16(uint16x8_t a, uint16x8_t b,
                                    unsigned int threshold, int adjdamp) {
  int16x8_t diff = vreinterpretq_s16_u16(vsubq_u16(a, b));
  const int16x8_t sign = vshrq_n_s16(diff, 15);
  diff = vabsq_s16(diff);
  const uint16x8_t s =
      vqsubq_u16(vdupq_n_u16(threshold),
                 vreinterpretq_u16_s16(vshlq_s16(diff, vdupq_n_s16(-adjdamp))));
  return veorq_s16(vaddq_s16(sign, vminq_s16(diff, vreinterpretq_s16_u16(s))),
                   sign);
}

static INLINE uint16x8_t get_max_primary(const int is_lowbd, uint16x8_t *tap,
                                         uint16x8_t max,
                                         uint16x8_t cdef_large_value_mask) {
  if (is_lowbd) {
    uint8x16_t max_u8 = vreinterpretq_u8_u16(tap[0]);
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[1]));
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[2]));
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[3]));
    /* The source is 16 bits, however, we only really care about the lower
    8 bits.  The upper 8 bits contain the "large" flag.  After the final
    primary max has been calculated, zero out the upper 8 bits.  Use this
    to find the "16 bit" max. */
    max = vmaxq_u16(
        max, vandq_u16(vreinterpretq_u16_u8(max_u8), cdef_large_value_mask));
  } else {
    /* Convert CDEF_VERY_LARGE to 0 before calculating max. */
    max = vmaxq_u16(max, vandq_u16(tap[0], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[1], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[2], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[3], cdef_large_value_mask));
  }
  return max;
}

static INLINE uint16x8_t get_max_secondary(const int is_lowbd, uint16x8_t *tap,
                                           uint16x8_t max,
                                           uint16x8_t cdef_large_value_mask) {
  if (is_lowbd) {
    uint8x16_t max_u8 = vreinterpretq_u8_u16(tap[0]);
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[1]));
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[2]));
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[3]));
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[4]));
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[5]));
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[6]));
    max_u8 = vmaxq_u8(max_u8, vreinterpretq_u8_u16(tap[7]));
    /* The source is 16 bits, however, we only really care about the lower
    8 bits.  The upper 8 bits contain the "large" flag.  After the final
    primary max has been calculated, zero out the upper 8 bits.  Use this
    to find the "16 bit" max. */
    max = vmaxq_u16(
        max, vandq_u16(vreinterpretq_u16_u8(max_u8), cdef_large_value_mask));
  } else {
    /* Convert CDEF_VERY_LARGE to 0 before calculating max. */
    max = vmaxq_u16(max, vandq_u16(tap[0], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[1], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[2], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[3], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[4], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[5], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[6], cdef_large_value_mask));
    max = vmaxq_u16(max, vandq_u16(tap[7], cdef_large_value_mask));
  }
  return max;
}

static INLINE void filter_block_4x4(const int is_lowbd, void *dest, int dstride,
                                    const uint16_t *in, int pri_strength,
                                    int sec_strength, int dir, int pri_damping,
                                    int sec_damping, int coeff_shift,
                                    int height, int enable_primary,
                                    int enable_secondary) {
  uint8_t *dst8 = (uint8_t *)dest;
  uint16_t *dst16 = (uint16_t *)dest;
  const int clipping_required = enable_primary && enable_secondary;
  uint16x8_t max, min;
  const uint16x8_t cdef_large_value_mask =
      vdupq_n_u16(((uint16_t)~CDEF_VERY_LARGE));
  const int po1 = cdef_directions[dir][0];
  const int po2 = cdef_directions[dir][1];
  const int s1o1 = cdef_directions[dir + 2][0];
  const int s1o2 = cdef_directions[dir + 2][1];
  const int s2o1 = cdef_directions[dir - 2][0];
  const int s2o2 = cdef_directions[dir - 2][1];
  const int *pri_taps = cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
  const int *sec_taps = cdef_sec_taps;

  if (enable_primary && pri_strength) {
    pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
  }
  if (enable_secondary && sec_strength) {
    sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));
  }

  int h = height;
  do {
    int16x8_t sum = vdupq_n_s16(0);
    uint16x8_t s = load_unaligned_u16_4x2(in, CDEF_BSTRIDE);
    max = min = s;

    if (enable_primary) {
      uint16x8_t tap[4];

      // Primary near taps
      tap[0] = load_unaligned_u16_4x2(in + po1, CDEF_BSTRIDE);
      tap[1] = load_unaligned_u16_4x2(in - po1, CDEF_BSTRIDE);
      int16x8_t p0 = constrain16(tap[0], s, pri_strength, pri_damping);
      int16x8_t p1 = constrain16(tap[1], s, pri_strength, pri_damping);

      // sum += pri_taps[0] * (p0 + p1)
      p0 = vaddq_s16(p0, p1);
      sum = vmlaq_s16(sum, p0, vdupq_n_s16(pri_taps[0]));

      // Primary far taps
      tap[2] = load_unaligned_u16_4x2(in + po2, CDEF_BSTRIDE);
      tap[3] = load_unaligned_u16_4x2(in - po2, CDEF_BSTRIDE);
      p0 = constrain16(tap[2], s, pri_strength, pri_damping);
      p1 = constrain16(tap[3], s, pri_strength, pri_damping);

      // sum += pri_taps[1] * (p0 + p1)
      p0 = vaddq_s16(p0, p1);
      sum = vmlaq_s16(sum, p0, vdupq_n_s16(pri_taps[1]));

      if (clipping_required) {
        max = get_max_primary(is_lowbd, tap, max, cdef_large_value_mask);

        min = vminq_u16(min, tap[0]);
        min = vminq_u16(min, tap[1]);
        min = vminq_u16(min, tap[2]);
        min = vminq_u16(min, tap[3]);
      }
    }

    if (enable_secondary) {
      uint16x8_t tap[8];

      // Secondary near taps
      tap[0] = load_unaligned_u16_4x2(in + s1o1, CDEF_BSTRIDE);
      tap[1] = load_unaligned_u16_4x2(in - s1o1, CDEF_BSTRIDE);
      tap[2] = load_unaligned_u16_4x2(in + s2o1, CDEF_BSTRIDE);
      tap[3] = load_unaligned_u16_4x2(in - s2o1, CDEF_BSTRIDE);
      int16x8_t p0 = constrain16(tap[0], s, sec_strength, sec_damping);
      int16x8_t p1 = constrain16(tap[1], s, sec_strength, sec_damping);
      int16x8_t p2 = constrain16(tap[2], s, sec_strength, sec_damping);
      int16x8_t p3 = constrain16(tap[3], s, sec_strength, sec_damping);

      // sum += sec_taps[0] * (p0 + p1 + p2 + p3)
      p0 = vaddq_s16(p0, p1);
      p2 = vaddq_s16(p2, p3);
      p0 = vaddq_s16(p0, p2);
      sum = vmlaq_s16(sum, p0, vdupq_n_s16(sec_taps[0]));

      // Secondary far taps
      tap[4] = load_unaligned_u16_4x2(in + s1o2, CDEF_BSTRIDE);
      tap[5] = load_unaligned_u16_4x2(in - s1o2, CDEF_BSTRIDE);
      tap[6] = load_unaligned_u16_4x2(in + s2o2, CDEF_BSTRIDE);
      tap[7] = load_unaligned_u16_4x2(in - s2o2, CDEF_BSTRIDE);
      p0 = constrain16(tap[4], s, sec_strength, sec_damping);
      p1 = constrain16(tap[5], s, sec_strength, sec_damping);
      p2 = constrain16(tap[6], s, sec_strength, sec_damping);
      p3 = constrain16(tap[7], s, sec_strength, sec_damping);

      // sum += sec_taps[1] * (p0 + p1 + p2 + p3)
      p0 = vaddq_s16(p0, p1);
      p2 = vaddq_s16(p2, p3);
      p0 = vaddq_s16(p0, p2);
      sum = vmlaq_s16(sum, p0, vdupq_n_s16(sec_taps[1]));

      if (clipping_required) {
        max = get_max_secondary(is_lowbd, tap, max, cdef_large_value_mask);

        min = vminq_u16(min, tap[0]);
        min = vminq_u16(min, tap[1]);
        min = vminq_u16(min, tap[2]);
        min = vminq_u16(min, tap[3]);
        min = vminq_u16(min, tap[4]);
        min = vminq_u16(min, tap[5]);
        min = vminq_u16(min, tap[6]);
        min = vminq_u16(min, tap[7]);
      }
    }

    // res = row + ((sum - (sum < 0) + 8) >> 4)
    sum = vaddq_s16(sum, vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0))));
    int16x8_t res = vaddq_s16(sum, vdupq_n_s16(8));
    res = vshrq_n_s16(res, 4);
    res = vaddq_s16(vreinterpretq_s16_u16(s), res);

    if (clipping_required) {
      res = vminq_s16(vmaxq_s16(res, vreinterpretq_s16_u16(min)),
                      vreinterpretq_s16_u16(max));
    }

    if (is_lowbd) {
      const uint8x8_t res_128 = vqmovun_s16(res);
      store_unaligned_u8_4x2(dst8, dstride, res_128);
    } else {
      store_unaligned_u16_4x2(dst16, dstride, vreinterpretq_u16_s16(res));
    }

    in += 2 * CDEF_BSTRIDE;
    dst8 += 2 * dstride;
    dst16 += 2 * dstride;
    h -= 2;
  } while (h != 0);
}

static INLINE void filter_block_8x8(const int is_lowbd, void *dest, int dstride,
                                    const uint16_t *in, int pri_strength,
                                    int sec_strength, int dir, int pri_damping,
                                    int sec_damping, int coeff_shift,
                                    int height, int enable_primary,
                                    int enable_secondary) {
  uint8_t *dst8 = (uint8_t *)dest;
  uint16_t *dst16 = (uint16_t *)dest;
  const int clipping_required = enable_primary && enable_secondary;
  uint16x8_t max, min;
  const uint16x8_t cdef_large_value_mask =
      vdupq_n_u16(((uint16_t)~CDEF_VERY_LARGE));
  const int po1 = cdef_directions[dir][0];
  const int po2 = cdef_directions[dir][1];
  const int s1o1 = cdef_directions[dir + 2][0];
  const int s1o2 = cdef_directions[dir + 2][1];
  const int s2o1 = cdef_directions[dir - 2][0];
  const int s2o2 = cdef_directions[dir - 2][1];
  const int *pri_taps = cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
  const int *sec_taps = cdef_sec_taps;

  if (enable_primary && pri_strength) {
    pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
  }
  if (enable_secondary && sec_strength) {
    sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));
  }

  int h = height;
  do {
    int16x8_t sum = vdupq_n_s16(0);
    uint16x8_t s = vld1q_u16(in);
    max = min = s;

    if (enable_primary) {
      uint16x8_t tap[4];

      // Primary near taps
      tap[0] = vld1q_u16(in + po1);
      tap[1] = vld1q_u16(in - po1);
      int16x8_t p0 = constrain16(tap[0], s, pri_strength, pri_damping);
      int16x8_t p1 = constrain16(tap[1], s, pri_strength, pri_damping);

      // sum += pri_taps[0] * (p0 + p1)
      p0 = vaddq_s16(p0, p1);
      sum = vmlaq_s16(sum, p0, vdupq_n_s16(pri_taps[0]));

      // Primary far taps
      tap[2] = vld1q_u16(in + po2);
      p0 = constrain16(tap[2], s, pri_strength, pri_damping);
      tap[3] = vld1q_u16(in - po2);
      p1 = constrain16(tap[3], s, pri_strength, pri_damping);

      // sum += pri_taps[1] * (p0 + p1)
      p0 = vaddq_s16(p0, p1);
      sum = vmlaq_s16(sum, p0, vdupq_n_s16(pri_taps[1]));
      if (clipping_required) {
        max = get_max_primary(is_lowbd, tap, max, cdef_large_value_mask);

        min = vminq_u16(min, tap[0]);
        min = vminq_u16(min, tap[1]);
        min = vminq_u16(min, tap[2]);
        min = vminq_u16(min, tap[3]);
      }
    }

    if (enable_secondary) {
      uint16x8_t tap[8];

      // Secondary near taps
      tap[0] = vld1q_u16(in + s1o1);
      tap[1] = vld1q_u16(in - s1o1);
      tap[2] = vld1q_u16(in + s2o1);
      tap[3] = vld1q_u16(in - s2o1);
      int16x8_t p0 = constrain16(tap[0], s, sec_strength, sec_damping);
      int16x8_t p1 = constrain16(tap[1], s, sec_strength, sec_damping);
      int16x8_t p2 = constrain16(tap[2], s, sec_strength, sec_damping);
      int16x8_t p3 = constrain16(tap[3], s, sec_strength, sec_damping);

      // sum += sec_taps[0] * (p0 + p1 + p2 + p3)
      p0 = vaddq_s16(p0, p1);
      p2 = vaddq_s16(p2, p3);
      p0 = vaddq_s16(p0, p2);
      sum = vmlaq_s16(sum, p0, vdupq_n_s16(sec_taps[0]));

      // Secondary far taps
      tap[4] = vld1q_u16(in + s1o2);
      tap[5] = vld1q_u16(in - s1o2);
      tap[6] = vld1q_u16(in + s2o2);
      tap[7] = vld1q_u16(in - s2o2);
      p0 = constrain16(tap[4], s, sec_strength, sec_damping);
      p1 = constrain16(tap[5], s, sec_strength, sec_damping);
      p2 = constrain16(tap[6], s, sec_strength, sec_damping);
      p3 = constrain16(tap[7], s, sec_strength, sec_damping);

      // sum += sec_taps[1] * (p0 + p1 + p2 + p3)
      p0 = vaddq_s16(p0, p1);
      p2 = vaddq_s16(p2, p3);
      p0 = vaddq_s16(p0, p2);
      sum = vmlaq_s16(sum, p0, vdupq_n_s16(sec_taps[1]));

      if (clipping_required) {
        max = get_max_secondary(is_lowbd, tap, max, cdef_large_value_mask);

        min = vminq_u16(min, tap[0]);
        min = vminq_u16(min, tap[1]);
        min = vminq_u16(min, tap[2]);
        min = vminq_u16(min, tap[3]);
        min = vminq_u16(min, tap[4]);
        min = vminq_u16(min, tap[5]);
        min = vminq_u16(min, tap[6]);
        min = vminq_u16(min, tap[7]);
      }
    }

    // res = row + ((sum - (sum < 0) + 8) >> 4)
    sum = vaddq_s16(sum, vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0))));
    int16x8_t res = vaddq_s16(sum, vdupq_n_s16(8));
    res = vshrq_n_s16(res, 4);
    res = vaddq_s16(vreinterpretq_s16_u16(s), res);
    if (clipping_required) {
      res = vminq_s16(vmaxq_s16(res, vreinterpretq_s16_u16(min)),
                      vreinterpretq_s16_u16(max));
    }

    if (is_lowbd) {
      const uint8x8_t res_128 = vqmovun_s16(res);
      vst1_u8(dst8, res_128);
    } else {
      vst1q_u16(dst16, vreinterpretq_u16_s16(res));
    }

    in += CDEF_BSTRIDE;
    dst8 += dstride;
    dst16 += dstride;
  } while (--h != 0);
}

static INLINE void copy_block_4xh(const int is_lowbd, void *dest, int dstride,
                                  const uint16_t *in, int height) {
  uint8_t *dst8 = (uint8_t *)dest;
  uint16_t *dst16 = (uint16_t *)dest;

  int h = height;
  do {
    const uint16x8_t row = load_unaligned_u16_4x2(in, CDEF_BSTRIDE);
    if (is_lowbd) {
      const uint8x8_t res_128 = vqmovn_u16(row);
      store_unaligned_u8_4x2(dst8, dstride, res_128);
    } else {
      store_unaligned_u16_4x2(dst16, dstride, row);
    }

    in += 2 * CDEF_BSTRIDE;
    dst8 += 2 * dstride;
    dst16 += 2 * dstride;
    h -= 2;
  } while (h != 0);
}

static INLINE void copy_block_8xh(const int is_lowbd, void *dest, int dstride,
                                  const uint16_t *in, int height) {
  uint8_t *dst8 = (uint8_t *)dest;
  uint16_t *dst16 = (uint16_t *)dest;

  int h = height;
  do {
    const uint16x8_t row = vld1q_u16(in);
    if (is_lowbd) {
      const uint8x8_t res_128 = vqmovn_u16(row);
      vst1_u8(dst8, res_128);
    } else {
      vst1q_u16(dst16, row);
    }

    in += CDEF_BSTRIDE;
    dst8 += dstride;
    dst16 += dstride;
  } while (--h != 0);
}

void cdef_filter_8_0_neon(void *dest, int dstride, const uint16_t *in,
                          int pri_strength, int sec_strength, int dir,
                          int pri_damping, int sec_damping, int coeff_shift,
                          int block_width, int block_height) {
  if (block_width == 8) {
    filter_block_8x8(/*is_lowbd=*/1, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/1,
                     /*enable_secondary=*/1);
  } else {
    filter_block_4x4(/*is_lowbd=*/1, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/1,
                     /*enable_secondary=*/1);
  }
}

void cdef_filter_8_1_neon(void *dest, int dstride, const uint16_t *in,
                          int pri_strength, int sec_strength, int dir,
                          int pri_damping, int sec_damping, int coeff_shift,
                          int block_width, int block_height) {
  if (block_width == 8) {
    filter_block_8x8(/*is_lowbd=*/1, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/1,
                     /*enable_secondary=*/0);
  } else {
    filter_block_4x4(/*is_lowbd=*/1, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/1,
                     /*enable_secondary=*/0);
  }
}

void cdef_filter_8_2_neon(void *dest, int dstride, const uint16_t *in,
                          int pri_strength, int sec_strength, int dir,
                          int pri_damping, int sec_damping, int coeff_shift,
                          int block_width, int block_height) {
  if (block_width == 8) {
    filter_block_8x8(/*is_lowbd=*/1, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/0,
                     /*enable_secondary=*/1);
  } else {
    filter_block_4x4(/*is_lowbd=*/1, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/0,
                     /*enable_secondary=*/1);
  }
}

void cdef_filter_8_3_neon(void *dest, int dstride, const uint16_t *in,
                          int pri_strength, int sec_strength, int dir,
                          int pri_damping, int sec_damping, int coeff_shift,
                          int block_width, int block_height) {
  (void)pri_strength;
  (void)sec_strength;
  (void)dir;
  (void)pri_damping;
  (void)sec_damping;
  (void)coeff_shift;
  (void)block_width;
  if (block_width == 8) {
    copy_block_8xh(/*is_lowbd=*/1, dest, dstride, in, block_height);
  } else {
    copy_block_4xh(/*is_lowbd=*/1, dest, dstride, in, block_height);
  }
}

void cdef_filter_16_0_neon(void *dest, int dstride, const uint16_t *in,
                           int pri_strength, int sec_strength, int dir,
                           int pri_damping, int sec_damping, int coeff_shift,
                           int block_width, int block_height) {
  if (block_width == 8) {
    filter_block_8x8(/*is_lowbd=*/0, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/1,
                     /*enable_secondary=*/1);
  } else {
    filter_block_4x4(/*is_lowbd=*/0, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/1,
                     /*enable_secondary=*/1);
  }
}

void cdef_filter_16_1_neon(void *dest, int dstride, const uint16_t *in,
                           int pri_strength, int sec_strength, int dir,
                           int pri_damping, int sec_damping, int coeff_shift,
                           int block_width, int block_height) {
  if (block_width == 8) {
    filter_block_8x8(/*is_lowbd=*/0, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/1,
                     /*enable_secondary=*/0);
  } else {
    filter_block_4x4(/*is_lowbd=*/0, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/1,
                     /*enable_secondary=*/0);
  }
}

void cdef_filter_16_2_neon(void *dest, int dstride, const uint16_t *in,
                           int pri_strength, int sec_strength, int dir,
                           int pri_damping, int sec_damping, int coeff_shift,
                           int block_width, int block_height) {
  if (block_width == 8) {
    filter_block_8x8(/*is_lowbd=*/0, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/0,
                     /*enable_secondary=*/1);
  } else {
    filter_block_4x4(/*is_lowbd=*/0, dest, dstride, in, pri_strength,
                     sec_strength, dir, pri_damping, sec_damping, coeff_shift,
                     block_height, /*enable_primary=*/0,
                     /*enable_secondary=*/1);
  }
}

void cdef_filter_16_3_neon(void *dest, int dstride, const uint16_t *in,
                           int pri_strength, int sec_strength, int dir,
                           int pri_damping, int sec_damping, int coeff_shift,
                           int block_width, int block_height) {
  (void)pri_strength;
  (void)sec_strength;
  (void)dir;
  (void)pri_damping;
  (void)sec_damping;
  (void)coeff_shift;
  (void)block_width;
  if (block_width == 8) {
    copy_block_8xh(/*is_lowbd=*/0, dest, dstride, in, block_height);
  } else {
    copy_block_4xh(/*is_lowbd=*/0, dest, dstride, in, block_height);
  }
}
