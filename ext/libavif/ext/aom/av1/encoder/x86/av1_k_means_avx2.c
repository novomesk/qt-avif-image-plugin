/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#include <immintrin.h>  // AVX2

#include "config/aom_dsp_rtcd.h"
#include "aom_dsp/x86/synonyms.h"

void av1_calc_indices_dim1_avx2(const int *data, const int *centroids,
                                uint8_t *indices, int n, int k) {
  __m256i dist[PALETTE_MAX_SIZE];
  __m256i v_zero = _mm256_setzero_si256();

  for (int i = 0; i < n; i += 8) {
    __m256i ind = _mm256_loadu_si256((__m256i *)data);
    for (int j = 0; j < k; j++) {
      __m256i cent = _mm256_set1_epi32((uint32_t)centroids[j]);
      __m256i d1 = _mm256_sub_epi32(ind, cent);
      dist[j] = _mm256_mullo_epi32(d1, d1);
    }

    ind = _mm256_setzero_si256();
    for (int j = 1; j < k; j++) {
      __m256i cmp = _mm256_cmpgt_epi32(dist[0], dist[j]);
      __m256i dist1 = _mm256_andnot_si256(cmp, dist[0]);
      __m256i dist2 = _mm256_and_si256(cmp, dist[j]);
      dist[0] = _mm256_or_si256(dist1, dist2);
      __m256i ind1 = _mm256_set1_epi32(j);
      ind = _mm256_or_si256(_mm256_andnot_si256(cmp, ind),
                            _mm256_and_si256(cmp, ind1));
    }

    __m256i p1 = _mm256_packus_epi32(ind, v_zero);
    __m256i px = _mm256_permute4x64_epi64(p1, 0x58);
    __m256i p2 = _mm256_packus_epi16(px, v_zero);
    __m128i d1 = _mm256_extracti128_si256(p2, 0);

    _mm_storel_epi64((__m128i *)indices, d1);

    indices += 8;
    data += 8;
  }
}
