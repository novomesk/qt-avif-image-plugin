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

#include <cstdlib>
#include <new>
#include <tuple>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#include "aom/aom_codec.h"
#include "aom/aom_integer.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/aom_timer.h"
#include "aom_ports/mem.h"
#include "test/acm_random.h"
#include "av1/encoder/palette.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace AV1Kmeans {
typedef void (*av1_calc_indices_dim1_func)(const int *data,
                                           const int *centroids,
                                           uint8_t *indices, int n, int k);

typedef std::tuple<av1_calc_indices_dim1_func, BLOCK_SIZE>
    av1_calc_indices_dim1Param;

class AV1KmeansTest
    : public ::testing::TestWithParam<av1_calc_indices_dim1Param> {
 public:
  ~AV1KmeansTest();
  void SetUp();

  void TearDown();

 protected:
  void RunCheckOutput(av1_calc_indices_dim1_func test_impl, BLOCK_SIZE bsize,
                      int centroids);
  void RunSpeedTest(av1_calc_indices_dim1_func test_impl, BLOCK_SIZE bsize,
                    int centroids);
  bool CheckResult(int n) {
    for (int idx = 0; idx < n; ++idx) {
      if (indices1_[idx] != indices2_[idx]) {
        printf("%d ", idx);
        printf("%d != %d ", indices1_[idx], indices2_[idx]);
        return false;
      }
    }
    return true;
  }

  libaom_test::ACMRandom rnd_;
  int data_[5096];
  int centroids_[8];
  uint8_t indices1_[5096];
  uint8_t indices2_[5096];
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1KmeansTest);

AV1KmeansTest::~AV1KmeansTest() { ; }

void AV1KmeansTest::SetUp() {
  rnd_.Reset(libaom_test::ACMRandom::DeterministicSeed());
  /*uint8_t indices1_[5096];
  uint8_t indices2_[5096];
  int data_[5096];*/
  for (int i = 0; i < 5096; ++i) {
    data_[i] = (int)rnd_.Rand8() << 4;
  }
  for (int i = 0; i < 8; i++) {
    centroids_[i] = (int)rnd_.Rand8() << 4;
  }
}

void AV1KmeansTest::TearDown() { libaom_test::ClearSystemState(); }

void AV1KmeansTest::RunCheckOutput(av1_calc_indices_dim1_func test_impl,
                                   BLOCK_SIZE bsize, int k) {
  const int w = block_size_wide[bsize];
  const int h = block_size_high[bsize];
  const int n = w * h;
  av1_calc_indices_dim1_c(data_, centroids_, indices1_, n, k);
  test_impl(data_, centroids_, indices2_, n, k);

  ASSERT_EQ(CheckResult(n), true) << " block " << bsize << " Centroids " << n;
}

void AV1KmeansTest::RunSpeedTest(av1_calc_indices_dim1_func test_impl,
                                 BLOCK_SIZE bsize, int k) {
  const int w = block_size_wide[bsize];
  const int h = block_size_high[bsize];
  const int n = w * h;
  const int num_loops = 1000000000 / n;

  av1_calc_indices_dim1_func funcs[2] = { av1_calc_indices_dim1_c, test_impl };
  double elapsed_time[2] = { 0 };
  for (int i = 0; i < 2; ++i) {
    aom_usec_timer timer;
    aom_usec_timer_start(&timer);
    av1_calc_indices_dim1_func func = funcs[i];
    for (int j = 0; j < num_loops; ++j) {
      func(data_, centroids_, indices1_, n, k);
    }
    aom_usec_timer_mark(&timer);
    double time = static_cast<double>(aom_usec_timer_elapsed(&timer));
    elapsed_time[i] = 1000.0 * time / num_loops;
  }
  printf("av1_calc_indices_dim1 indices= %d centroids=%d: %7.2f/%7.2fns", n, k,
         elapsed_time[0], elapsed_time[1]);
  printf("(%3.2f)\n", elapsed_time[0] / elapsed_time[1]);
}

TEST_P(AV1KmeansTest, CheckOutput) {
  // centroids = 2..8
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 2);
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 3);
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 4);
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 5);
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 6);
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 7);
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 8);
}

TEST_P(AV1KmeansTest, DISABLED_Speed) {
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1), 2);
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1), 3);
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1), 4);
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1), 5);
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1), 6);
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1), 7);
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1), 8);
}

#if HAVE_AVX2
const BLOCK_SIZE kValidBlockSize[] = { BLOCK_8X8,   BLOCK_8X16,  BLOCK_8X32,
                                       BLOCK_16X8,  BLOCK_16X16, BLOCK_16X32,
                                       BLOCK_32X8,  BLOCK_32X16, BLOCK_32X32,
                                       BLOCK_32X64, BLOCK_64X32, BLOCK_64X64,
                                       BLOCK_16X64, BLOCK_64X16 };

INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1KmeansTest,
    ::testing::Combine(::testing::Values(&av1_calc_indices_dim1_avx2),
                       ::testing::ValuesIn(kValidBlockSize)));
#endif

}  // namespace AV1Kmeans
