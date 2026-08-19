// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/kernel/hlo/rank.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "libspu/core/context.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {
namespace {

void RunCryptoMoETopKTest(FieldType field) {
  // CryptoMoE Figure 4, expert 2:
  //
  // [[S_2]] = [0, 0.2, 0, 0.6]
  // t = 1
  //
  // Pi_topk([[S_2]], 1) should produce:
  // [[S'_2]] = [0.6]
  // [[K_2]]  = [3]
  const xt::xarray<float> priority_scores = {
      0.0F, 0.2F, 0.0F, 0.6F,
  };

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx =
            test::makeSPUContext(ProtocolKind::CHEETAH, field, lctx);

        // Secret priority scores coming from Pi_mux.
        auto scores_s =
            test::makeValue(&ctx, priority_scores, VIS_SECRET);

        // Public copy is used only as the fixed-point encoding oracle.
        auto scores_p =
            test::makeValue(&ctx, priority_scores, VIS_PUBLIC);
        auto quantized_scores =
            hal::dump_public_as<float>(&ctx, scores_p);

        ASSERT_TRUE(scores_s.isSecret());

        // CryptoMoE Algorithm 1, line 5:
        // [[K_i]], [[S'_i]] = Pi_topk([[S_i]], k=t).
        auto out = TopK(&ctx, scores_s, 1);

        ASSERT_EQ(out.size(), 2U);

        auto selected_scores_s = out[0];
        auto selected_indices_s = out[1];

        EXPECT_TRUE(selected_scores_s.isSecret());
        EXPECT_TRUE(selected_indices_s.isSecret());
        EXPECT_EQ(selected_scores_s.dtype(), DT_F32);

        // Reveal only for the correctness oracle at the end of the test.
        auto selected_scores_p =
            hal::reveal(&ctx, selected_scores_s);
        auto selected_scores =
            hal::dump_public_as<float>(&ctx, selected_scores_p);

        ASSERT_EQ(selected_scores.size(), 1U);

        // TopK must preserve the already encoded routing score exactly.
        EXPECT_FLOAT_EQ(selected_scores(0), quantized_scores(3));

        // The TopK index dtype depends on the ring field in this version:
        // FM32 -> DT_I32, FM64 -> DT_I64.
        int64_t selected_index = -1;

        auto selected_indices_p =
            hal::reveal(&ctx, selected_indices_s);

        if (field == FieldType::FM32) {
          EXPECT_EQ(selected_indices_s.dtype(), DT_I32);
          auto indices =
              hal::dump_public_as<int32_t>(&ctx, selected_indices_p);
          ASSERT_EQ(indices.size(), 1U);
          selected_index = indices(0);
        } else {
          EXPECT_EQ(selected_indices_s.dtype(), DT_I64);
          auto indices =
              hal::dump_public_as<int64_t>(&ctx, selected_indices_p);
          ASSERT_EQ(indices.size(), 1U);
          selected_index = indices(0);
        }

        EXPECT_EQ(selected_index, 3);
      });
}

TEST(CryptoMoETopKTest, FM32) {
  RunCryptoMoETopKTest(FieldType::FM32);
}

TEST(CryptoMoETopKTest, FM64) {
  RunCryptoMoETopKTest(FieldType::FM64);
}

void RunCryptoMoETopKStrictTieTest(FieldType field) {
  // CipherGPT requires TopK elements to be strictly comparable by
  // appending the original index.
  //
  // Three candidates have exactly the same score 0.6:
  //
  //   score: [0, 0.6, 0, 0.6, 0, 0.6]
  //   index:  0    1  2    3  4    5
  //
  // With descending lexicographic (score,index), Top-2 must select
  // original indices {5, 3}, excluding index 1.
  const xt::xarray<float> priority_scores = {
      0.0F, 0.6F, 0.0F, 0.6F, 0.0F, 0.6F,
  };

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx =
            test::makeSPUContext(ProtocolKind::CHEETAH, field, lctx);

        auto scores_s =
            test::makeValue(&ctx, priority_scores, VIS_SECRET);

        auto scores_p =
            test::makeValue(&ctx, priority_scores, VIS_PUBLIC);
        auto quantized_scores =
            hal::dump_public_as<float>(&ctx, scores_p);

        auto out = TopK(&ctx, scores_s, 2);

        ASSERT_EQ(out.size(), 2U);
        EXPECT_TRUE(out[0].isSecret());
        EXPECT_TRUE(out[1].isSecret());

        auto selected_scores =
            hal::dump_public_as<float>(
                &ctx, hal::reveal(&ctx, out[0]));

        ASSERT_EQ(selected_scores.size(), 2U);
        EXPECT_FLOAT_EQ(selected_scores(0), quantized_scores(1));
        EXPECT_FLOAT_EQ(selected_scores(1), quantized_scores(1));

        int64_t idx0 = -1;
        int64_t idx1 = -1;

        auto indices_p = hal::reveal(&ctx, out[1]);

        if (field == FieldType::FM32) {
          EXPECT_EQ(out[1].dtype(), DT_I32);

          auto indices =
              hal::dump_public_as<int32_t>(&ctx, indices_p);

          ASSERT_EQ(indices.size(), 2U);
          idx0 = indices(0);
          idx1 = indices(1);
        } else {
          EXPECT_EQ(out[1].dtype(), DT_I64);

          auto indices =
              hal::dump_public_as<int64_t>(&ctx, indices_p);

          ASSERT_EQ(indices.size(), 2U);
          idx0 = indices(0);
          idx1 = indices(1);
        }

        // QuickSelect need not return the selected set in sorted order,
        // so verify the set rather than its output ordering.
        EXPECT_TRUE(
            (idx0 == 5 && idx1 == 3) ||
            (idx0 == 3 && idx1 == 5));

        EXPECT_NE(idx0, 1);
        EXPECT_NE(idx1, 1);
      });
}

TEST(CryptoMoETopKTest, StrictScoreIndexTieFM32) {
  RunCryptoMoETopKStrictTieTest(FieldType::FM32);
}

TEST(CryptoMoETopKTest, StrictScoreIndexTieFM64) {
  RunCryptoMoETopKStrictTieTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hlo
