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
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
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

  // Original token embeddings x, shape [m=2, d=2].
  // Since K'_2 = 1, Algorithm 1 line 8 must select the second row.
  const xt::xarray<float> token_embeddings = {
      {1.25F, -2.0F},
      {3.5F, 4.25F},
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

        // CryptoMoE Algorithm 1, line 6:
        // [[K'_i]] = [[K_i]] // k.
        //
        // Figure 4 has k = 2 routed experts per token. Since TopK returns
        // flattened routing position K_2 = 3, integer division must recover
        // original token index K'_2 = 3 // 2 = 1.
        auto token_indices_s = [&]() {
          if (field == FieldType::FM32) {
            const xt::xarray<int32_t> routing_k = {2};
            auto routing_k_p =
                test::makeValue(&ctx, routing_k, VIS_PUBLIC);
            return hal::div(&ctx, selected_indices_s, routing_k_p);
          }

          const xt::xarray<int64_t> routing_k = {2};
          auto routing_k_p =
              test::makeValue(&ctx, routing_k, VIS_PUBLIC);
          return hal::div(&ctx, selected_indices_s, routing_k_p);
        }();

        EXPECT_TRUE(token_indices_s.isSecret());
        EXPECT_EQ(token_indices_s.dtype(), selected_indices_s.dtype());

        // CryptoMoE Algorithm 1, line 7:
        // [[O_i]] = Pi_onehot([[K'_i]], m).
        //
        // Figure 4 has m = 2 original tokens.  Pi_onehot is implemented
        // exactly as defined in CryptoMoE: compare every secret token index
        // against the public candidates [0, ..., m - 1].
        constexpr int64_t num_tokens = 2;
        constexpr int64_t capacity = 1;

        auto token_candidates_p =
            hal::iota(&ctx, selected_indices_s.dtype(), num_tokens);

        auto token_indices_matrix_s =
            hal::broadcast_to(&ctx, token_indices_s,
                              {capacity, num_tokens}, {0});

        auto token_candidates_matrix_p =
            hal::broadcast_to(&ctx, token_candidates_p,
                              {capacity, num_tokens}, {1});

        auto onehot_s =
            hal::equal(&ctx, token_indices_matrix_s,
                       token_candidates_matrix_p);

        EXPECT_TRUE(onehot_s.isSecret());
        EXPECT_EQ(onehot_s.dtype(), DT_I1);
        EXPECT_EQ(onehot_s.shape(), Shape({capacity, num_tokens}));

        // CryptoMoE Algorithm 1, line 8:
        // [[X_i]] = [[O_i]] * [[x]].
        //
        // Equality returns a Boolean share. Convert it to arithmetic sharing
        // before matrix multiplication. Keep the integer 0/1 dtype so
        // hal::matmul uses the mixed integer/fixed-point fast path.
        auto onehot_a = hal::_prefer_a(&ctx, onehot_s);

        EXPECT_TRUE(onehot_a.isSecret());
        EXPECT_EQ(onehot_a.dtype(), DT_I1);

        auto tokens_s =
            test::makeValue(&ctx, token_embeddings, VIS_SECRET);

        ASSERT_TRUE(tokens_s.isSecret());
        ASSERT_EQ(tokens_s.dtype(), DT_F32);

        auto selected_tokens_s =
            hal::matmul(&ctx, onehot_a, tokens_s);

        EXPECT_TRUE(selected_tokens_s.isSecret());
        EXPECT_EQ(selected_tokens_s.dtype(), DT_F32);
        EXPECT_EQ(selected_tokens_s.shape(), Shape({capacity, 2}));

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

        auto token_indices_p =
            hal::reveal(&ctx, token_indices_s);

        if (field == FieldType::FM32) {
          auto token_indices =
              hal::dump_public_as<int32_t>(&ctx, token_indices_p);
          ASSERT_EQ(token_indices.size(), 1U);
          EXPECT_EQ(token_indices(0), 1);
        } else {
          auto token_indices =
              hal::dump_public_as<int64_t>(&ctx, token_indices_p);
          ASSERT_EQ(token_indices.size(), 1U);
          EXPECT_EQ(token_indices(0), 1);
        }

        // K'_2 = 1 over m = 2 tokens must encode as [0, 1].
        auto onehot_p =
            hal::_s2p(&ctx, onehot_s).setDtype(DT_I1);
        auto onehot =
            hal::dump_public_as<bool>(&ctx, onehot_p);

        ASSERT_EQ(onehot.shape().size(), 2U);
        EXPECT_EQ(onehot.shape()[0], 1U);
        EXPECT_EQ(onehot.shape()[1], 2U);
        EXPECT_FALSE(onehot(0, 0));
        EXPECT_TRUE(onehot(0, 1));

        // O_2 = [0, 1], so O_2 * x must return the second token embedding.
        auto selected_tokens_p =
            hal::reveal(&ctx, selected_tokens_s);
        auto selected_tokens =
            hal::dump_public_as<float>(&ctx, selected_tokens_p);

        auto tokens_oracle_p =
            test::makeValue(&ctx, token_embeddings, VIS_PUBLIC);
        auto quantized_tokens =
            hal::dump_public_as<float>(&ctx, tokens_oracle_p);

        ASSERT_EQ(selected_tokens.shape().size(), 2U);
        EXPECT_EQ(selected_tokens.shape()[0], 1U);
        EXPECT_EQ(selected_tokens.shape()[1], 2U);
        EXPECT_FLOAT_EQ(selected_tokens(0, 0), quantized_tokens(1, 0));
        EXPECT_FLOAT_EQ(selected_tokens(0, 1), quantized_tokens(1, 1));
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
