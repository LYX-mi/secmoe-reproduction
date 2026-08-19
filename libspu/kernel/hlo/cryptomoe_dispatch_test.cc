// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {
namespace {

void RunCryptoMoEDispatchTest(FieldType field) {
  // CryptoMoE Figure 4.
  //
  // m = 2 tokens, k = 2 routed experts per token.
  //
  // K = [[0, 2],
  //      [1, 2]]
  //
  // W = [[0.7, 0.2],
  //      [0.2, 0.6]]
  //
  // For expert i = 2 and capacity t = 1:
  //
  // line 3: M_2  = [0, 1, 0, 1]
  // line 4: S_2  = [0, 0.2, 0, 0.6]
  // line 5: K_2  = [3]
  // line 6: K'_2 = [3 // 2] = [1]
  // line 7: O_2  = [[0, 1]]
  // line 8: X_2  = O_2 * x = x[1]

  const xt::xarray<int32_t> routing_indices = {
      {0, 2},
      {1, 2},
  };

  const xt::xarray<float> routing_weights = {
      {0.7F, 0.2F},
      {0.2F, 0.6F},
  };

  const xt::xarray<float> token_embeddings = {
      {1.25F, -2.0F},
      {3.5F, 4.25F},
  };

  const xt::xarray<int32_t> expert_ids = {
      2, 2, 2, 2,
  };

  constexpr int64_t num_tokens = 2;
  constexpr int64_t routed_experts_per_token = 2;
  constexpr int64_t capacity = 1;

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx =
            test::makeSPUContext(ProtocolKind::CHEETAH, field, lctx);

        auto routing_indices_s =
            test::makeValue(&ctx, routing_indices, VIS_SECRET);
        auto routing_weights_s =
            test::makeValue(&ctx, routing_weights, VIS_SECRET);
        auto tokens_s =
            test::makeValue(&ctx, token_embeddings, VIS_SECRET);

        ASSERT_TRUE(routing_indices_s.isSecret());
        ASSERT_TRUE(routing_weights_s.isSecret());
        ASSERT_TRUE(tokens_s.isSecret());

        // CryptoMoE Algorithm 1, line 2:
        // Flatten K and W from [m, k] to [m*k].
        auto k_flat_s =
            hal::reshape(&ctx, routing_indices_s,
                         {num_tokens * routed_experts_per_token});
        auto w_flat_s =
            hal::reshape(&ctx, routing_weights_s,
                         {num_tokens * routed_experts_per_token});

        auto expert_ids_p =
            hal::constant(&ctx, expert_ids, DT_I32);

        // CryptoMoE Algorithm 1, line 3:
        // [[M_i]]^B = Pi_equal([[K]], i).
        auto mask_s =
            hal::equal(&ctx, k_flat_s, expert_ids_p);

        ASSERT_TRUE(mask_s.isSecret());
        ASSERT_EQ(mask_s.dtype(), DT_I1);

        // CryptoMoE Algorithm 1, line 4:
        // [[S_i]] = Pi_mux([[M_i]]^B, [[W]]).
        auto zero_p =
            hal::zeros(&ctx, DT_F32,
                       {num_tokens * routed_experts_per_token});

        auto scores_s =
            hal::select(&ctx, mask_s, w_flat_s, zero_p);

        ASSERT_TRUE(scores_s.isSecret());
        ASSERT_EQ(scores_s.dtype(), DT_F32);

        // CryptoMoE Algorithm 1, line 5:
        // [[K_i]], [[S'_i]] = Pi_topk([[S_i]], t).
        auto topk_out =
            TopK(&ctx, scores_s, capacity);

        ASSERT_EQ(topk_out.size(), 2U);

        auto selected_scores_s = topk_out[0];
        auto selected_indices_s = topk_out[1];

        ASSERT_TRUE(selected_scores_s.isSecret());
        ASSERT_TRUE(selected_indices_s.isSecret());

        // CryptoMoE Algorithm 1, line 6:
        // [[K'_i]] = [[K_i]] // k.
        auto token_indices_s = [&]() {
          if (field == FieldType::FM32) {
            const xt::xarray<int32_t> k = {
                static_cast<int32_t>(routed_experts_per_token)};
            auto k_p =
                test::makeValue(&ctx, k, VIS_PUBLIC);
            return hal::div(&ctx, selected_indices_s, k_p);
          }

          const xt::xarray<int64_t> k = {
              routed_experts_per_token};
          auto k_p =
              test::makeValue(&ctx, k, VIS_PUBLIC);
          return hal::div(&ctx, selected_indices_s, k_p);
        }();

        ASSERT_TRUE(token_indices_s.isSecret());

        // CryptoMoE Algorithm 1, line 7:
        // [[O_i]] = Pi_onehot([[K'_i]], m).
        //
        // Pi_onehot is the paper-defined batch of Pi_equal calls.
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

        ASSERT_TRUE(onehot_s.isSecret());
        ASSERT_EQ(onehot_s.dtype(), DT_I1);
        ASSERT_EQ(onehot_s.shape(), Shape({capacity, num_tokens}));

        // CryptoMoE Algorithm 1, line 8:
        // [[X_i]] = [[O_i]] * [[x]].
        //
        // Pi_equal produces Boolean shares. Convert the 0/1 one-hot matrix
        // to arithmetic sharing, then use Cheetah's secret-secret MatMul,
        // whose cross terms are evaluated through DotOLE.
        auto onehot_a =
            hal::_prefer_a(&ctx, onehot_s);

        ASSERT_TRUE(onehot_a.isSecret());
        ASSERT_EQ(onehot_a.dtype(), DT_I1);

        auto dispatched_tokens_s =
            hal::matmul(&ctx, onehot_a, tokens_s);

        ASSERT_TRUE(dispatched_tokens_s.isSecret());
        ASSERT_EQ(dispatched_tokens_s.dtype(), DT_F32);
        ASSERT_EQ(dispatched_tokens_s.shape(), Shape({capacity, 2}));

        // No intermediate value above is revealed.
        // Reveal only final X_i for the correctness oracle.
        auto dispatched_tokens_p =
            hal::_s2p(&ctx, dispatched_tokens_s).setDtype(DT_F32);
        auto got =
            hal::dump_public_as<float>(&ctx, dispatched_tokens_p);

        auto tokens_oracle_p =
            test::makeValue(&ctx, token_embeddings, VIS_PUBLIC);
        auto quantized_tokens =
            hal::dump_public_as<float>(&ctx, tokens_oracle_p);

        ASSERT_EQ(got.shape().size(), 2U);
        ASSERT_EQ(got.shape()[0], 1U);
        ASSERT_EQ(got.shape()[1], 2U);

        EXPECT_FLOAT_EQ(got(0, 0), quantized_tokens(1, 0));
        EXPECT_FLOAT_EQ(got(0, 1), quantized_tokens(1, 1));
      });
}

TEST(CryptoMoEDispatchTest, FM32) {
  RunCryptoMoEDispatchTest(FieldType::FM32);
}

TEST(CryptoMoEDispatchTest, FM64) {
  RunCryptoMoEDispatchTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hlo
