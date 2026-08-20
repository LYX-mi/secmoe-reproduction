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

#include "libspu/kernel/hlo/cryptomoe_combine.h"
#include "libspu/kernel/hlo/cryptomoe_dispatch.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#include "libspu/core/context.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {
namespace {

void RunCryptoMoEEndToEndTest(FieldType field) {
  // CryptoMoE Figure 4 routing:
  //
  // K = [[0, 2],
  //      [1, 2]]
  //
  // W = [[0.7, 0.2],
  //      [0.2, 0.6]]
  //
  // Use identity experts, so y_Ei = X_i. Algorithm 2 should therefore return
  // token A weighted by 0.7 and token B weighted by 0.2 + 0.6.
  const xt::xarray<int32_t> routing_indices = {
      {0, 2},
      {1, 2},
  };
  const xt::xarray<float> routing_weights = {
      {0.7F, 0.2F},
      {0.2F, 0.6F},
  };
  const xt::xarray<float> token_embeddings = {
      {10.0F, 20.0F},
      {30.0F, 40.0F},
  };

  constexpr int64_t num_experts = 4;
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

        std::vector<spu::Value> expert_outputs_s;
        std::vector<spu::Value> onehots_s;
        std::vector<spu::Value> scores_s;
        expert_outputs_s.reserve(num_experts);
        onehots_s.reserve(num_experts);
        scores_s.reserve(num_experts);

        for (int64_t expert = 0; expert < num_experts; ++expert) {
          auto dispatch =
              CryptoMoEDispatchWithAux(&ctx, routing_indices_s,
                                         routing_weights_s, tokens_s, expert,
                                         capacity);
          expert_outputs_s.push_back(dispatch.tokens);
          onehots_s.push_back(dispatch.onehot);
          scores_s.push_back(dispatch.scores);
        }

        auto output_s =
            CryptoMoECombine(&ctx, expert_outputs_s, onehots_s, scores_s);

        ASSERT_TRUE(output_s.isSecret());
        ASSERT_EQ(output_s.dtype(), DT_F32);
        ASSERT_EQ(output_s.shape(), Shape({2, 2}));

        auto output_p = hal::_s2p(&ctx, output_s).setDtype(DT_F32);
        auto got = hal::dump_public_as<float>(&ctx, output_p);

        const xt::xarray<float> expected = {
            {7.0F, 14.0F},
            {24.0F, 32.0F},
        };
        EXPECT_TRUE(xt::allclose(expected, got, 0.01, 0.001));
      });
}

TEST(CryptoMoEEndToEndTest, FM32) {
  RunCryptoMoEEndToEndTest(FieldType::FM32);
}

TEST(CryptoMoEEndToEndTest, FM64) {
  RunCryptoMoEEndToEndTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hlo
