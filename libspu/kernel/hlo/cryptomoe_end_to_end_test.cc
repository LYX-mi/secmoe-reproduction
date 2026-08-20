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
#include "libspu/kernel/hlo/cryptomoe_router.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "libspu/core/context.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {
namespace {

void RunCryptoMoEEndToEndTest(FieldType field) {
  // End-to-end secure CryptoMoE routing path:
  //
  //   Gate Routing -> Dispatch -> identity experts -> Combine
  //
  // Gate Routing computes secret [[W]] and [[K]] from the secret token
  // embeddings and secret router weights. Identity experts keep this test
  // focused on the routing, dispatch, and combine protocols.
  const xt::xarray<float> token_embeddings = {
      {1.0F, 0.0F},
      {0.0F, 1.0F},
  };

  // Secret gate linear-layer weights [d=2, n=4].
  //
  // Linear(tokens) =
  //   [[ 4,  1, -2, -3],
  //    [-3, -2,  5,  2]]
  //
  // Hence top-2 routing is:
  //   token 0 -> experts {0, 1}
  //   token 1 -> experts {2, 3}
  const xt::xarray<float> router_weight = {
      {4.0F, 1.0F, -2.0F, -3.0F},
      {-3.0F, -2.0F, 5.0F, 2.0F},
  };

  constexpr int64_t num_experts = 4;
  constexpr int64_t top_k = 2;
  constexpr int64_t capacity = 1;

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig config;
        config.set_protocol(ProtocolKind::CHEETAH);
        config.set_field(field);
        config.set_fxp_fraction_bits(field == FieldType::FM32 ? 10 : 16);
        config.set_fxp_exp_iters(5);
        config.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
        config.mutable_cheetah_2pc_config()->set_approx_less_precision(4);

        SPUContext ctx = test::makeSPUContext(config, lctx);

        auto tokens_s =
            test::makeValue(&ctx, token_embeddings, VIS_SECRET);
        auto router_weight_s =
            test::makeValue(&ctx, router_weight, VIS_SECRET);

        ASSERT_TRUE(tokens_s.isSecret());
        ASSERT_TRUE(router_weight_s.isSecret());

        // Gate Routing: [[W]], [[K]] =
        // TopK(Softmax(Linear([[x]])), k).
        auto routing =
            CryptoMoERoute(&ctx, tokens_s, router_weight_s, top_k);

        ASSERT_TRUE(routing.weights.isSecret());
        ASSERT_TRUE(routing.indices.isSecret());

        auto routing_indices_s = routing.indices;
        auto routing_weights_s = routing.weights;

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
            {0.9967806F, 0.0F},
            {0.0F, 0.9988132F},
        };
        EXPECT_NEAR(got(0, 0), expected(0, 0), 0.01);
        EXPECT_NEAR(got(0, 1), expected(0, 1), 0.01);
        EXPECT_NEAR(got(1, 0), expected(1, 0), 0.01);
        EXPECT_NEAR(got(1, 1), expected(1, 1), 0.01);
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
