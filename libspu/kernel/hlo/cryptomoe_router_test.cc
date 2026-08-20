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

#include "libspu/kernel/hlo/cryptomoe_router.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "libspu/core/context.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {
namespace {

void RunCryptoMoERouterTest(FieldType field) {
  // Two tokens, hidden dimension d = 2, n = 3 experts, top-k = 2.
  //
  // tokens = [[1, 0],
  //           [0, 1]]
  //
  // router_weight = [[ 4, 1, -2],
  //                    [-3, 5,  0]]
  //
  // Therefore Linear(tokens) produces:
  // logits = [[ 4, 1, -2],
  //           [-3, 5,  0]]
  //
  // The expected top-2 expert sets after Softmax are:
  // token 0 -> {0, 1}
  // token 1 -> {1, 2}
  const xt::xarray<float> token_embeddings = {
      {1.0F, 0.0F},
      {0.0F, 1.0F},
  };

  const xt::xarray<float> router_weight = {
      {4.0F, 1.0F, -2.0F},
     {-3.0F, 5.0F, 0.0F},
  };

  constexpr int64_t top_k = 2;

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

        // CryptoMoE Equation (1):
        // [[W]], [[K]] = TopK(Softmax(Linear([[x]])), k).
        auto routing =
            CryptoMoERoute(&ctx, tokens_s, router_weight_s, top_k);

        ASSERT_TRUE(routing.weights.isSecret());
        ASSERT_TRUE(routing.indices.isSecret());
        ASSERT_EQ(routing.weights.dtype(), DT_F32);
        ASSERT_EQ(routing.weights.shape(), Shape({2, top_k}));
        ASSERT_EQ(routing.indices.shape(), Shape({2, top_k}));

        auto weights_p = hal::_s2p(&ctx, routing.weights).setDtype(DT_F32);
        auto weights = hal::dump_public_as<float>(&ctx, weights_p);

        auto indices_p = hal::_s2p(&ctx, routing.indices).setDtype(routing.indices.dtype());
        std::array<int64_t, 4> indices{};

        if (field == FieldType::FM32) {
          ASSERT_EQ(routing.indices.dtype(), DT_I32);
          auto got = hal::dump_public_as<int32_t>(&ctx, indices_p);
          ASSERT_EQ(got.shape().size(), 2U);
          ASSERT_EQ(got.shape()[0], 2U);
          ASSERT_EQ(got.shape()[1], 2U);
          for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = got.data()[i];
          }
        } else {
          ASSERT_EQ(routing.indices.dtype(), DT_I64);
          auto got = hal::dump_public_as<int64_t>(&ctx, indices_p);
          ASSERT_EQ(got.shape().size(), 2U);
          ASSERT_EQ(got.shape()[0], 2U);
          ASSERT_EQ(got.shape()[1], 2U);
          for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = got.data()[i];
          }
        }

        // TopK does not guarantee sorted output, so verify each token's
        // selected expert set independent of order.
        std::array<int64_t, 2> token0 = {indices[0], indices[1]};
        std::array<int64_t, 2> token1 = {indices[2], indices[3]};
        std::sort(token0.begin(), token0.end());
        std::sort(token1.begin(), token1.end());

        EXPECT_EQ(token0[0], 0);
        EXPECT_EQ(token0[1], 1);
        EXPECT_EQ(token1[0], 1);
        EXPECT_EQ(token1[1], 2);

        // Validate that each returned routing weight matches the Softmax
        // probability of its returned expert index. f_neg_exp_taylor is an
        // approximation, so use a small absolute tolerance.
        const std::array<std::array<double, 3>, 2> logits = {{
            {{4.0, 1.0, -2.0}},
            {{-3.0, 5.0, 0.0}},
        }};
        std::array<std::array<double, 3>, 2> expected{};

        for (size_t r = 0; r < 2; ++r) {
          const double row_max =
              *std::max_element(logits[r].begin(), logits[r].end());
          double divisor = 0.0;
          for (size_t e = 0; e < 3; ++e) {
            expected[r][e] = std::exp(logits[r][e] - row_max);
            divisor += expected[r][e];
          }
          for (size_t e = 0; e < 3; ++e) {
            expected[r][e] /= divisor;
          }
        }

        const double atol = field == FieldType::FM32 ? 0.05 : 0.03;
        for (size_t r = 0; r < 2; ++r) {
          for (size_t j = 0; j < 2; ++j) {
            const int64_t expert = indices[r * 2 + j];
            ASSERT_GE(expert, 0);
            ASSERT_LT(expert, 3);
            EXPECT_NEAR(weights(r, j), expected[r][expert], atol);
          }
        }
      });
}

TEST(CryptoMoERouterTest, FM32) {
  RunCryptoMoERouterTest(FieldType::FM32);
}

TEST(CryptoMoERouterTest, FM64) {
  RunCryptoMoERouterTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hlo
