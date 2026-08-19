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

#include "libspu/kernel/hlo/cryptomoe_dispatch.h"

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

  constexpr int64_t expert_id = 2;
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

        // CryptoMoE Algorithm 1, lines 2-8.
        // No intermediate value is revealed inside CryptoMoEDispatch.
        auto dispatched_tokens_s =
            CryptoMoEDispatch(&ctx, routing_indices_s, routing_weights_s,
                              tokens_s, expert_id, capacity);

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
