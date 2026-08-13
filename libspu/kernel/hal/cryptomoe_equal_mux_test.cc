// Copyright 2021 Ant Group Co., Ltd.
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

#include "libspu/kernel/hal/polymorphic.h"

#include <cstddef>
#include <cstdint>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"

#include "libspu/core/context.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {
namespace {

void RunCryptoMoEEqualMuxTest(FieldType field) {
  // CryptoMoE Figure 4:
  //
  // K = [0, 2, 1, 2]
  // W = [0.7, 0.2, 0.2, 0.6]
  // i = 2
  //
  // Pi_equal([[K]], 2)
  //   -> [[M_2]]^B = [0, 1, 0, 1]
  //
  // Pi_mux([[M_2]]^B, [[W]])
  //   -> [[S_2]] = [0, 0.2, 0, 0.6]

  const xt::xarray<int32_t> routing_indices = {
      0, 2, 1, 2,
  };

  // hal::equal requires equal-shaped operands.
  const xt::xarray<int32_t> expert_ids = {
      2, 2, 2, 2,
  };

  const xt::xarray<float> weights = {
      0.7F, 0.2F, 0.2F, 0.6F,
  };

  const xt::xarray<float> zeros = {
      0.0F, 0.0F, 0.0F, 0.0F,
  };

  const xt::xarray<bool> expected_mask = {
      false, true, false, true,
  };

  const xt::xarray<float> expected_scores = {
      0.0F, 0.2F, 0.0F, 0.6F,
  };

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig config;
        config.set_protocol(ProtocolKind::CHEETAH);
        config.set_field(field);

        auto ctx = test::makeSPUContext(config, lctx);

        // [[K]]
        auto k_p = constant(&ctx, routing_indices, DT_I32);
        auto k_s = _p2s(&ctx, k_p).setDtype(DT_I32);

        // Public expert ID i.
        auto i_p = constant(&ctx, expert_ids, DT_I32);

        // [[W]]
        auto weights_p = constant(&ctx, weights, DT_F32);
        auto weights_s = _p2s(&ctx, weights_p).setDtype(DT_F32);

        auto zero_p = constant(&ctx, zeros, DT_F32);

        ASSERT_TRUE(k_s.isSecret());
        ASSERT_TRUE(weights_s.isSecret());
        ASSERT_TRUE(i_p.isPublic());

        // CryptoMoE Algorithm 1, line 3:
        // [[M_i]]^B = Pi_equal([[K]], i).
        auto mask_s = equal(&ctx, k_s, i_p);

        ASSERT_TRUE(mask_s.isSecret());
        ASSERT_EQ(mask_s.dtype(), DT_I1);

        // IMPORTANT:
        // Do NOT reveal mask_s here.
        //
        // CryptoMoE Algorithm 1, line 4:
        // [[S_i]] = Pi_mux([[M_i]]^B, [[W]]).
        auto scores_s = select(&ctx, mask_s, weights_s, zero_p);

        ASSERT_TRUE(scores_s.isSecret());
        ASSERT_EQ(scores_s.dtype(), DT_F32);

        // Only reveal the final output of the chained protocol in the test.
        auto scores_p = _s2p(&ctx, scores_s).setDtype(DT_F32);
        auto got = dump_public_as<float>(&ctx, scores_p);

        // Verify the MUX preserves the fixed-point encoded routing score
        // exactly when the equality mask is 1, and outputs exactly zero
        // otherwise.
        auto quantized_weights = dump_public_as<float>(&ctx, weights_p);

        for (size_t idx = 0; idx < expected_mask.size(); ++idx) {
          EXPECT_FLOAT_EQ(
              got(idx),
              expected_mask(idx) ? quantized_weights(idx) : 0.0F);
        }

        // Also verify agreement with the paper-level floating-point values,
        // allowing at most one fixed-point LSB for encoding quantization.
        const double quantization_atol =
            1.0 / static_cast<double>(1ULL << ctx.getFxpBits());

        EXPECT_TRUE(
            xt::allclose(expected_scores, got, 0.0, quantization_atol))
            << "expected:\n"
            << expected_scores << "\ngot:\n"
            << got;
      });
}

TEST(CryptoMoEEqualMuxTest, FM32) {
  RunCryptoMoEEqualMuxTest(FieldType::FM32);
}

TEST(CryptoMoEEqualMuxTest, FM64) {
  RunCryptoMoEEqualMuxTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hal
