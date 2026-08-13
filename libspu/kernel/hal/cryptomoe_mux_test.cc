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

void RunCryptoMoEMuxTest(FieldType field) {
  // CryptoMoE Figure 4, expert 2:
  //
  // flattened W = [0.7, 0.2, 0.2, 0.6]
  // mask M_2    = [0,   1,   0,   1]
  //
  // Pi_mux(M_2, W) should produce:
  // S_2         = [0, 0.2,   0, 0.6]
  const xt::xarray<bool> mask = {
      false, true, false, true,
  };

  const xt::xarray<float> weights = {
      0.7F, 0.2F, 0.2F, 0.6F,
  };

  const xt::xarray<float> zeros = {
      0.0F, 0.0F, 0.0F, 0.0F,
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

        // Boolean public mask -> Boolean secret share.
        auto mask_p = constant(&ctx, mask, DT_I1);
        auto mask_s = _p2s(&ctx, mask_p).setDtype(DT_I1);

        // Routing scores [[W]] are arithmetic/fixed-point secret shares.
        auto weights_p = constant(&ctx, weights, DT_F32);
        auto weights_s = _p2s(&ctx, weights_p).setDtype(DT_F32);

        // Public zero is sufficient for:
        // select(mask, W, 0) = mask * W.
        auto zero_p = constant(&ctx, zeros, DT_F32);

        ASSERT_TRUE(mask_s.isSecret());
        ASSERT_EQ(mask_s.dtype(), DT_I1);
        ASSERT_TRUE(weights_s.isSecret());
        ASSERT_EQ(weights_s.dtype(), DT_F32);

        // CryptoMoE:
        // [[S_i]] = Pi_mux([[M_i]]^B, [[W]]).
        //
        // OpenBumbleBee:
        // select(pred, a, b) -> _mux -> b + pred * (a - b).
        //
        // With b = 0:
        // S_i = M_i * W.
        auto scores_s = select(&ctx, mask_s, weights_s, zero_p);

        EXPECT_TRUE(scores_s.isSecret());
        EXPECT_EQ(scores_s.dtype(), DT_F32);

        auto scores_p = _s2p(&ctx, scores_s).setDtype(DT_F32);
        auto got = dump_public_as<float>(&ctx, scores_p);

        // First verify the MUX itself exactly:
        // mask=1 must preserve the already encoded fixed-point weight,
        // while mask=0 must produce exactly zero.
        auto quantized_weights = dump_public_as<float>(&ctx, weights_p);
        for (size_t idx = 0; idx < mask.size(); ++idx) {
          EXPECT_FLOAT_EQ(
              got(idx), mask(idx) ? quantized_weights(idx) : 0.0F);
        }

        // Then verify the decoded value against the paper-level floating
        // point values. FM32 uses 8 fractional bits by default, so its
        // quantization error can be as large as one fixed-point LSB.
        const double quantization_atol =
            1.0 / static_cast<double>(1ULL << ctx.getFxpBits());

        EXPECT_TRUE(
            xt::allclose(expected_scores, got, 0.0, quantization_atol))
            << "expected:\n"
            << expected_scores << "\ngot:\n"
            << got;
      });
}

TEST(CryptoMoEMuxTest, FM32) {
  RunCryptoMoEMuxTest(FieldType::FM32);
}

TEST(CryptoMoEMuxTest, FM64) {
  RunCryptoMoEMuxTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hal
