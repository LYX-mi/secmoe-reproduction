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

#include "libspu/kernel/hlo/cryptomoe_expert.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "libspu/core/context.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

namespace {

void RunCryptoMoEExpertTest(FieldType field) {
  if (field == FieldType::FM32) {
    GTEST_SKIP() << "upstream f_seg4_silu does not support FM32";
  }

  // n = 2 experts, capacity t = 2, model dimension d = 2,
  // intermediate dimension h = 2.
  const xt::xarray<float> expert0 = {
      {0.5F, -1.0F},
      {1.5F, 0.25F},
  };
  const xt::xarray<float> expert1 = {
      {-0.75F, 1.25F},
      {0.5F, 2.0F},
  };

  // Server-private SwiGLU weights [n, d, h], [n, d, h], [n, h, d].
  const xt::xarray<float> gate_weight = {
      {{1.0F, 0.5F}, {-0.5F, 1.0F}},
      {{0.75F, -1.0F}, {1.0F, 0.5F}},
  };
  const xt::xarray<float> up_weight = {
      {{1.0F, -0.25F}, {0.5F, 1.0F}},
      {{-0.5F, 1.0F}, {1.25F, 0.75F}},
  };
  const xt::xarray<float> down_weight = {
      {{1.0F, 0.5F}, {-0.75F, 1.0F}},
      {{0.5F, -1.0F}, {1.0F, 0.25F}},
  };

  // Plaintext oracle for:
  // y = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down.
  const xt::xarray<float> expected = {
      {{-0.20301973F, 0.27069297F},
       {1.85198532F, 0.80034197F}},
      {{0.64895636F, -0.83490203F},
       {3.06696032F, -4.73338715F}},
  };

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig config;
        config.set_protocol(ProtocolKind::CHEETAH);
        config.set_field(field);
        config.set_fxp_fraction_bits(16);
        config.set_experimental_enable_bmm(true);
        config.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
        config.mutable_cheetah_2pc_config()->set_approx_less_precision(4);

        SPUContext ctx = test::makeSPUContext(config, lctx);

        std::vector<spu::Value> expert_inputs;
        expert_inputs.push_back(
            test::makeValue(&ctx, expert0, VIS_SECRET));
        expert_inputs.push_back(
            test::makeValue(&ctx, expert1, VIS_SECRET));

        auto gate_p = hal::constant(&ctx, gate_weight, DT_F32);
        auto up_p = hal::constant(&ctx, up_weight, DT_F32);
        auto down_p = hal::constant(&ctx, down_weight, DT_F32);

        auto gate_v = hal::_p2v(&ctx, gate_p, 1).setDtype(DT_F32);
        auto up_v = hal::_p2v(&ctx, up_p, 1).setDtype(DT_F32);
        auto down_v = hal::_p2v(&ctx, down_p, 1).setDtype(DT_F32);

        ASSERT_TRUE(expert_inputs[0].isSecret());
        ASSERT_TRUE(expert_inputs[1].isSecret());
        ASSERT_TRUE(gate_v.isPrivate());
        ASSERT_TRUE(up_v.isPrivate());
        ASSERT_TRUE(down_v.isPrivate());

        auto outputs = CryptoMoEExpertCompute(
            &ctx, expert_inputs, gate_v, up_v, down_v);

        ASSERT_EQ(outputs.size(), 2U);

        const double atol = field == FieldType::FM32 ? 0.08 : 0.05;
        for (size_t expert = 0; expert < outputs.size(); ++expert) {
          ASSERT_TRUE(outputs[expert].isSecret());
          ASSERT_EQ(outputs[expert].dtype(), DT_F32);
          ASSERT_EQ(outputs[expert].shape(), Shape({2, 2}));

          auto output_p =
            hal::_s2p(&ctx, outputs[expert]).setDtype(DT_F32);
          auto got = hal::dump_public_as<float>(&ctx, output_p);

          for (size_t row = 0; row < 2; ++row) {
            for (size_t col = 0; col < 2; ++col) {
              EXPECT_NEAR(got(row, col), expected(expert, row, col), atol)
                  << "field=" << field << ", expert=" << expert
                  << ", row=" << row << ", col=" << col;
            }
          }
        }
      });
}

TEST(CryptoMoEExpertTest, FM32) {
  RunCryptoMoEExpertTest(FieldType::FM32);
}

TEST(CryptoMoEExpertTest, FM64) {
  RunCryptoMoEExpertTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hlo
