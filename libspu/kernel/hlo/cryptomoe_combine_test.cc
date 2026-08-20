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

void RunCryptoMoECombineTest(FieldType field) {
  // CryptoMoE Figure 5 style combine with m = 2, t = 1, d = 2.
  //
  // Expert 0 contributes score 0.7 to token A.
  // Expert 1 contributes score 0.2 to token B.
  // Expert 2 contributes score 0.6 to token B.
  // Expert 3 is a dummy selection with score 0 and must contribute nothing.
  const xt::xarray<float> expert0 = {{10.0F, 20.0F}};
  const xt::xarray<float> expert1 = {{30.0F, 40.0F}};
  const xt::xarray<float> expert2 = {{50.0F, 60.0F}};
  const xt::xarray<float> expert3 = {{100.0F, 200.0F}};

  const xt::xarray<bool> onehot0 = {{true, false}};
  const xt::xarray<bool> onehot1 = {{false, true}};
  const xt::xarray<bool> onehot2 = {{false, true}};
  const xt::xarray<bool> onehot3 = {{true, false}};

  const xt::xarray<float> score0 = {0.7F};
  const xt::xarray<float> score1 = {0.2F};
  const xt::xarray<float> score2 = {0.6F};
  const xt::xarray<float> score3 = {0.0F};

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx =
            test::makeSPUContext(ProtocolKind::CHEETAH, field, lctx);

        std::vector<spu::Value> expert_outputs_s = {
            test::makeValue(&ctx, expert0, VIS_SECRET),
            test::makeValue(&ctx, expert1, VIS_SECRET),
            test::makeValue(&ctx, expert2, VIS_SECRET),
            test::makeValue(&ctx, expert3, VIS_SECRET),
        };
        std::vector<spu::Value> onehots_s = {
            test::makeValue(&ctx, onehot0, VIS_SECRET),
            test::makeValue(&ctx, onehot1, VIS_SECRET),
            test::makeValue(&ctx, onehot2, VIS_SECRET),
            test::makeValue(&ctx, onehot3, VIS_SECRET),
        };
        std::vector<spu::Value> scores_s = {
            test::makeValue(&ctx, score0, VIS_SECRET),
            test::makeValue(&ctx, score1, VIS_SECRET),
            test::makeValue(&ctx, score2, VIS_SECRET),
            test::makeValue(&ctx, score3, VIS_SECRET),
        };

        for (const auto& onehot : onehots_s) {
          ASSERT_TRUE(onehot.isSecret());
          ASSERT_EQ(onehot.dtype(), DT_I1);
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
            {36.0F, 44.0F},
        };
        EXPECT_TRUE(xt::allclose(expected, got, 0.01, 0.001));
      });
}

TEST(CryptoMoECombineTest, FM32) {
  RunCryptoMoECombineTest(FieldType::FM32);
}

TEST(CryptoMoECombineTest, FM64) {
  RunCryptoMoECombineTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hlo
