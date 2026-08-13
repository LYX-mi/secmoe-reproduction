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

void RunCryptoMoEEqualTest(FieldType field) {
  // Flattened routing expert indices [[K]].
  const xt::xarray<int32_t> routing_indices = {0, 2, 1, 2, 3, 0};

  // CryptoMoE Algorithm 1 loops over a public expert id i.
  // hal::equal requires equal-shaped operands, so explicitly broadcast i=2.
  const xt::xarray<int32_t> expert_ids = {2, 2, 2, 2, 2, 2};

  // M_i[j] = 1{K[j] == i}.
  const xt::xarray<bool> expected_mask = {
      false, true, false, true, false, false,
  };

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig config;
        config.set_protocol(ProtocolKind::CHEETAH);
        config.set_field(field);

        auto ctx = test::makeSPUContext(config, lctx);

        auto k_p = constant(&ctx, routing_indices, DT_I32);
        auto i_p = constant(&ctx, expert_ids, DT_I32);

        auto k_s = _p2s(&ctx, k_p).setDtype(DT_I32);

        ASSERT_TRUE(k_s.isSecret());
        ASSERT_TRUE(i_p.isPublic());

        // CryptoMoE Algorithm 1, line 3:
        // [[M_i]]^B = Pi_equal([[K]], i).
        auto mask_s = equal(&ctx, k_s, i_p);

        EXPECT_TRUE(mask_s.isSecret());
        EXPECT_EQ(mask_s.dtype(), DT_I1);

        auto mask_p = _s2p(&ctx, mask_s).setDtype(DT_I1);
        auto got = dump_public_as<bool>(&ctx, mask_p);

        EXPECT_TRUE(xt::all(xt::equal(got, expected_mask)))
            << "expected:\n"
            << expected_mask << "\ngot:\n"
            << got;
      });
}

TEST(CryptoMoEEqualTest, FM32) {
  RunCryptoMoEEqualTest(FieldType::FM32);
}

TEST(CryptoMoEEqualTest, FM64) {
  RunCryptoMoEEqualTest(FieldType::FM64);
}

}  // namespace
}  // namespace spu::kernel::hal
