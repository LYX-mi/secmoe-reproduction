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

TEST(BatchMatMulTest, SecretPrivateCheetahBMM) {
  // X: [B=2, M=2, K=3]
  const xt::xarray<int64_t> x = {
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 0, 2}, {3, 2, 1}},
  };

  // W: [B=2, K=3, N=2]
  const xt::xarray<int64_t> w = {
      {{1, 2}, {3, 4}, {5, 6}},
      {{2, 1}, {0, 3}, {4, 2}},
  };

  // X @ W: [B=2, M=2, N=2]
  const xt::xarray<int64_t> expected = {
      {{22, 28}, {49, 64}},
      {{10, 5}, {10, 11}},
  };

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig config;
        config.set_protocol(ProtocolKind::CHEETAH);
        config.set_field(FieldType::FM64);
        config.set_experimental_enable_bmm(true);

        auto ctx = test::makeSPUContext(config, lctx);

        auto x_p = constant(&ctx, x, DT_I64);
        auto w_p = constant(&ctx, w, DT_I64);

        auto x_s = _p2s(&ctx, x_p).setDtype(DT_I64);
        auto w_v = _p2v(&ctx, w_p, 1).setDtype(DT_I64);

        EXPECT_TRUE(x_s.isSecret());
        EXPECT_TRUE(w_v.isPrivate());

        auto out = batch_matmul(&ctx, x_s, w_v);
        ASSERT_TRUE(out.has_value());

        auto out_p = _s2p(&ctx, *out).setDtype(DT_I64);
        auto got = dump_public_as<int64_t>(&ctx, out_p);

        EXPECT_TRUE(xt::allclose(expected, got, 0.0, 0.0))
            << "expected:\n"
            << expected << "\ngot:\n"
            << got;
      });
}

}  // namespace
}  // namespace spu::kernel::hal
