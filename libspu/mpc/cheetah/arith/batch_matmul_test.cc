// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/arith/batch_matmul.h"

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class BatchMatMulTest
    : public ::testing::TestWithParam<std::tuple<FieldType, Shape4D, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, BatchMatMulTest,
    testing::Combine(
        testing::Values(FieldType::FM32, FieldType::FM64),
        testing::Values(Shape4D{64, 32, 512, 512}, Shape4D{4, 1, 2048, 768},
                        Shape4D{4, 18, 768, 78}, Shape4D{4, 1024, 16, 16}),
        testing::Values(false)),
    [](const testing::TestParamInfo<BatchMatMulTest::ParamType>& p) {
      return fmt::format(
          "{}_{}x{}x{}x{}_{}", std::get<0>(p.param),
          std::get<0>(std::get<1>(p.param)), std::get<1>(std::get<1>(p.param)),
          std::get<2>(std::get<1>(p.param)), std::get<3>(std::get<1>(p.param)),
          std::get<2>(p.param) ? "Approx" : "Exact");
    });

TEST_P(BatchMatMulTest, Basic) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto dim4 = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());

  std::vector<NdArrayRef> mat(kWorldSize);

  // TODO: inputs should be multi-dim tensors.
  mat[0] = ring_rand(field, {dim4[0], dim4[1], dim4[2]});
  mat[1] = ring_rand(field, {dim4[0], dim4[2], dim4[3]});

  std::vector<NdArrayRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    auto matmul = std::make_shared<BatchMatMul>(lctx, allow_approx);
    matmul->LazyInitKeys(field);

    result[rank] = matmul->BatchDotOLE(mat[rank], lctx.get(), dim4, rank == 0);
  });

  auto computed = ring_add(result[0], result[1]);

  // compute expected result
  NdArrayRef expected;
  expected = ring_zeros(field, {dim4[0], dim4[1], dim4[3]});

  for (int64_t b = 0; b < dim4[0]; b++) {
    auto lhs = mat[0].slice({b, 0, 0}, {b + 1, dim4[1], dim4[2]}, {1, 1, 1})
            .reshape({dim4[1], dim4[2]});
    auto rhs = mat[1].slice({b, 0, 0}, {b + 1, dim4[2], dim4[3]}, {1, 1, 1})
                   .reshape({dim4[2], dim4[3]});
    auto slice = expected.slice({b, 0, 0}, {b + 1, dim4[1], dim4[3]}, {1, 1, 1})
                     .reshape({dim4[1], dim4[3]});
    ring_mmul_(slice, lhs, rhs);
  }

  EXPECT_EQ(expected.numel(), computed.numel());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto e = NdArrayView<ring2k_t>(expected);
    auto c = NdArrayView<ring2k_t>(computed);

    for (auto idx = 0; idx < expected.numel(); idx++) {
      // only exact version supported now.
      SPU_ENFORCE(e[idx] == c[idx], "expected {}, got {}, at {}", e[idx],
                  c[idx], idx);
    }
  });
}

}  // namespace spu::mpc::cheetah

namespace spu::mpc::cheetah {

TEST(BatchMatMulTestStandalone, SequentialDifferentShapes) {
  constexpr size_t kWorldSize = 2;
  const auto field = FieldType::FM64;

  const Shape4D dim_gate{4, 4, 64, 128};
  const Shape4D dim_down{4, 4, 128, 64};

  std::vector<NdArrayRef> gate_mat(kWorldSize);
  gate_mat[0] =
      ring_rand(field, {dim_gate[0], dim_gate[1], dim_gate[2]});
  gate_mat[1] =
      ring_rand(field, {dim_gate[0], dim_gate[2], dim_gate[3]});

  std::vector<NdArrayRef> down_mat(kWorldSize);
  down_mat[0] =
      ring_rand(field, {dim_down[0], dim_down[1], dim_down[2]});
  down_mat[1] =
      ring_rand(field, {dim_down[0], dim_down[2], dim_down[3]});

  std::vector<NdArrayRef> gate_result(kWorldSize);
  std::vector<NdArrayRef> down_result(kWorldSize);

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    lctx->SetRecvTimeout(10 * 1000);

    const int rank = lctx->Rank();
    auto matmul = std::make_shared<BatchMatMul>(lctx, false);
    matmul->LazyInitKeys(field);

    gate_result[rank] =
        matmul->BatchDotOLE(gate_mat[rank], lctx.get(), dim_gate, rank == 0);

    down_result[rank] =
        matmul->BatchDotOLE(down_mat[rank], lctx.get(), dim_down, rank == 0);
  });

  auto gate_computed = ring_add(gate_result[0], gate_result[1]);
  auto down_computed = ring_add(down_result[0], down_result[1]);

  auto compute_expected =
      [&](const std::vector<NdArrayRef>& mat, const Shape4D& dim4) {
        auto expected =
            ring_zeros(field, {dim4[0], dim4[1], dim4[3]});
        for (int64_t b = 0; b < dim4[0]; ++b) {
          auto lhs =
              mat[0]
                  .slice({b, 0, 0}, {b + 1, dim4[1], dim4[2]}, {1, 1, 1})
                  .reshape({dim4[1], dim4[2]});
          auto rhs =
              mat[1]
                  .slice({b, 0, 0}, {b + 1, dim4[2], dim4[3]}, {1, 1, 1})
                  .reshape({dim4[2], dim4[3]});
          auto slice =
              expected
                  .slice({b, 0, 0}, {b + 1, dim4[1], dim4[3]}, {1, 1, 1})
                  .reshape({dim4[1], dim4[3]});
          ring_mmul_(slice, lhs, rhs);
        }
        return expected;
      };

  auto gate_expected = compute_expected(gate_mat, dim_gate);
  auto down_expected = compute_expected(down_mat, dim_down);

  EXPECT_EQ(gate_expected.numel(), gate_computed.numel());
  EXPECT_EQ(down_expected.numel(), down_computed.numel());

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto ge = NdArrayView<ring2k_t>(gate_expected);
    auto gc = NdArrayView<ring2k_t>(gate_computed);
    for (int64_t idx = 0; idx < gate_expected.numel(); ++idx) {
      SPU_ENFORCE(ge[idx] == gc[idx],
                  "gate expected {}, got {}, at {}", ge[idx], gc[idx], idx);
    }

    auto de = NdArrayView<ring2k_t>(down_expected);
    auto dc = NdArrayView<ring2k_t>(down_computed);
    for (int64_t idx = 0; idx < down_expected.numel(); ++idx) {
      SPU_ENFORCE(de[idx] == dc[idx],
                  "down expected {}, got {}, at {}", de[idx], dc[idx], idx);
    }
  });
}

}  // namespace spu::mpc::cheetah
