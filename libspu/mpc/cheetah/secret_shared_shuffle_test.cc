// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/secret_shared_shuffle.h"

#include <array>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::secret_shared_shuffle {

namespace {

void RunOpvCorrectness(int64_t full_rank, int64_t puncture_index) {
  constexpr size_t kWorldSize = 2;
  constexpr int64_t kN = 8;

  std::array<std::vector<uint128_t>, 2> output;

  utils::simulate(
      kWorldSize,
      [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        auto comm = std::make_shared<Communicator>(lctx);

        // Use the same Ferret OT family used by Cheetah.
        auto ot = std::make_shared<BasicOTProtocols>(
            comm, CheetahOtKind::YACL_Ferret);

        // The full-vector party must not receive the private puncture index.
        const int64_t local_index =
            static_cast<int64_t>(lctx->Rank()) == full_rank
                ? -1
                : puncture_index;

        output[lctx->Rank()] =
            ObliviousPuncturedVector(
                comm,
                ot, full_rank, kN, local_index);
      });

  const int64_t punctured_rank = 1 - full_rank;

  ASSERT_EQ(output[full_rank].size(), static_cast<size_t>(kN));
  ASSERT_EQ(output[punctured_rank].size(), static_cast<size_t>(kN));

  // [14] OPV correctness:
  // v_full[j] == v_punctured[j] for every j != puncture_index.
  for (int64_t j = 0; j < kN; ++j) {
    if (j == puncture_index) {
      continue;
    }

    EXPECT_EQ(
        output[full_rank][j],
        output[punctured_rank][j])
        << "OPV mismatch at non-punctured position " << j;
  }

  // Our local representation for the unavailable leaf.
  EXPECT_EQ(output[punctured_rank][puncture_index],
            static_cast<uint128_t>(0));
}

}  // namespace

TEST(SecretSharedShuffleTest, OpvFullRank0) {
  RunOpvCorrectness(/*full_rank=*/0, /*puncture_index=*/5);
}

TEST(SecretSharedShuffleTest, OpvFullRank1) {
  RunOpvCorrectness(/*full_rank=*/1, /*puncture_index=*/2);
}


TEST(SecretSharedShuffleTest, BatchedOpvCorrectness) {
  constexpr size_t kWorldSize = 2;
  constexpr int64_t kN = 8;
  constexpr int64_t kInstances = 4;
  constexpr int64_t kFullRank = 0;

  const std::array<int64_t, kInstances> punctures = {
      0, 3, 5, 7,
  };

  std::array<std::vector<std::vector<uint128_t>>, 2> output;

  utils::simulate(
      kWorldSize,
      [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        auto comm = std::make_shared<Communicator>(lctx);
        auto ot = std::make_shared<BasicOTProtocols>(
            comm, CheetahOtKind::YACL_Ferret);

        if (lctx->Rank() == kFullRank) {
          // Position-hiding boundary: full party gets no indices.
          output[lctx->Rank()] =
              ObliviousPuncturedVectors(
                  comm,
                  ot, kFullRank, kN, kInstances,
                  absl::Span<const int64_t>());
        } else {
          output[lctx->Rank()] =
              ObliviousPuncturedVectors(
                  comm,
                  ot, kFullRank, kN, kInstances,
                  absl::MakeConstSpan(punctures));
        }
      });

  const int64_t punctured_rank = 1 - kFullRank;

  ASSERT_EQ(
      output[kFullRank].size(),
      static_cast<size_t>(kInstances));
  ASSERT_EQ(
      output[punctured_rank].size(),
      static_cast<size_t>(kInstances));

  for (int64_t instance = 0; instance < kInstances; ++instance) {
    ASSERT_EQ(
        output[kFullRank][instance].size(),
        static_cast<size_t>(kN));
    ASSERT_EQ(
        output[punctured_rank][instance].size(),
        static_cast<size_t>(kN));

    for (int64_t j = 0; j < kN; ++j) {
      if (j == punctures[instance]) {
        EXPECT_EQ(
            output[punctured_rank][instance][j],
            static_cast<uint128_t>(0));
        continue;
      }

      EXPECT_EQ(
          output[kFullRank][instance][j],
          output[punctured_rank][instance][j])
          << "instance=" << instance << ", position=" << j;
    }
  }
}

namespace {

void RunShareTranslationCorrectness(
    int64_t permutation_owner, int bit_width) {
  constexpr size_t kWorldSize = 2;
  constexpr int64_t kN = 4;

  // Paper convention:
  //
  //   pi(x)[i] = x[pi(i)]
  //
  // This is a non-trivial permutation with no repeated positions.
  const std::array<int64_t, kN> permutation = {
      2, 0, 3, 1,
  };

  std::array<ShareTranslationOutput, 2> output;

  utils::simulate(
      kWorldSize,
      [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        auto comm = std::make_shared<Communicator>(lctx);

        auto ot = std::make_shared<BasicOTProtocols>(
            comm, CheetahOtKind::YACL_Ferret);

        const int64_t rank =
            static_cast<int64_t>(lctx->Rank());

        if (rank == permutation_owner) {
          output[rank] = ShareTranslation(
              comm,
              ot,
              permutation_owner,
              kN,
              absl::MakeConstSpan(permutation),
              bit_width);
        } else {
          // The full-matrix party gets no permutation.
          output[rank] = ShareTranslation(
              comm,
              ot,
              permutation_owner,
              kN,
              absl::Span<const int64_t>(),
              bit_width);
        }
      });

  const int64_t mask_owner = 1 - permutation_owner;

  ASSERT_EQ(
      output[permutation_owner].delta.size(),
      static_cast<size_t>(kN));
  EXPECT_TRUE(output[permutation_owner].a.empty());
  EXPECT_TRUE(output[permutation_owner].b.empty());

  EXPECT_TRUE(output[mask_owner].delta.empty());
  ASSERT_EQ(
      output[mask_owner].a.size(),
      static_cast<size_t>(kN));
  ASSERT_EQ(
      output[mask_owner].b.size(),
      static_cast<size_t>(kN));

  const uint128_t mask =
      bit_width == 128
          ? ~static_cast<uint128_t>(0)
          : (static_cast<uint128_t>(1) << bit_width) - 1;

  // [14] Share Translation correctness:
  //
  //   delta = b - pi(a)
  //
  // i.e.
  //
  //   delta[i] = b[i] - a[pi(i)].
  for (int64_t i = 0; i < kN; ++i) {
    const uint128_t expected =
        (output[mask_owner].b[static_cast<size_t>(i)] -
         output[mask_owner].a[
             static_cast<size_t>(permutation[i])]) &
        mask;

    EXPECT_TRUE(
        output[permutation_owner]
            .delta[static_cast<size_t>(i)] == expected)
        << "Share Translation mismatch at i=" << i;
  }
}

}  // namespace

TEST(SecretSharedShuffleTest, ShareTranslationOwner0FM32) {
  RunShareTranslationCorrectness(
      /*permutation_owner=*/0,
      /*bit_width=*/32);
}

TEST(SecretSharedShuffleTest, ShareTranslationOwner1FM64) {
  RunShareTranslationCorrectness(
      /*permutation_owner=*/1,
      /*bit_width=*/64);
}

TEST(SecretSharedShuffleTest, ShareTranslationOwner0FM128) {
  RunShareTranslationCorrectness(
      /*permutation_owner=*/0,
      /*bit_width=*/128);
}


namespace {

std::vector<int64_t> ExpandBenesLayerForTest(
    const BenesSubPermutationLayer& layer,
    int64_t N) {
  std::vector<int64_t> out(
      static_cast<size_t>(N), -1);

  for (const auto& subperm : layer) {
    if (subperm.positions.size() !=
        subperm.permutation.size()) {
      ADD_FAILURE()
          << "positions/permutation size mismatch";
      return {};
    }

    for (size_t local_output = 0;
         local_output < subperm.positions.size();
         ++local_output) {
      const int64_t global_output =
          subperm.positions[local_output];

      const int64_t local_input =
          subperm.permutation[local_output];

      if (local_input < 0 ||
          static_cast<size_t>(local_input) >=
              subperm.positions.size()) {
        ADD_FAILURE()
            << "invalid local_input=" << local_input;
        return {};
      }

      out[static_cast<size_t>(global_output)] =
          subperm.positions[
              static_cast<size_t>(local_input)];
    }
  }

  for (int64_t i = 0; i < N; ++i) {
    EXPECT_GE(out[static_cast<size_t>(i)], 0);
  }

  return out;
}

std::vector<int64_t> ComposePermutationForTest(
    const std::vector<int64_t>& lhs,
    const std::vector<int64_t>& rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());

  std::vector<int64_t> out(lhs.size());

  for (size_t i = 0; i < lhs.size(); ++i) {
    out[i] = lhs[static_cast<size_t>(rhs[i])];
  }

  return out;
}

void CheckBenesDecomposition(
    const std::vector<int64_t>& permutation,
    int64_t T,
    int64_t expected_d) {
  const int64_t N =
      static_cast<int64_t>(permutation.size());

  const auto layers =
      BenesDecomposePermutation(
          absl::MakeConstSpan(permutation), T);

  ASSERT_EQ(
      static_cast<int64_t>(layers.size()),
      expected_d);

  std::vector<int64_t> composed(
      static_cast<size_t>(N));

  std::iota(composed.begin(), composed.end(), 0);

  for (const auto& layer : layers) {
    ASSERT_EQ(
        static_cast<int64_t>(layer.size()),
        N / T);

    for (const auto& subperm : layer) {
      EXPECT_EQ(
          static_cast<int64_t>(subperm.positions.size()),
          T);

      EXPECT_EQ(
          static_cast<int64_t>(subperm.permutation.size()),
          T);
    }

    const auto expanded =
        ExpandBenesLayerForTest(layer, N);

    composed =
        ComposePermutationForTest(
            composed, expanded);
  }

  EXPECT_EQ(composed, permutation);
}

}  // namespace

TEST(SecretSharedShuffleTest, BenesDecompositionN8T4) {
  // log2(N)=3, log2(T)=2:
  // d = 2*ceil(3/2)-1 = 3.
  CheckBenesDecomposition(
      {5, 2, 7, 0, 3, 6, 1, 4},
      /*T=*/4,
      /*expected_d=*/3);
}

TEST(SecretSharedShuffleTest, BenesDecompositionN16T4) {
  // log2(N)=4, log2(T)=2:
  // d = 3.
  CheckBenesDecomposition(
      {9, 0, 14, 3, 7, 12, 5, 10,
       1, 15, 4, 8, 13, 2, 11, 6},
      /*T=*/4,
      /*expected_d=*/3);
}

TEST(SecretSharedShuffleTest, BenesDecompositionNonDivisibleN32T8) {
  // Important non-divisible case:
  // log2(N)=5, log2(T)=3:
  // d = 2*ceil(5/3)-1 = 3.
  CheckBenesDecomposition(
      {17, 4, 29, 1, 22, 13, 8, 31,
       6, 25, 10, 19, 0, 27, 15, 3,
       30, 11, 21, 7, 26, 14, 2, 18,
       9, 28, 5, 23, 16, 12, 24, 20},
      /*T=*/8,
      /*expected_d=*/3);
}


namespace {

void RunBatchedShareTranslationCorrectness(
    int64_t permutation_owner,
    int bit_width) {
  constexpr size_t kWorldSize = 2;
  constexpr int64_t kN = 4;
  constexpr int64_t kNumPermutations = 3;

  const std::array<int64_t,
                   kN * kNumPermutations>
      permutations = {
          // pi_0
          2, 0, 3, 1,

          // pi_1
          1, 3, 0, 2,

          // pi_2
          3, 2, 1, 0,
      };

  std::array<BatchedShareTranslationOutput, 2>
      output;

  utils::simulate(
      kWorldSize,
      [&](const std::shared_ptr<
          yacl::link::Context>& lctx) {
        auto comm =
            std::make_shared<Communicator>(lctx);

        auto ot =
            std::make_shared<BasicOTProtocols>(
                comm,
                CheetahOtKind::YACL_Ferret);

        const int64_t rank =
            static_cast<int64_t>(
                lctx->Rank());

        if (rank == permutation_owner) {
          output[static_cast<size_t>(rank)] =
              ShareTranslations(
                  comm,
                  ot,
                  permutation_owner,
                  kN,
                  kNumPermutations,
                  absl::MakeConstSpan(
                      permutations),
                  bit_width);
        } else {
          output[static_cast<size_t>(rank)] =
              ShareTranslations(
                  comm,
                  ot,
                  permutation_owner,
                  kN,
                  kNumPermutations,
                  absl::Span<const int64_t>(),
                  bit_width);
        }
      });

  const int64_t masks_owner =
      1 - permutation_owner;

  const auto& owner_output =
      output[
          static_cast<size_t>(
              permutation_owner)];

  const auto& mask_output =
      output[
          static_cast<size_t>(
              masks_owner)];

  ASSERT_EQ(
      owner_output.delta.size(),
      static_cast<size_t>(
          kNumPermutations));

  ASSERT_EQ(
      mask_output.a.size(),
      static_cast<size_t>(
          kNumPermutations));

  ASSERT_EQ(
      mask_output.b.size(),
      static_cast<size_t>(
          kNumPermutations));

  EXPECT_TRUE(owner_output.a.empty());
  EXPECT_TRUE(owner_output.b.empty());
  EXPECT_TRUE(mask_output.delta.empty());

  const uint128_t mask =
      bit_width == 128
          ? ~static_cast<uint128_t>(0)
          : (static_cast<uint128_t>(1)
             << bit_width) -
                1;

  for (int64_t instance = 0;
       instance < kNumPermutations;
       ++instance) {
    ASSERT_EQ(
        owner_output
            .delta[
                static_cast<size_t>(
                    instance)]
            .size(),
        static_cast<size_t>(kN));

    for (int64_t i = 0; i < kN; ++i) {
      const int64_t pi_i =
          permutations[
              static_cast<size_t>(
                  instance * kN + i)];

      const uint128_t expected =
          (mask_output
               .b[
                   static_cast<size_t>(
                       instance)]
               [static_cast<size_t>(i)] -
           mask_output
               .a[
                   static_cast<size_t>(
                       instance)]
               [static_cast<size_t>(
                   pi_i)]) &
          mask;

      EXPECT_EQ(
          owner_output
              .delta[
                  static_cast<size_t>(
                      instance)]
              [static_cast<size_t>(i)],
          expected)
          << "instance=" << instance
          << ", i=" << i;
    }
  }
}

}  // namespace

TEST(SecretSharedShuffleTest,
     BatchedShareTranslationOwner0FM32) {
  RunBatchedShareTranslationCorrectness(
      /*permutation_owner=*/0,
      /*bit_width=*/32);
}

TEST(SecretSharedShuffleTest,
     BatchedShareTranslationOwner1FM64) {
  RunBatchedShareTranslationCorrectness(
      /*permutation_owner=*/1,
      /*bit_width=*/64);
}


namespace {

void RunPermuteAndShareCorrectness(
    int64_t permutation_owner,
    int bit_width) {
  constexpr size_t kWorldSize = 2;
  constexpr int64_t kN = 8;
  constexpr int64_t kT = 4;

  const std::array<int64_t, kN> permutation = {
      5, 2, 7, 0, 3, 6, 1, 4,
  };

  const std::array<uint128_t, kN> x = {
      11, 22, 33, 44, 55, 66, 77, 88,
  };

  std::array<std::vector<uint128_t>, 2> shares;

  utils::simulate(
      kWorldSize,
      [&](const std::shared_ptr<
          yacl::link::Context>& lctx) {
        auto comm =
            std::make_shared<Communicator>(lctx);

        auto ot =
            std::make_shared<BasicOTProtocols>(
                comm,
                CheetahOtKind::YACL_Ferret);

        const int64_t rank =
            static_cast<int64_t>(
                lctx->Rank());

        if (rank == permutation_owner) {
          shares[static_cast<size_t>(rank)] =
              PermuteAndShare(
                  comm,
                  ot,
                  permutation_owner,
                  kN,
                  kT,
                  absl::MakeConstSpan(permutation),
                  absl::Span<const uint128_t>(),
                  bit_width);
        } else {
          shares[static_cast<size_t>(rank)] =
              PermuteAndShare(
                  comm,
                  ot,
                  permutation_owner,
                  kN,
                  kT,
                  absl::Span<const int64_t>(),
                  absl::MakeConstSpan(x),
                  bit_width);
        }
      });

  const uint128_t mask =
      bit_width == 128
          ? ~static_cast<uint128_t>(0)
          : (static_cast<uint128_t>(1)
             << bit_width) - 1;

  ASSERT_EQ(
      shares[0].size(),
      static_cast<size_t>(kN));
  ASSERT_EQ(
      shares[1].size(),
      static_cast<size_t>(kN));

  // FPermute+Share correctness:
  //
  //   share_0 + share_1 = pi(x)
  for (int64_t i = 0; i < kN; ++i) {
    const uint128_t reconstructed =
        (shares[0][static_cast<size_t>(i)] +
         shares[1][static_cast<size_t>(i)]) &
        mask;

    const uint128_t expected =
        x[static_cast<size_t>(
            permutation[static_cast<size_t>(i)])] &
        mask;

    EXPECT_TRUE(reconstructed == expected)
        << "Permute+Share mismatch at i=" << i;
  }
}

}  // namespace

TEST(SecretSharedShuffleTest, PermuteAndShareOwner0FM32) {
  RunPermuteAndShareCorrectness(
      /*permutation_owner=*/0,
      /*bit_width=*/32);
}

TEST(SecretSharedShuffleTest, PermuteAndShareOwner1FM64) {
  RunPermuteAndShareCorrectness(
      /*permutation_owner=*/1,
      /*bit_width=*/64);
}


}  // namespace spu::mpc::cheetah::secret_shared_shuffle
