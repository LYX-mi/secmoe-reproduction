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

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "yacl/base/int128.h"

namespace spu::mpc {
class Communicator;
}

namespace spu::mpc::cheetah {

class BasicOTProtocols;

namespace secret_shared_shuffle {

// Chase-Ghosh-Poburinnaya [ASIACRYPT 2020], Appendix B:
//
//   H(x) = pi(x) XOR x,
//   G(s) = H(s XOR 1) || H(s XOR 2),
//
// where pi is instantiated with a common AES key shared by both parties.
// Oblivious Punctured Vector from [14], Appendix A.
//
// `full_rank` owns the GGM root and obtains the full pseudorandom vector.
// The other party owns `puncture_index` and obtains every vector element
// except that position. The punctured slot is represented locally as zero
// and MUST NOT be consumed by callers.
//
// n must be a power of two. This is intentional: [14] applies OPV to
// power-of-two T-element subpermutations in its final shuffle protocol.
// Execute many independent OPVs in one OT batch.
//
// The full-vector party must pass an empty puncture_indices span.
// Only the punctured/index-owning party supplies puncture_indices.
// This keeps the OPV position private at the API boundary.
//
// All num_instances * log2(n) RCOTs are generated in one batch.
// Desired sibling-path choices are converted from random-choice RCOT
// using CipherGPT's masked-choice conversion.
std::vector<std::vector<uint128_t>> ObliviousPuncturedVectors(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot, int64_t full_rank,
    int64_t n, int64_t num_instances,
    absl::Span<const int64_t> puncture_indices);

// Convenience wrapper for one OPV.
//
// On the full-vector party puncture_index must be -1.
// On the punctured party it is the private puncture position.
std::vector<uint128_t> ObliviousPuncturedVector(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot, int64_t full_rank,
    int64_t n, int64_t puncture_index);

// Output of Chase-Ghosh-Poburinnaya Share Translation.
//
// If this party owns the permutation:
//   delta is populated; a and b are empty.
//
// Otherwise:
//   a and b are populated; delta is empty.
//
// All elements are represented canonically in Z_{2^bit_width}.
// One T-element permutation inside a (T,d)-subpermutation layer.
//
// positions[local] maps local wire -> global wire.
// permutation[local_output] gives the local input wire, following
// the paper convention:
//
//   pi(x)[i] = x[pi(i)].
struct BenesSubPermutation {
  std::vector<int64_t> positions;
  std::vector<int64_t> permutation;
};

using BenesSubPermutationLayer =
    std::vector<BenesSubPermutation>;

// [14], Section 5.
//
// Decompose an N-element permutation into d layers where every layer
// consists of N/T disjoint permutations acting on T elements:
//
//   permutation = pi_1 o ... o pi_d
//
// with
//
//   d = 2 * ceil(log2(N) / log2(T)) - 1.
//
// N and T must be powers of two and 2 <= T <= N.
std::vector<BenesSubPermutationLayer> BenesDecomposePermutation(
    absl::Span<const int64_t> permutation, int64_t T);


struct ShareTranslationOutput {
  std::vector<uint128_t> delta;
  std::vector<uint128_t> a;
  std::vector<uint128_t> b;
};

// [14], Section 4.2: Share Translation.
//
// The permutation owner is the punctured party in the underlying OPVs.
// The other party obtains the complete OPV matrix.
//
// permutation[i] follows the paper's convention:
//   pi(x)[i] = x[permutation[i]].
//
// `n` is the domain size of this Share Translation instance.
// In the final optimized Permute+Share this will be T, the
// power-of-two subpermutation size.
//
// On permutation_owner:
//   permutation.size() must equal n.
//
// On the other party:
//   permutation must be empty, so the private permutation is not exposed.
ShareTranslationOutput ShareTranslation(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot,
    int64_t permutation_owner,
    int64_t n,
    absl::Span<const int64_t> permutation,
    int bit_width);


struct BatchedShareTranslationOutput {
  std::vector<std::vector<uint128_t>> delta;
  std::vector<std::vector<uint128_t>> a;
  std::vector<std::vector<uint128_t>> b;
};

// Run many independent ShareTrans_n instances in one OPV/OT batch.
//
// On permutation_owner:
//   permutations_flat contains num_permutations consecutive permutations,
//   each of length n.
//
// On the other party:
//   permutations_flat must be empty.
//
// This is the form required by [14] Section 6.2, where all d*N/T
// ShareTrans_T instances from all subpermutation layers run in parallel.
BatchedShareTranslationOutput ShareTranslations(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot,
    int64_t permutation_owner,
    int64_t n,
    int64_t num_permutations,
    absl::Span<const int64_t> permutations_flat,
    int bit_width);


// [14], Section 6.2: Permute+Share.
//
// Functionality:
//
//   FPermute+Share(pi, x) = (r, pi(x) - r).
//
// permutation_owner owns `permutation` and must pass empty x.
// The other party owns `x` and must pass empty permutation.
//
// permutation follows the paper convention:
//
//   pi(x)[i] = x[permutation[i]].
//
// N and T are public. Currently N and T must be powers of two;
// arbitrary-length FShuffle padding is handled at the higher layer.
std::vector<uint128_t> PermuteAndShare(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot,
    int64_t permutation_owner,
    int64_t N,
    int64_t T,
    absl::Span<const int64_t> permutation,
    absl::Span<const uint128_t> x,
    int bit_width);


}  // namespace secret_shared_shuffle
}  // namespace spu::mpc::cheetah
