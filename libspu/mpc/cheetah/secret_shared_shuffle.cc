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

#include <immintrin.h>

#include "libspu/mpc/cheetah/secret_shared_shuffle.h"

#include <utility>

#include "absl/types/span.h"
#include "yacl/crypto/block_cipher/symmetric_crypto.h"
#include "yacl/crypto/rand/rand.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/common/communicator.h"

namespace spu::mpc::cheetah::secret_shared_shuffle {

namespace {

// Return the MSB-first bit at GGM level `level`.
// level is 0-based: level 0 corresponds to the first branch below root.
uint8_t IndexBit(int64_t index, int depth, int level) {
  return static_cast<uint8_t>(
      (index >> (depth - 1 - level)) & static_cast<int64_t>(1));
}

int TreeDepth(int64_t n) {
  SPU_ENFORCE(n >= 2, "OPV vector length must be >= 2, got {}", n);
  SPU_ENFORCE((n & (n - 1)) == 0,
              "OPV vector length must be a power of two, got {}", n);

  int depth = 0;
  for (int64_t x = n; x > 1; x >>= 1) {
    ++depth;
  }
  return depth;
}

}  // namespace

namespace {

// Literal CipherGPT Common.h CCR preprocessing:
//
//   sigma(x) = _mm_shuffle_epi32(x, 78)
//              XOR
//              (x & makeBlock(0xffffffffffffffff, 0))
//
// Work on the exact 16-byte representation instead of reinterpreting
// uint128_t high/low words manually.
uint128_t CcrSigma(uint128_t x) {
  static_assert(sizeof(uint128_t) == sizeof(__m128i));

  __m128i block;
  std::memcpy(&block, &x, sizeof(block));

  const __m128i mask =
      _mm_set_epi64x(
          static_cast<long long>(0xffffffffffffffffULL),
          0);

  const __m128i shuffled =
      _mm_shuffle_epi32(block, 78);

  const __m128i sigma_block =
      _mm_xor_si128(
          shuffled,
          _mm_and_si128(block, mask));

  uint128_t sigma = 0;
  std::memcpy(&sigma, &sigma_block, sizeof(sigma));
  return sigma;
}

// CipherGPT:
//
//   H(x) = sigma(x) XOR AES_common_key(sigma(x)).
uint128_t CcrFunctionH(
    uint128_t x,
    yacl::crypto::SymmetricCrypto* aes) {
  SPU_ENFORCE(aes != nullptr);

  const uint128_t sigma = CcrSigma(x);

  const std::array<uint128_t, 1> plain = {sigma};
  std::array<uint128_t, 1> cipher{};

  aes->Encrypt(
      absl::MakeConstSpan(plain),
      absl::MakeSpan(cipher));

  return sigma ^ cipher[0];
}

// CipherGPT samples common_AES_key on SERVER and sends it to CLIENT.
// SPU uses zero-based ranks, so rank 0 performs that setup role here.
//
// NOTE: CipherGPT installs this once in StartComputation().  For now
// this port scopes the shared key to one OPV batch; protocol semantics
// are identical, but lifecycle is not yet byte-for-byte source-shaped.
uint128_t ExchangeCommonAesKey(
    const std::shared_ptr<Communicator>& comm) {
  SPU_ENFORCE(comm != nullptr);
  SPU_ENFORCE_EQ(comm->getWorldSize(), size_t{2});

  constexpr size_t kSetupServer = 0;
  constexpr size_t kSetupClient = 1;
  constexpr char kTag[] =
      "cheetah.sss.opv.common_aes_key";

  if (comm->getRank() == kSetupServer) {
    const std::array<uint128_t, 1> key = {
        yacl::crypto::SecureRandU128(),
    };

    comm->sendAsync(
        kSetupClient,
        absl::MakeConstSpan(key),
        kTag);

    return key[0];
  }

  auto key =
      comm->recv<uint128_t>(
          kSetupServer,
          kTag);

  SPU_ENFORCE_EQ(key.size(), size_t{1});
  return key[0];
}

}  // namespace

std::vector<std::vector<uint128_t>> ObliviousPuncturedVectors(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot, int64_t full_rank,
    int64_t n, int64_t num_instances,
    absl::Span<const int64_t> puncture_indices) {
  SPU_ENFORCE(comm != nullptr);
  SPU_ENFORCE(ot != nullptr);
  SPU_ENFORCE(full_rank == 0 || full_rank == 1);
  SPU_ENFORCE_EQ(
      static_cast<int64_t>(comm->getRank()),
      static_cast<int64_t>(ot->Rank()));
  SPU_ENFORCE(num_instances > 0);

  const int depth = TreeDepth(n);
  const int64_t rank =
      static_cast<int64_t>(ot->Rank());
  const int64_t punctured_rank =
      1 - full_rank;

  const size_t num_ots =
      static_cast<size_t>(num_instances) *
      static_cast<size_t>(depth);

  constexpr char kMaskedChoiceTag[] =
      "cheetah.sss.opv.masked_choice";
  constexpr char kLeftSumTag[] =
      "cheetah.sss.opv.left_child_sum";

  const uint128_t common_aes_key =
      ExchangeCommonAesKey(comm);

  yacl::crypto::SymmetricCrypto aes(
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_ECB,
      common_aes_key,
      /*iv=*/0);

  if (rank == full_rank) {
    // CipherGPT full-tree / RCOT-sender side.
    SPU_ENFORCE(puncture_indices.empty());

    auto sender = ot->GetSenderCOT();
    SPU_ENFORCE(sender != nullptr);

    std::vector<uint128_t> cot_messages(num_ots);

    sender->SendRCOT(absl::MakeSpan(cot_messages));
    const uint128_t delta = sender->GetDelta();
    sender->Flush();

    auto masked_choice =
        comm->recv<uint8_t>(
            static_cast<size_t>(punctured_rank),
            kMaskedChoiceTag);

    SPU_ENFORCE_EQ(masked_choice.size(), num_ots);

    // CipherGPT:
    //
    //   if (masked_choice[i])
    //       COT_messages[i] ^= RCOT_Delta;
    for (size_t i = 0; i < num_ots; ++i) {
      SPU_ENFORCE(masked_choice[i] < 2);
      if (masked_choice[i]) {
        cot_messages[i] ^= delta;
      }
    }

    std::vector<std::vector<uint128_t>> outputs(
        static_cast<size_t>(num_instances));

    std::vector<uint128_t> all_left_child_sum(
        num_ots, 0);

    for (int64_t instance = 0;
         instance < num_instances;
         ++instance) {
      std::vector<uint128_t> tree(
          static_cast<size_t>(n), 0);

      for (int d = 0; d < depth; ++d) {
        const size_t ot_idx =
            static_cast<size_t>(instance) *
                static_cast<size_t>(depth) +
            static_cast<size_t>(d);

        uint128_t left_sum = 0;

        if (d == 0) {
          // CipherGPT first layer:
          //
          // tree[0] = random
          // tree[1] = tree[0] XOR Delta
          tree[0] = yacl::crypto::SecureRandU128();
          tree[1] = tree[0] ^ delta;
          left_sum = tree[0];
        } else {
          const int64_t cur_length =
              int64_t{1} << (d + 1);
          const int64_t parents =
              cur_length >> 1;

          // CipherGPT expands backwards in-place.
          for (int64_t j = parents - 1;
               j >= 0;
               --j) {
            const uint128_t parent =
                tree[static_cast<size_t>(j)];

            const uint128_t right =
                CcrFunctionH(parent, &aes);
            const uint128_t left =
                right ^ parent;

            tree[static_cast<size_t>(2 * j)] =
                left;
            tree[static_cast<size_t>(2 * j + 1)] =
                right;

            left_sum ^= left;
          }
        }

        // CipherGPT:
        //
        // all_left_child_sum ^= COT_messages[COT_cnt++]
        all_left_child_sum[ot_idx] =
            left_sum ^ cot_messages[ot_idx];
      }

      outputs[static_cast<size_t>(instance)] =
          std::move(tree);
    }

    comm->sendAsync(
        static_cast<size_t>(punctured_rank),
        absl::MakeConstSpan(all_left_child_sum),
        kLeftSumTag);

    return outputs;
  }

  // CipherGPT punctured / RCOT-receiver side.
  SPU_ENFORCE_EQ(
      static_cast<int64_t>(puncture_indices.size()),
      num_instances);

  for (int64_t instance = 0;
       instance < num_instances;
       ++instance) {
    SPU_ENFORCE(
        puncture_indices[instance] >= 0 &&
        puncture_indices[instance] < n);
  }

  // CipherGPT choice is the complement of the puncture-path bit,
  // i.e. the sibling direction needed for reconstruction.
  std::vector<uint8_t> desired_choice(
      num_ots, 0);

  for (int64_t instance = 0;
       instance < num_instances;
       ++instance) {
    for (int d = 0; d < depth; ++d) {
      const size_t idx =
          static_cast<size_t>(instance) *
              static_cast<size_t>(depth) +
          static_cast<size_t>(d);

      desired_choice[idx] =
          static_cast<uint8_t>(
              1U -
              IndexBit(
                  puncture_indices[instance],
                  depth,
                  d));
    }
  }

  auto receiver = ot->GetReceiverCOT();
  SPU_ENFORCE(receiver != nullptr);

  std::vector<uint128_t> cot_chosen_msg(num_ots);
  std::vector<uint8_t> rcot_choice(num_ots);

  receiver->RecvRCOT(
      absl::MakeSpan(cot_chosen_msg),
      absl::MakeSpan(rcot_choice));

  // CipherGPT:
  //
  // masked_choice =
  //     desired_choice XOR rcot_recv_choice.
  std::vector<uint8_t> masked_choice(num_ots);

  for (size_t i = 0; i < num_ots; ++i) {
    SPU_ENFORCE(rcot_choice[i] < 2);
    masked_choice[i] =
        desired_choice[i] ^ rcot_choice[i];
  }

  comm->sendAsync(
      static_cast<size_t>(full_rank),
      absl::MakeConstSpan(masked_choice),
      kMaskedChoiceTag);

  auto all_left_child_sum =
      comm->recv<uint128_t>(
          static_cast<size_t>(full_rank),
          kLeftSumTag);

  SPU_ENFORCE_EQ(all_left_child_sum.size(), num_ots);

  std::vector<std::vector<uint128_t>> outputs(
      static_cast<size_t>(num_instances));

  for (int64_t instance = 0;
       instance < num_instances;
       ++instance) {
    const int64_t puncture_index =
        puncture_indices[instance];

    std::vector<uint128_t> tree(
        static_cast<size_t>(n), 0);

    int64_t pre_punctured_id = 0;
    int64_t cur_pre_punctured_id = 0;

    for (int d = 0; d < depth; ++d) {
      const size_t ot_idx =
          static_cast<size_t>(instance) *
              static_cast<size_t>(depth) +
          static_cast<size_t>(d);

      const uint8_t choice_bit =
          desired_choice[ot_idx];

      uint128_t b =
          all_left_child_sum[ot_idx] ^
          cot_chosen_msg[ot_idx];

      cur_pre_punctured_id =
          (pre_punctured_id << 1) |
          static_cast<int64_t>(
              choice_bit ^ 1);

      if (d == 0) {
        tree[static_cast<size_t>(choice_bit)] =
            b;
      } else {
        const int64_t cur_length =
            int64_t{1} << (d + 1);
        const int64_t parents =
            cur_length >> 1;

        for (int64_t j = parents - 1;
             j >= 0;
             --j) {
          const uint128_t parent =
              tree[static_cast<size_t>(j)];

          const uint128_t right =
              CcrFunctionH(parent, &aes);
          const uint128_t left =
              right ^ parent;

          const int64_t left_idx =
              2 * j;

          tree[static_cast<size_t>(left_idx)] =
              left;
          tree[static_cast<size_t>(left_idx + 1)] =
              right;

          // CipherGPT:
          //
          // b ^= tree[i | choice_bit]
          b ^= tree[static_cast<size_t>(
              left_idx |
              static_cast<int64_t>(
                  choice_bit))];
        }

        tree[static_cast<size_t>(
            cur_pre_punctured_id ^ 1)] ^= b;
      }

      pre_punctured_id =
          cur_pre_punctured_id;
    }

    SPU_ENFORCE_EQ(
        cur_pre_punctured_id,
        puncture_index);

    tree[static_cast<size_t>(puncture_index)] = 0;

    outputs[static_cast<size_t>(instance)] =
        std::move(tree);
  }

  return outputs;
}

std::vector<uint128_t> ObliviousPuncturedVector(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot, int64_t full_rank,
    int64_t n, int64_t puncture_index) {
  SPU_ENFORCE(ot != nullptr);

  if (ot->Rank() == full_rank) {
    SPU_ENFORCE(
        puncture_index == -1,
        "full-vector OPV party must not receive puncture index");

    auto outputs = ObliviousPuncturedVectors(
        comm,
        ot, full_rank, n,
        /*num_instances=*/1,
        absl::Span<const int64_t>());

    return std::move(outputs.front());
  }

  SPU_ENFORCE(puncture_index >= 0 && puncture_index < n);

  const std::array<int64_t, 1> indices = {
      puncture_index,
  };

  auto outputs = ObliviousPuncturedVectors(
      comm,
      ot, full_rank, n,
      /*num_instances=*/1,
      absl::MakeConstSpan(indices));

  return std::move(outputs.front());
}

namespace {

// Compose output->input permutation maps in execution order.
//
// Applying lhs first and rhs second gives:
//
//   out[i] = in[lhs[rhs[i]]].
std::vector<int64_t> ComposePermutation(
    absl::Span<const int64_t> lhs,
    absl::Span<const int64_t> rhs) {
  SPU_ENFORCE(lhs.size() == rhs.size());

  std::vector<int64_t> out(lhs.size());

  for (size_t i = 0; i < lhs.size(); ++i) {
    out[i] = lhs[static_cast<size_t>(rhs[i])];
  }

  return out;
}

void ValidatePermutation(absl::Span<const int64_t> permutation) {
  const int64_t n =
      static_cast<int64_t>(permutation.size());

  std::vector<bool> seen(static_cast<size_t>(n), false);

  for (int64_t i = 0; i < n; ++i) {
    const int64_t p = permutation[i];

    SPU_ENFORCE(
        p >= 0 && p < n,
        "invalid permutation entry pi[{}]={}",
        i, p);

    SPU_ENFORCE(
        !seen[static_cast<size_t>(p)],
        "duplicate permutation entry {}",
        p);

    seen[static_cast<size_t>(p)] = true;
  }
}

int64_t ExactLog2(int64_t n) {
  SPU_ENFORCE(
      n >= 1 && (n & (n - 1)) == 0,
      "{} is not a power of two",
      n);

  int64_t log = 0;

  while (n > 1) {
    n >>= 1;
    ++log;
  }

  return log;
}

// Route an arbitrary output->input permutation through a Beneš network.
//
// The returned network contains exactly:
//
//   2 * log2(N) - 1
//
// switch layers. At recursive level n, the outer switch layers pair
// wires that differ in the current most-significant routing bit.
//
// This gives the bit-layer sequence:
//
//   0,1,...,n-1,n-2,...,1,0
//
// used by the Section 5 decomposition in [14].
std::vector<std::vector<int64_t>> RouteBenes(
    absl::Span<const int64_t> permutation) {
  const int64_t N =
      static_cast<int64_t>(permutation.size());

  ValidatePermutation(permutation);

  if (N == 1) {
    return {};
  }

  if (N == 2) {
    return {
        std::vector<int64_t>(
            permutation.begin(),
            permutation.end()),
    };
  }

  const int64_t half = N / 2;

  // f[r] is the first-stage switch bit for pair:
  //
  //   r, r + half
  //
  // l[s] is the corresponding final-stage switch bit.
  //
  // For every desired connection output o -> input permutation[o],
  // the two chosen switch settings must send both ends into the same
  // recursive half-network:
  //
  //   f[r] XOR l[s]
  //     = MSB(input) XOR MSB(output).
  std::vector<std::vector<std::pair<int64_t, int>>> first_edges(
      static_cast<size_t>(half));
  std::vector<std::vector<std::pair<int64_t, int>>> last_edges(
      static_cast<size_t>(half));

  for (int64_t output = 0; output < N; ++output) {
    const int64_t input = permutation[output];

    const int64_t first_pair = input % half;
    const int64_t last_pair = output % half;

    const int constraint =
        static_cast<int>(
            (input / half) ^ (output / half));

    first_edges[static_cast<size_t>(first_pair)]
        .emplace_back(last_pair, constraint);

    last_edges[static_cast<size_t>(last_pair)]
        .emplace_back(first_pair, constraint);
  }

  std::vector<int> first_switch(
      static_cast<size_t>(half), -1);
  std::vector<int> last_switch(
      static_cast<size_t>(half), -1);

  // Each connected component is an even cycle/path in the standard
  // Beneš routing constraint graph. Fix one switch arbitrarily and
  // propagate all XOR constraints.
  for (int64_t root = 0; root < half; ++root) {
    if (first_switch[static_cast<size_t>(root)] != -1) {
      continue;
    }

    first_switch[static_cast<size_t>(root)] = 0;

    std::vector<std::pair<bool, int64_t>> queue;
    queue.emplace_back(true, root);

    size_t head = 0;

    while (head < queue.size()) {
      const auto node = queue[head++];
      const bool is_first = node.first;
      const int64_t idx = node.second;

      if (is_first) {
        const int current =
            first_switch[static_cast<size_t>(idx)];

        for (const auto& edge :
             first_edges[static_cast<size_t>(idx)]) {
          const int64_t next = edge.first;
          const int expected = current ^ edge.second;

          auto& value =
              last_switch[static_cast<size_t>(next)];

          if (value == -1) {
            value = expected;
            queue.emplace_back(false, next);
          } else {
            SPU_ENFORCE(
                value == expected,
                "inconsistent Beneš routing constraint");
          }
        }
      } else {
        const int current =
            last_switch[static_cast<size_t>(idx)];

        for (const auto& edge :
             last_edges[static_cast<size_t>(idx)]) {
          const int64_t next = edge.first;
          const int expected = current ^ edge.second;

          auto& value =
              first_switch[static_cast<size_t>(next)];

          if (value == -1) {
            value = expected;
            queue.emplace_back(true, next);
          } else {
            SPU_ENFORCE(
                value == expected,
                "inconsistent Beneš routing constraint");
          }
        }
      }
    }
  }

  std::vector<int64_t> first_layer(
      static_cast<size_t>(N));
  std::vector<int64_t> last_layer(
      static_cast<size_t>(N));

  std::iota(first_layer.begin(), first_layer.end(), 0);
  std::iota(last_layer.begin(), last_layer.end(), 0);

  for (int64_t pair = 0; pair < half; ++pair) {
    if (first_switch[static_cast<size_t>(pair)] != 0) {
      std::swap(
          first_layer[static_cast<size_t>(pair)],
          first_layer[static_cast<size_t>(pair + half)]);
    }

    if (last_switch[static_cast<size_t>(pair)] != 0) {
      std::swap(
          last_layer[static_cast<size_t>(pair)],
          last_layer[static_cast<size_t>(pair + half)]);
    }
  }

  // Derive the two independent middle permutations.
  std::array<std::vector<int64_t>, 2> sub = {
      std::vector<int64_t>(static_cast<size_t>(half), -1),
      std::vector<int64_t>(static_cast<size_t>(half), -1),
  };

  for (int64_t output = 0; output < N; ++output) {
    const int64_t input = permutation[output];

    // first_layer and last_layer are involutions.
    const int64_t middle_output =
        last_layer[static_cast<size_t>(output)];

    const int64_t middle_input =
        first_layer[static_cast<size_t>(input)];

    SPU_ENFORCE(
        middle_output / half == middle_input / half,
        "Beneš route did not enter the same recursive half");

    const int64_t side = middle_output / half;

    sub[static_cast<size_t>(side)]
       [static_cast<size_t>(middle_output % half)] =
        middle_input % half;
  }

  auto upper = RouteBenes(
      absl::MakeConstSpan(sub[0]));
  auto lower = RouteBenes(
      absl::MakeConstSpan(sub[1]));

  SPU_ENFORCE(upper.size() == lower.size());

  std::vector<std::vector<int64_t>> layers;
  layers.reserve(upper.size() + 2);

  layers.push_back(std::move(first_layer));

  for (size_t stage = 0; stage < upper.size(); ++stage) {
    std::vector<int64_t> layer(
        static_cast<size_t>(N));

    for (int64_t i = 0; i < half; ++i) {
      layer[static_cast<size_t>(i)] =
          upper[stage][static_cast<size_t>(i)];

      layer[static_cast<size_t>(i + half)] =
          lower[stage][static_cast<size_t>(i)] + half;
    }

    layers.push_back(std::move(layer));
  }

  layers.push_back(std::move(last_layer));

  return layers;
}

std::vector<int64_t> ComposeLayerRange(
    const std::vector<std::vector<int64_t>>& stages,
    int64_t begin, int64_t end) {
  SPU_ENFORCE(begin >= 0);
  SPU_ENFORCE(begin < end);
  SPU_ENFORCE(
      end <= static_cast<int64_t>(stages.size()));

  const int64_t N =
      static_cast<int64_t>(stages.front().size());

  std::vector<int64_t> composed(
      static_cast<size_t>(N));

  std::iota(composed.begin(), composed.end(), 0);

  for (int64_t stage = begin; stage < end; ++stage) {
    composed = ComposePermutation(
        absl::MakeConstSpan(composed),
        absl::MakeConstSpan(
            stages[static_cast<size_t>(stage)]));
  }

  return composed;
}

}  // namespace

std::vector<BenesSubPermutationLayer> BenesDecomposePermutation(
    absl::Span<const int64_t> permutation, int64_t T) {
  const int64_t N =
      static_cast<int64_t>(permutation.size());

  SPU_ENFORCE(N >= 2);
  ValidatePermutation(permutation);

  const int64_t n = ExactLog2(N);
  const int64_t t = ExactLog2(T);

  SPU_ENFORCE(
      T >= 2 && T <= N,
      "require 2 <= T <= N, got T={}, N={}",
      T, N);

  const auto stages = RouteBenes(permutation);

  SPU_ENFORCE(
      static_cast<int64_t>(stages.size()) ==
          2 * n - 1);

  const int64_t q = (n + t - 1) / t;
  const int64_t d = 2 * q - 1;

  // [14], Section 5:
  //
  // Outer groups contain t consecutive Beneš stages.
  // The remaining middle group contains
  //
  //   (2n - 1) - t(d - 1)
  //
  // stages.
  std::vector<std::pair<int64_t, int64_t>> ranges;
  ranges.reserve(static_cast<size_t>(d));

  int64_t left = 0;
  int64_t right =
      static_cast<int64_t>(stages.size());

  for (int64_t i = 0; i < q - 1; ++i) {
    ranges.emplace_back(left, left + t);
    left += t;
  }

  const int64_t middle_end = right - (q - 1) * t;
  ranges.emplace_back(left, middle_end);
  left = middle_end;

  for (int64_t i = 0; i < q - 1; ++i) {
    ranges.emplace_back(left, left + t);
    left += t;
  }

  SPU_ENFORCE(
      left == static_cast<int64_t>(stages.size()));
  SPU_ENFORCE(
      static_cast<int64_t>(ranges.size()) == d);

  // Beneš stage bit sequence:
  //
  //   0,1,...,n-1,n-2,...,0
  //
  // where bit 0 is the MSB of the wire number.
  std::vector<int64_t> stage_bits;
  stage_bits.reserve(stages.size());

  for (int64_t bit = 0; bit < n; ++bit) {
    stage_bits.push_back(bit);
  }

  for (int64_t bit = n - 2; bit >= 0; --bit) {
    stage_bits.push_back(bit);
  }

  std::vector<BenesSubPermutationLayer> result;
  result.reserve(static_cast<size_t>(d));

  for (const auto& range : ranges) {
    const int64_t begin = range.first;
    const int64_t end = range.second;

    auto global_perm =
        ComposeLayerRange(stages, begin, end);

    int64_t min_bit = n;
    int64_t max_bit = -1;

    for (int64_t stage = begin; stage < end; ++stage) {
      const int64_t bit =
          stage_bits[static_cast<size_t>(stage)];

      min_bit = std::min(min_bit, bit);
      max_bit = std::max(max_bit, bit);
    }

    SPU_ENFORCE(max_bit >= min_bit);

    // Every ordinary t-stage block already spans t consecutive bits.
    // The middle block can span fewer than t distinct routing bits when
    // log(T) does not divide log(N). Enlarge it to a t-bit window; this
    // only adds identity dimensions and therefore still yields an
    // N/T collection of T-element permutations as required by [14].
    int64_t window_begin = min_bit;

    if (max_bit - min_bit + 1 < t) {
      window_begin =
          std::max<int64_t>(0, max_bit - t + 1);

      if (window_begin + t > n) {
        window_begin = n - t;
      }
    }

    SPU_ENFORCE(window_begin >= 0);
    SPU_ENFORCE(window_begin + t <= n);
    SPU_ENFORCE(min_bit >= window_begin);
    SPU_ENFORCE(max_bit < window_begin + t);

    // Convert the MSB-based Beneš bit window to the ordinary integer
    // bit positions used by wire numbers.
    const int64_t shift =
        n - (window_begin + t);

    const int64_t local_mask = T - 1;
    const int64_t lower_mask =
        shift == 0 ? 0 : ((int64_t{1} << shift) - 1);

    const int64_t num_groups = N / T;

    std::vector<BenesSubPermutation> layer(
        static_cast<size_t>(num_groups));

    for (auto& subperm : layer) {
      subperm.positions.assign(
          static_cast<size_t>(T), -1);

      subperm.permutation.assign(
          static_cast<size_t>(T), -1);
    }

    // Build each T-wire group.
    for (int64_t global = 0; global < N; ++global) {
      const int64_t local =
          (global >> shift) & local_mask;

      const int64_t lower =
          global & lower_mask;

      const int64_t upper =
          global >> (shift + t);

      const int64_t group =
          (upper << shift) | lower;

      SPU_ENFORCE(
          group >= 0 && group < num_groups);

      layer[static_cast<size_t>(group)]
           .positions[static_cast<size_t>(local)] =
          global;
    }

    // Convert the composed global permutation into N/T local
    // T-element permutations.
    for (int64_t group = 0;
         group < num_groups;
         ++group) {
      auto& subperm =
          layer[static_cast<size_t>(group)];

      for (int64_t local_output = 0;
           local_output < T;
           ++local_output) {
        const int64_t global_output =
            subperm.positions[
                static_cast<size_t>(local_output)];

        SPU_ENFORCE(global_output >= 0);

        const int64_t global_input =
            global_perm[
                static_cast<size_t>(global_output)];

        const int64_t input_lower =
            global_input & lower_mask;

        const int64_t input_upper =
            global_input >> (shift + t);

        const int64_t input_group =
            (input_upper << shift) | input_lower;

        SPU_ENFORCE(
            input_group == group,
            "Beneš grouped layer crossed a T-element group");

        const int64_t local_input =
            (global_input >> shift) & local_mask;

        subperm.permutation[
            static_cast<size_t>(local_output)] =
            local_input;
      }

      ValidatePermutation(
          absl::MakeConstSpan(
              subperm.permutation));
    }

    result.push_back(std::move(layer));
  }

  return result;
}



BatchedShareTranslationOutput ShareTranslations(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot,
    int64_t permutation_owner,
    int64_t n,
    int64_t num_permutations,
    absl::Span<const int64_t> permutations_flat,
    int bit_width) {
  SPU_ENFORCE(ot != nullptr);
  SPU_ENFORCE(
      permutation_owner == 0 || permutation_owner == 1,
      "permutation_owner must be rank 0 or 1");
  SPU_ENFORCE(n >= 2);
  SPU_ENFORCE(num_permutations > 0);
  SPU_ENFORCE(
      bit_width >= 1 && bit_width <= 128,
      "invalid bit width {}",
      bit_width);

  const int64_t rank =
      static_cast<int64_t>(ot->Rank());

  const bool owns_permutation =
      rank == permutation_owner;

  if (owns_permutation) {
    SPU_ENFORCE(
        static_cast<int64_t>(permutations_flat.size()) ==
            num_permutations * n,
        "expected {} flattened permutation entries, got {}",
        num_permutations * n,
        permutations_flat.size());

    // Validate every private permutation independently.
    for (int64_t instance = 0;
         instance < num_permutations;
         ++instance) {
      std::vector<bool> seen(
          static_cast<size_t>(n), false);

      for (int64_t i = 0; i < n; ++i) {
        const int64_t value =
            permutations_flat[
                static_cast<size_t>(instance * n + i)];

        SPU_ENFORCE(
            value >= 0 && value < n,
            "invalid permutation entry {}",
            value);

        SPU_ENFORCE(
            !seen[static_cast<size_t>(value)],
            "duplicate permutation entry {}",
            value);

        seen[static_cast<size_t>(value)] = true;
      }
    }
  } else {
    SPU_ENFORCE(
        permutations_flat.empty(),
        "non-owner must not receive private permutations");
  }

  const uint128_t mask =
      bit_width == 128
          ? ~static_cast<uint128_t>(0)
          : (static_cast<uint128_t>(1) << bit_width) - 1;

  const auto add_mod =
      [mask](uint128_t x, uint128_t y) {
        return (x + y) & mask;
      };

  const auto sub_mod =
      [mask](uint128_t x, uint128_t y) {
        return (x - y) & mask;
      };

  const int64_t full_rank =
      1 - permutation_owner;

  // Each Share Translation on n elements needs n OPVs.
  //
  // Flatten:
  //
  //   ST 0: rows 0 ... n-1
  //   ST 1: rows n ... 2n-1
  //   ...
  //
  // All num_permutations*n OPVs enter ONE batched OPV invocation.
  const int64_t num_opvs =
      num_permutations * n;

  std::vector<std::vector<uint128_t>> opv_rows;

  if (owns_permutation) {
    opv_rows = ObliviousPuncturedVectors(
        comm,
        ot,
        full_rank,
        n,
        num_opvs,
        permutations_flat);
  } else {
    opv_rows = ObliviousPuncturedVectors(
        comm,
        ot,
        full_rank,
        n,
        num_opvs,
        absl::Span<const int64_t>());
  }

  SPU_ENFORCE(
      static_cast<int64_t>(opv_rows.size()) ==
          num_opvs);

  BatchedShareTranslationOutput output;

  if (!owns_permutation) {
    output.a.resize(
        static_cast<size_t>(num_permutations));
    output.b.resize(
        static_cast<size_t>(num_permutations));

    for (int64_t instance = 0;
         instance < num_permutations;
         ++instance) {
      auto& a =
          output.a[static_cast<size_t>(instance)];
      auto& b =
          output.b[static_cast<size_t>(instance)];

      a.assign(static_cast<size_t>(n), 0);
      b.assign(static_cast<size_t>(n), 0);

      for (int64_t row = 0; row < n; ++row) {
        const auto& v =
            opv_rows[
                static_cast<size_t>(
                    instance * n + row)];

        SPU_ENFORCE(
            static_cast<int64_t>(v.size()) == n);

        for (int64_t col = 0; col < n; ++col) {
          const uint128_t value =
              v[static_cast<size_t>(col)] & mask;

          // b[row] = row sum.
          b[static_cast<size_t>(row)] =
              add_mod(
                  b[static_cast<size_t>(row)],
                  value);

          // a[col] = column sum.
          a[static_cast<size_t>(col)] =
              add_mod(
                  a[static_cast<size_t>(col)],
                  value);
        }
      }
    }

    return output;
  }

  output.delta.resize(
      static_cast<size_t>(num_permutations));

  for (int64_t instance = 0;
       instance < num_permutations;
       ++instance) {
    auto& delta =
        output.delta[static_cast<size_t>(instance)];

    delta.assign(static_cast<size_t>(n), 0);

    for (int64_t i = 0; i < n; ++i) {
      const int64_t pi_i =
          permutations_flat[
              static_cast<size_t>(
                  instance * n + i)];

      uint128_t row_without = 0;
      uint128_t col_without = 0;

      // Row i, excluding v_i[pi(i)].
      const auto& row =
          opv_rows[
              static_cast<size_t>(
                  instance * n + i)];

      for (int64_t j = 0; j < n; ++j) {
        if (j != pi_i) {
          row_without =
              add_mod(
                  row_without,
                  row[static_cast<size_t>(j)] &
                      mask);
        }
      }

      // Column pi(i), excluding v_i[pi(i)].
      for (int64_t j = 0; j < n; ++j) {
        if (j == i) {
          continue;
        }

        const auto& other_row =
            opv_rows[
                static_cast<size_t>(
                    instance * n + j)];

        col_without =
            add_mod(
                col_without,
                other_row[
                    static_cast<size_t>(pi_i)] &
                    mask);
      }

      // Additive-group convention from [14]:
      //
      //   delta = b - pi(a)
      //
      // therefore
      //
      //   delta[i] =
      //       row_i_without -
      //       column_pi(i)_without.
      delta[static_cast<size_t>(i)] =
          sub_mod(
              row_without,
              col_without);
    }
  }

  return output;
}

ShareTranslationOutput ShareTranslation(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot,
    int64_t permutation_owner,
    int64_t n,
    absl::Span<const int64_t> permutation,
    int bit_width) {
  SPU_ENFORCE(ot != nullptr);
  SPU_ENFORCE(
      permutation_owner == 0 || permutation_owner == 1,
      "permutation_owner must be rank 0 or 1");
  SPU_ENFORCE(n >= 2, "Share Translation n must be >= 2");
  SPU_ENFORCE(
      bit_width >= 1 && bit_width <= 128,
      "invalid Share Translation bit width {}",
      bit_width);

  const int64_t rank = static_cast<int64_t>(ot->Rank());
  const bool owns_permutation = rank == permutation_owner;

  if (owns_permutation) {
    SPU_ENFORCE(
        static_cast<int64_t>(permutation.size()) == n,
        "permutation owner expects {} permutation entries, got {}",
        n, permutation.size());

    // Validate that the private input is actually a permutation.
    std::vector<bool> seen(static_cast<size_t>(n), false);

    for (int64_t i = 0; i < n; ++i) {
      const int64_t p = permutation[i];

      SPU_ENFORCE(
          p >= 0 && p < n,
          "invalid permutation entry pi[{}]={}",
          i, p);

      SPU_ENFORCE(
          !seen[static_cast<size_t>(p)],
          "duplicate permutation entry {}",
          p);

      seen[static_cast<size_t>(p)] = true;
    }
  } else {
    // The non-owner must never receive the private permutation.
    SPU_ENFORCE(
        permutation.empty(),
        "non permutation-owner must not receive permutation");
  }

  const uint128_t mask =
      bit_width == 128
          ? ~static_cast<uint128_t>(0)
          : (static_cast<uint128_t>(1) << bit_width) - 1;

  const auto normalize =
      [mask](uint128_t x) -> uint128_t {
    return x & mask;
  };

  const auto add_mod =
      [mask](uint128_t lhs, uint128_t rhs) -> uint128_t {
    return (lhs + rhs) & mask;
  };

  const auto sub_mod =
      [mask](uint128_t lhs, uint128_t rhs) -> uint128_t {
    return (lhs - rhs) & mask;
  };

  // ST permutation owner must be the punctured OPV party.
  // The opposite rank obtains each complete OPV vector.
  const int64_t full_rank = 1 - permutation_owner;

  std::vector<std::vector<uint128_t>> matrix;

  if (owns_permutation) {
    // N parallel OPVs, with puncture position pi(i) in row i.
    matrix = ObliviousPuncturedVectors(
        comm,
        ot,
        full_rank,
        n,
        /*num_instances=*/n,
        permutation);
  } else {
    matrix = ObliviousPuncturedVectors(
        comm,
        ot,
        full_rank,
        n,
        /*num_instances=*/n,
        absl::Span<const int64_t>());
  }

  SPU_ENFORCE(
      static_cast<int64_t>(matrix.size()) == n);

  for (const auto& row : matrix) {
    SPU_ENFORCE(
        static_cast<int64_t>(row.size()) == n);
  }

  ShareTranslationOutput output;

  if (!owns_permutation) {
    // [14], Section 4.2, Step 3:
    //
    //   b[i] = sum_j v_i[j]   -- row sums
    //   a[i] = sum_j v_j[i]   -- column sums
    //
    // The full-matrix party learns both masks.
    output.a.assign(static_cast<size_t>(n), 0);
    output.b.assign(static_cast<size_t>(n), 0);

    for (int64_t i = 0; i < n; ++i) {
      uint128_t row_sum = 0;
      uint128_t col_sum = 0;

      for (int64_t j = 0; j < n; ++j) {
        row_sum = add_mod(
            row_sum,
            normalize(matrix[static_cast<size_t>(i)]
                            [static_cast<size_t>(j)]));

        col_sum = add_mod(
            col_sum,
            normalize(matrix[static_cast<size_t>(j)]
                            [static_cast<size_t>(i)]));
      }

      output.b[static_cast<size_t>(i)] = row_sum;
      output.a[static_cast<size_t>(i)] = col_sum;
    }

    return output;
  }

  // [14], Section 4.2, Step 2:
  //
  // delta[i] =
  //   sum_{j != pi(i)} v_i[j]
  //   -
  //   sum_{j != i} v_j[pi(i)]
  //
  // The unavailable element v_i[pi(i)] appears in neither sum.
  output.delta.assign(static_cast<size_t>(n), 0);

  for (int64_t i = 0; i < n; ++i) {
    const int64_t pi_i = permutation[i];

    uint128_t row_without_puncture = 0;
    uint128_t column_without_puncture = 0;

    for (int64_t j = 0; j < n; ++j) {
      if (j != pi_i) {
        row_without_puncture = add_mod(
            row_without_puncture,
            normalize(matrix[static_cast<size_t>(i)]
                            [static_cast<size_t>(j)]));
      }

      if (j != i) {
        column_without_puncture = add_mod(
            column_without_puncture,
            normalize(matrix[static_cast<size_t>(j)]
                            [static_cast<size_t>(pi_i)]));
      }
    }

    output.delta[static_cast<size_t>(i)] =
        sub_mod(
            row_without_puncture,
            column_without_puncture);
  }

  return output;
}


std::vector<uint128_t> PermuteAndShare(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<BasicOTProtocols>& ot,
    int64_t permutation_owner,
    int64_t N,
    int64_t T,
    absl::Span<const int64_t> permutation,
    absl::Span<const uint128_t> x,
    int bit_width) {
  SPU_ENFORCE(comm != nullptr);
  SPU_ENFORCE(ot != nullptr);
  SPU_ENFORCE(
      permutation_owner == 0 || permutation_owner == 1,
      "permutation_owner must be 0 or 1");
  SPU_ENFORCE(
      N >= 2 && (N & (N - 1)) == 0,
      "Permute+Share currently requires power-of-two N, got {}",
      N);
  SPU_ENFORCE(
      T >= 2 && T <= N && (T & (T - 1)) == 0,
      "Permute+Share requires power-of-two T with 2<=T<=N");
  SPU_ENFORCE(
      bit_width >= 1 && bit_width <= 128,
      "invalid bit width {}",
      bit_width);

  const int64_t rank =
      static_cast<int64_t>(comm->getRank());
  const int64_t data_owner =
      1 - permutation_owner;
  const bool owns_permutation =
      rank == permutation_owner;

  SPU_ENFORCE(
      static_cast<int64_t>(ot->Rank()) == rank,
      "Communicator and BasicOTProtocols rank mismatch");

  if (owns_permutation) {
    SPU_ENFORCE(
        static_cast<int64_t>(permutation.size()) == N,
        "permutation owner expects {} entries, got {}",
        N, permutation.size());
    SPU_ENFORCE(
        x.empty(),
        "permutation owner must not receive plaintext x");
  } else {
    SPU_ENFORCE(
        permutation.empty(),
        "data owner must not receive private permutation");
    SPU_ENFORCE(
        static_cast<int64_t>(x.size()) == N,
        "data owner expects {} x elements, got {}",
        N, x.size());
  }

  const uint128_t mask =
      bit_width == 128
          ? ~static_cast<uint128_t>(0)
          : (static_cast<uint128_t>(1) << bit_width) - 1;

  const auto add_mod =
      [mask](uint128_t lhs, uint128_t rhs) -> uint128_t {
    return (lhs + rhs) & mask;
  };

  const auto sub_mod =
      [mask](uint128_t lhs, uint128_t rhs) -> uint128_t {
    return (lhs - rhs) & mask;
  };

  // ----------------------------------------------------------
  // Step 1: P0/permutation-owner computes the (T,d)
  // subpermutation representation.
  //
  // The non-owner derives only the public group-position template
  // from the identity permutation. Group positions depend on N,T
  // and layer number, not on the private switch settings.
  // ----------------------------------------------------------

  std::vector<int64_t> local_perm;

  if (owns_permutation) {
    local_perm.assign(
        permutation.begin(),
        permutation.end());
  } else {
    local_perm.resize(static_cast<size_t>(N));
    std::iota(local_perm.begin(), local_perm.end(), 0);
  }

  const auto layers =
      BenesDecomposePermutation(
          absl::MakeConstSpan(local_perm), T);

  const int64_t d =
      static_cast<int64_t>(layers.size());
  const int64_t groups_per_layer =
      N / T;
  const int64_t num_share_translations =
      d * groups_per_layer;

  SPU_ENFORCE(d >= 1);

  // Flatten all d*N/T private T-element permutations in
  // layer-major/group-major order.
  std::vector<int64_t> flattened_permutations;

  if (owns_permutation) {
    flattened_permutations.reserve(
        static_cast<size_t>(
            num_share_translations * T));

    for (const auto& layer : layers) {
      SPU_ENFORCE(
          static_cast<int64_t>(layer.size()) ==
              groups_per_layer);

      for (const auto& subperm : layer) {
        flattened_permutations.insert(
            flattened_permutations.end(),
            subperm.permutation.begin(),
            subperm.permutation.end());
      }
    }
  }

  // ----------------------------------------------------------
  // Step 2:
  //
  // Run all d*N/T ShareTrans_T instances in parallel.
  // ----------------------------------------------------------

  BatchedShareTranslationOutput st;

  if (owns_permutation) {
    st = ShareTranslations(
        comm,
        ot,
        permutation_owner,
        T,
        num_share_translations,
        absl::MakeConstSpan(flattened_permutations),
        bit_width);
  } else {
    st = ShareTranslations(
        comm,
        ot,
        permutation_owner,
        T,
        num_share_translations,
        absl::Span<const int64_t>(),
        bit_width);
  }

  // Reassemble each collection of N/T T-element ShareTrans outputs
  // into the full N-element vectors:
  //
  //   Delta^(i), a^(i), b^(i).
  std::vector<std::vector<uint128_t>> delta_layers;
  std::vector<std::vector<uint128_t>> a_layers;
  std::vector<std::vector<uint128_t>> b_layers;

  if (owns_permutation) {
    SPU_ENFORCE(
        static_cast<int64_t>(st.delta.size()) ==
            num_share_translations);

    delta_layers.assign(
        static_cast<size_t>(d),
        std::vector<uint128_t>(
            static_cast<size_t>(N), 0));
  } else {
    SPU_ENFORCE(
        static_cast<int64_t>(st.a.size()) ==
            num_share_translations);
    SPU_ENFORCE(
        static_cast<int64_t>(st.b.size()) ==
            num_share_translations);

    a_layers.assign(
        static_cast<size_t>(d),
        std::vector<uint128_t>(
            static_cast<size_t>(N), 0));

    b_layers.assign(
        static_cast<size_t>(d),
        std::vector<uint128_t>(
            static_cast<size_t>(N), 0));
  }

  for (int64_t layer_idx = 0;
       layer_idx < d;
       ++layer_idx) {
    const auto& layer =
        layers[static_cast<size_t>(layer_idx)];

    for (int64_t group = 0;
         group < groups_per_layer;
         ++group) {
      const int64_t instance =
          layer_idx * groups_per_layer + group;

      const auto& subperm =
          layer[static_cast<size_t>(group)];

      SPU_ENFORCE(
          static_cast<int64_t>(
              subperm.positions.size()) == T);

      for (int64_t local = 0;
           local < T;
           ++local) {
        const int64_t global =
            subperm.positions[
                static_cast<size_t>(local)];

        SPU_ENFORCE(
            global >= 0 && global < N);

        if (owns_permutation) {
          SPU_ENFORCE(
              static_cast<int64_t>(
                  st.delta[
                      static_cast<size_t>(instance)]
                      .size()) == T);

          delta_layers[
              static_cast<size_t>(layer_idx)]
              [static_cast<size_t>(global)] =
              st.delta[
                  static_cast<size_t>(instance)]
                  [static_cast<size_t>(local)] &
              mask;
        } else {
          SPU_ENFORCE(
              static_cast<int64_t>(
                  st.a[
                      static_cast<size_t>(instance)]
                      .size()) == T);
          SPU_ENFORCE(
              static_cast<int64_t>(
                  st.b[
                      static_cast<size_t>(instance)]
                      .size()) == T);

          a_layers[
              static_cast<size_t>(layer_idx)]
              [static_cast<size_t>(global)] =
              st.a[
                  static_cast<size_t>(instance)]
                  [static_cast<size_t>(local)] &
              mask;

          b_layers[
              static_cast<size_t>(layer_idx)]
              [static_cast<size_t>(global)] =
              st.b[
                  static_cast<size_t>(instance)]
                  [static_cast<size_t>(local)] &
              mask;
        }
      }
    }
  }

  // Helper: apply one public-structure/private-switch subpermutation
  // layer to a full N-element vector.
  const auto apply_layer =
      [&](const BenesSubPermutationLayer& layer,
          const std::vector<uint128_t>& input)
          -> std::vector<uint128_t> {
    SPU_ENFORCE(
        static_cast<int64_t>(input.size()) == N);

    std::vector<uint128_t> output(
        static_cast<size_t>(N), 0);

    for (const auto& subperm : layer) {
      SPU_ENFORCE(
          subperm.positions.size() ==
              subperm.permutation.size());

      for (size_t local_output = 0;
           local_output < subperm.positions.size();
           ++local_output) {
        const int64_t global_output =
            subperm.positions[local_output];

        const int64_t local_input =
            subperm.permutation[local_output];

        SPU_ENFORCE(
            local_input >= 0 &&
            static_cast<size_t>(local_input) <
                subperm.positions.size());

        const int64_t global_input =
            subperm.positions[
                static_cast<size_t>(local_input)];

        output[
            static_cast<size_t>(global_output)] =
            input[
                static_cast<size_t>(global_input)] &
            mask;
      }
    }

    return output;
  };

  // ----------------------------------------------------------
  // Step 3: data owner P1
  //
  //   delta^(i) = a^(i+1) - b^(i)
  //   m         = x + a^(1)
  //   w <- G^N
  //
  // Send all of them as one message:
  //
  //   [delta^1 ... delta^(d-1) | m | w]
  //
  // This contains exactly (d+1)N ring elements, matching the
  // paper's online communication term.
  // ----------------------------------------------------------

  constexpr const char* kTagOwner0 =
      "CheetahSSS.PermuteShare.Owner0.Online";
  constexpr const char* kTagOwner1 =
      "CheetahSSS.PermuteShare.Owner1.Online";

  const char* tag =
      permutation_owner == 0
          ? kTagOwner0
          : kTagOwner1;

  if (!owns_permutation) {
    std::vector<uint128_t> online_message;
    online_message.reserve(
        static_cast<size_t>((d + 1) * N));

    for (int64_t layer_idx = 0;
         layer_idx < d - 1;
         ++layer_idx) {
      for (int64_t j = 0; j < N; ++j) {
        online_message.push_back(
            sub_mod(
                a_layers[
                    static_cast<size_t>(
                        layer_idx + 1)]
                    [static_cast<size_t>(j)],
                b_layers[
                    static_cast<size_t>(
                        layer_idx)]
                    [static_cast<size_t>(j)]));
      }
    }

    std::vector<uint128_t> m(
        static_cast<size_t>(N));

    for (int64_t j = 0; j < N; ++j) {
      m[static_cast<size_t>(j)] =
          add_mod(
              x[static_cast<size_t>(j)] & mask,
              a_layers[0][static_cast<size_t>(j)]);

      online_message.push_back(
          m[static_cast<size_t>(j)]);
    }

    std::vector<uint128_t> w(
        static_cast<size_t>(N));

    for (int64_t j = 0; j < N; ++j) {
      w[static_cast<size_t>(j)] =
          yacl::crypto::SecureRandU128() & mask;

      online_message.push_back(
          w[static_cast<size_t>(j)]);
    }

    SPU_ENFORCE(
        static_cast<int64_t>(
            online_message.size()) ==
            (d + 1) * N);

    comm->sendAsync(
        static_cast<size_t>(permutation_owner),
        absl::MakeConstSpan(online_message),
        tag);

    // P1 output:
    //
    //   w - b^(d)
    std::vector<uint128_t> output(
        static_cast<size_t>(N));

    for (int64_t j = 0; j < N; ++j) {
      output[static_cast<size_t>(j)] =
          sub_mod(
              w[static_cast<size_t>(j)],
              b_layers.back()[
                  static_cast<size_t>(j)]);
    }

    return output;
  }

  // ----------------------------------------------------------
  // Step 4: permutation owner P0
  // ----------------------------------------------------------

  auto online_message =
      comm->recv<uint128_t>(
          static_cast<size_t>(data_owner),
          tag);

  SPU_ENFORCE(
      static_cast<int64_t>(
          online_message.size()) ==
          (d + 1) * N,
      "unexpected Permute+Share online message size");

  std::vector<std::vector<uint128_t>>
      transition_delta(
          static_cast<size_t>(
              std::max<int64_t>(0, d - 1)),
          std::vector<uint128_t>(
              static_cast<size_t>(N), 0));

  size_t cursor = 0;

  for (int64_t layer_idx = 0;
       layer_idx < d - 1;
       ++layer_idx) {
    for (int64_t j = 0; j < N; ++j) {
      transition_delta[
          static_cast<size_t>(layer_idx)]
          [static_cast<size_t>(j)] =
          online_message[cursor++] & mask;
    }
  }

  std::vector<uint128_t> m(
      static_cast<size_t>(N));
  std::vector<uint128_t> w(
      static_cast<size_t>(N));

  for (int64_t j = 0; j < N; ++j) {
    m[static_cast<size_t>(j)] =
        online_message[cursor++] & mask;
  }

  for (int64_t j = 0; j < N; ++j) {
    w[static_cast<size_t>(j)] =
        online_message[cursor++] & mask;
  }

  SPU_ENFORCE(cursor == online_message.size());

  // Compute the nested:
  //
  // Delta =
  //   Delta^d +
  //   pi_d(delta^(d-1) + Delta^(d-1) +
  //     pi_(d-1)( ... pi_2(delta^1 + Delta^1)))
  std::vector<uint128_t> final_delta;

  if (d == 1) {
    final_delta = delta_layers[0];
  } else {
    std::vector<uint128_t> acc(
        static_cast<size_t>(N));

    for (int64_t j = 0; j < N; ++j) {
      acc[static_cast<size_t>(j)] =
          add_mod(
              transition_delta[0]
                  [static_cast<size_t>(j)],
              delta_layers[0]
                  [static_cast<size_t>(j)]);
    }

    for (int64_t layer_idx = 1;
         layer_idx < d - 1;
         ++layer_idx) {
      acc = apply_layer(
          layers[static_cast<size_t>(
              layer_idx)],
          acc);

      for (int64_t j = 0; j < N; ++j) {
        acc[static_cast<size_t>(j)] =
            add_mod(
                add_mod(
                    acc[static_cast<size_t>(j)],
                    delta_layers[
                        static_cast<size_t>(
                            layer_idx)]
                        [static_cast<size_t>(j)]),
                transition_delta[
                    static_cast<size_t>(
                        layer_idx)]
                    [static_cast<size_t>(j)]);
      }
    }

    acc = apply_layer(
        layers.back(), acc);

    final_delta.resize(
        static_cast<size_t>(N));

    for (int64_t j = 0; j < N; ++j) {
      final_delta[
          static_cast<size_t>(j)] =
          add_mod(
              delta_layers.back()[
                  static_cast<size_t>(j)],
              acc[static_cast<size_t>(j)]);
    }
  }

  // Directly compute pi(m) using the original private permutation.
  std::vector<uint128_t> pi_m(
      static_cast<size_t>(N));

  for (int64_t j = 0; j < N; ++j) {
    const int64_t src =
        permutation[static_cast<size_t>(j)];

    SPU_ENFORCE(src >= 0 && src < N);

    pi_m[static_cast<size_t>(j)] =
        m[static_cast<size_t>(src)] & mask;
  }

  // P0 output:
  //
  //   pi(m) + Delta - w
  std::vector<uint128_t> output(
      static_cast<size_t>(N));

  for (int64_t j = 0; j < N; ++j) {
    output[static_cast<size_t>(j)] =
        sub_mod(
            add_mod(
                pi_m[static_cast<size_t>(j)],
                final_delta[
                    static_cast<size_t>(j)]),
            w[static_cast<size_t>(j)]);
  }

  return output;
}


}  // namespace spu::mpc::cheetah::secret_shared_shuffle
