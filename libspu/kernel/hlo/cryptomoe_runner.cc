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
#include "libspu/kernel/hlo/cryptomoe_dispatch.h"
#include "libspu/kernel/hlo/cryptomoe_expert.h"
#include "libspu/kernel/hlo/cryptomoe_router.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "llvm/Support/CommandLine.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

#include "libspu/core/context.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace {

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
  double gate_ms = 0.0;
  double dispatch_ms = 0.0;
  double expert_ms = 0.0;
  double combine_ms = 0.0;
  double total_ms = 0.0;

  uint64_t gate_bytes = 0;
  uint64_t dispatch_bytes = 0;
  uint64_t expert_bytes = 0;
  uint64_t combine_bytes = 0;
  uint64_t total_bytes = 0;
};

llvm::cl::opt<std::string> cli_method(
    "method", llvm::cl::init("cryptomoe"),
    llvm::cl::desc("Method: cryptomoe or dense"));

llvm::cl::opt<int> cli_num_experts(
    "num_experts", llvm::cl::init(4),
    llvm::cl::desc("Number of experts"));

llvm::cl::opt<int> cli_top_k(
    "top_k", llvm::cl::init(2),
    llvm::cl::desc("Number of selected experts per token"));

llvm::cl::opt<int> cli_num_tokens(
    "num_tokens", llvm::cl::init(8),
    llvm::cl::desc("Number of input tokens"));

llvm::cl::opt<int> cli_hidden_dim(
    "hidden_dim", llvm::cl::init(64),
    llvm::cl::desc("Token hidden dimension"));

llvm::cl::opt<int> cli_intermediate_dim(
    "intermediate_dim", llvm::cl::init(128),
    llvm::cl::desc("SwiGLU intermediate dimension"));

llvm::cl::opt<double> cli_alpha(
    "alpha", llvm::cl::init(1.0),
    llvm::cl::desc("CryptoMoE capacity multiplier"));

llvm::cl::opt<int> cli_iterations(
    "iterations", llvm::cl::init(1),
    llvm::cl::desc("Number of measured runs"));

double ElapsedMs(const Clock::time_point& begin,
                 const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

xt::xarray<float> MakeTokens(int64_t m, int64_t d) {
  xt::xarray<float> x = xt::zeros<float>({m, d});
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      const int64_t v = ((i + 3) * (j + 5)) % 17 - 8;
      x(i, j) = static_cast<float>(v) * 0.025F;
    }
  }
  return x;
}

xt::xarray<float> MakeRouterWeight(int64_t d, int64_t n) {
  xt::xarray<float> w = xt::zeros<float>({d, n});
  for (int64_t i = 0; i < d; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const int64_t v = ((i + 7) * (j + 11)) % 19 - 9;
      w(i, j) = static_cast<float>(v) * 0.03F;
    }
  }
  return w;
}

xt::xarray<float> MakeExpertWeight(int64_t n, int64_t rows,
                                   int64_t cols, int64_t salt) {
  xt::xarray<float> w = xt::zeros<float>({n, rows, cols});
  for (int64_t e = 0; e < n; ++e) {
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        const int64_t v =
            ((e + 1) * (i + 3) + (j + 5) * salt) % 13 - 6;
        w(e, i, j) = static_cast<float>(v) * 0.01F;
      }
    }
  }
  return w;
}

spu::Value DenseCombine(
    spu::SPUContext* ctx,
    const std::vector<spu::Value>& expert_outputs,
    const spu::Value& routing_indices,
    const spu::Value& routing_weights) {
  SPU_ENFORCE(!expert_outputs.empty(), "expert_outputs must not be empty");
  SPU_ENFORCE(routing_indices.shape().size() == 2,
              "routing_indices must have shape [m, k]");
  SPU_ENFORCE(routing_weights.shape() == routing_indices.shape(),
              "routing_weights shape mismatch");

  const int64_t num_tokens = routing_indices.shape()[0];
  const int64_t top_k = routing_indices.shape()[1];
  const int64_t hidden_dim = expert_outputs[0].shape()[1];

  auto make_index_constant =
      [&](int64_t value, spu::DataType dtype, const spu::Shape& shape) {
        if (dtype == spu::DT_I32) {
          SPU_ENFORCE(value <= std::numeric_limits<int32_t>::max(),
                      "expert id does not fit DT_I32");
          return spu::kernel::hal::constant(
              ctx, static_cast<int32_t>(value), dtype, shape);
        }
        SPU_ENFORCE(dtype == spu::DT_I64,
                    "routing index dtype must be DT_I32 or DT_I64");
        return spu::kernel::hal::constant(ctx, value, dtype, shape);
      };

  auto output = spu::kernel::hal::zeros(
      ctx, expert_outputs[0].dtype(), expert_outputs[0].shape());

  for (int64_t expert = 0;
       expert < static_cast<int64_t>(expert_outputs.size()); ++expert) {
    SPU_ENFORCE(expert_outputs[expert].shape() ==
                    spu::Shape({num_tokens, hidden_dim}),
                "dense expert output shape mismatch");

    auto expert_scores = spu::kernel::hal::zeros(
        ctx, routing_weights.dtype(), {num_tokens});

    for (int64_t slot = 0; slot < top_k; ++slot) {
      auto index_col = spu::kernel::hal::reshape(
          ctx,
          spu::kernel::hal::slice(
              ctx, routing_indices, {0, slot},
              {num_tokens, slot + 1}),
          {num_tokens});

      auto weight_col = spu::kernel::hal::reshape(
          ctx,
          spu::kernel::hal::slice(
              ctx, routing_weights, {0, slot},
              {num_tokens, slot + 1}),
          {num_tokens});

      auto expert_ids =
          make_index_constant(expert, index_col.dtype(), index_col.shape());
      auto mask = spu::kernel::hal::equal(ctx, index_col, expert_ids);

      auto zeros = spu::kernel::hal::zeros(
          ctx, weight_col.dtype(), weight_col.shape());
      auto selected =
          spu::kernel::hal::select(ctx, mask, weight_col, zeros);

      expert_scores =
          spu::kernel::hal::add(ctx, expert_scores, selected);
    }

    auto score_matrix = spu::kernel::hal::broadcast_to(
        ctx, expert_scores, {num_tokens, hidden_dim}, {0});
    auto weighted_output = spu::kernel::hal::mul(
        ctx, expert_outputs[expert], score_matrix);

    output = spu::kernel::hal::add(ctx, output, weighted_output);
  }

  return output;
}

BenchResult RunCryptoMoE(
    const std::shared_ptr<yacl::link::Context>& lctx,
    const xt::xarray<float>& token_embeddings,
    const xt::xarray<float>& router_weight,
    const xt::xarray<float>& gate_weight,
    const xt::xarray<float>& up_weight,
    const xt::xarray<float>& down_weight,
    int64_t num_experts, int64_t top_k, int64_t capacity) {
  spu::RuntimeConfig config;
  config.set_protocol(spu::ProtocolKind::CHEETAH);
  config.set_field(spu::FieldType::FM64);
  config.set_fxp_fraction_bits(16);
  config.set_fxp_exp_iters(5);
  config.set_experimental_enable_bmm(true);
  config.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
  config.mutable_cheetah_2pc_config()->set_approx_less_precision(4);

  spu::SPUContext ctx = spu::kernel::test::makeSPUContext(config, lctx);

  auto tokens_s =
      spu::kernel::test::makeValue(&ctx, token_embeddings, spu::VIS_SECRET);
  auto router_weight_s =
      spu::kernel::test::makeValue(&ctx, router_weight, spu::VIS_SECRET);

  auto gate_weight_p =
      spu::kernel::hal::constant(&ctx, gate_weight, spu::DT_F32);
  auto up_weight_p =
      spu::kernel::hal::constant(&ctx, up_weight, spu::DT_F32);
  auto down_weight_p =
      spu::kernel::hal::constant(&ctx, down_weight, spu::DT_F32);

  auto gate_weight_v =
      spu::kernel::hal::_p2v(&ctx, gate_weight_p, 1).setDtype(spu::DT_F32);
  auto up_weight_v =
      spu::kernel::hal::_p2v(&ctx, up_weight_p, 1).setDtype(spu::DT_F32);
  auto down_weight_v =
      spu::kernel::hal::_p2v(&ctx, down_weight_p, 1).setDtype(spu::DT_F32);

  BenchResult result;

  const auto total_begin = Clock::now();
  const uint64_t total_bytes_begin =
      lctx->GetStats()->sent_bytes.load();

  auto stage_begin = Clock::now();
  uint64_t stage_bytes_begin = lctx->GetStats()->sent_bytes.load();

  auto routing = spu::kernel::hlo::CryptoMoERoute(
      &ctx, tokens_s, router_weight_s, top_k);

  auto stage_end = Clock::now();
  uint64_t stage_bytes_end = lctx->GetStats()->sent_bytes.load();
  result.gate_ms = ElapsedMs(stage_begin, stage_end);
  result.gate_bytes = stage_bytes_end - stage_bytes_begin;

  std::vector<spu::Value> expert_inputs;
  std::vector<spu::Value> onehots;
  std::vector<spu::Value> scores;
  expert_inputs.reserve(num_experts);
  onehots.reserve(num_experts);
  scores.reserve(num_experts);

  stage_begin = Clock::now();
  stage_bytes_begin = lctx->GetStats()->sent_bytes.load();

  for (int64_t expert = 0; expert < num_experts; ++expert) {
    auto dispatch = spu::kernel::hlo::CryptoMoEDispatchWithAux(
        &ctx, routing.indices, routing.weights, tokens_s,
        expert, capacity);
    expert_inputs.push_back(dispatch.tokens);
    onehots.push_back(dispatch.onehot);
    scores.push_back(dispatch.scores);
  }

  stage_end = Clock::now();
  stage_bytes_end = lctx->GetStats()->sent_bytes.load();
  result.dispatch_ms = ElapsedMs(stage_begin, stage_end);
  result.dispatch_bytes = stage_bytes_end - stage_bytes_begin;

  stage_begin = Clock::now();
  stage_bytes_begin = lctx->GetStats()->sent_bytes.load();

  auto expert_outputs = spu::kernel::hlo::CryptoMoEExpertCompute(
      &ctx, expert_inputs, gate_weight_v, up_weight_v, down_weight_v);

  stage_end = Clock::now();
  stage_bytes_end = lctx->GetStats()->sent_bytes.load();
  result.expert_ms = ElapsedMs(stage_begin, stage_end);
  result.expert_bytes = stage_bytes_end - stage_bytes_begin;

  stage_begin = Clock::now();
  stage_bytes_begin = lctx->GetStats()->sent_bytes.load();

  auto output = spu::kernel::hlo::CryptoMoECombine(
      &ctx, expert_outputs, onehots, scores);

  stage_end = Clock::now();
  stage_bytes_end = lctx->GetStats()->sent_bytes.load();
  result.combine_ms = ElapsedMs(stage_begin, stage_end);
  result.combine_bytes = stage_bytes_end - stage_bytes_begin;

  SPU_ENFORCE(output.isSecret(), "runner output must remain secret");
  SPU_ENFORCE(output.shape() ==
                  spu::Shape({static_cast<int64_t>(token_embeddings.shape()[0]),
                              static_cast<int64_t>(token_embeddings.shape()[1])}),
              "unexpected runner output shape");

  const auto total_end = Clock::now();
  const uint64_t total_bytes_end =
      lctx->GetStats()->sent_bytes.load();

  result.total_ms = ElapsedMs(total_begin, total_end);
  result.total_bytes = total_bytes_end - total_bytes_begin;

  return result;
}


BenchResult RunDense(
    const std::shared_ptr<yacl::link::Context>& lctx,
    const xt::xarray<float>& token_embeddings,
    const xt::xarray<float>& router_weight,
    const xt::xarray<float>& gate_weight,
    const xt::xarray<float>& up_weight,
    const xt::xarray<float>& down_weight,
    int64_t num_experts, int64_t top_k) {
  spu::RuntimeConfig config;
  config.set_protocol(spu::ProtocolKind::CHEETAH);
  config.set_field(spu::FieldType::FM64);
  config.set_fxp_fraction_bits(16);
  config.set_fxp_exp_iters(5);
  config.set_experimental_enable_bmm(true);
  config.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
  config.mutable_cheetah_2pc_config()->set_approx_less_precision(4);

  spu::SPUContext ctx = spu::kernel::test::makeSPUContext(config, lctx);

  auto tokens_s =
      spu::kernel::test::makeValue(&ctx, token_embeddings, spu::VIS_SECRET);
  auto router_weight_s =
      spu::kernel::test::makeValue(&ctx, router_weight, spu::VIS_SECRET);

  auto gate_weight_p =
      spu::kernel::hal::constant(&ctx, gate_weight, spu::DT_F32);
  auto up_weight_p =
      spu::kernel::hal::constant(&ctx, up_weight, spu::DT_F32);
  auto down_weight_p =
      spu::kernel::hal::constant(&ctx, down_weight, spu::DT_F32);

  auto gate_weight_v =
      spu::kernel::hal::_p2v(&ctx, gate_weight_p, 1).setDtype(spu::DT_F32);
  auto up_weight_v =
      spu::kernel::hal::_p2v(&ctx, up_weight_p, 1).setDtype(spu::DT_F32);
  auto down_weight_v =
      spu::kernel::hal::_p2v(&ctx, down_weight_p, 1).setDtype(spu::DT_F32);

  BenchResult result;

  const auto total_begin = Clock::now();
  const uint64_t total_bytes_begin =
      lctx->GetStats()->sent_bytes.load();

  auto stage_begin = Clock::now();
  uint64_t stage_bytes_begin = lctx->GetStats()->sent_bytes.load();

  auto routing = spu::kernel::hlo::CryptoMoERoute(
      &ctx, tokens_s, router_weight_s, top_k);

  auto stage_end = Clock::now();
  uint64_t stage_bytes_end = lctx->GetStats()->sent_bytes.load();
  result.gate_ms = ElapsedMs(stage_begin, stage_end);
  result.gate_bytes = stage_bytes_end - stage_bytes_begin;

  std::vector<spu::Value> expert_inputs(
      static_cast<size_t>(num_experts), tokens_s);

  stage_begin = Clock::now();
  stage_bytes_begin = lctx->GetStats()->sent_bytes.load();

  auto expert_outputs = spu::kernel::hlo::CryptoMoEExpertCompute(
      &ctx, expert_inputs, gate_weight_v, up_weight_v, down_weight_v);

  stage_end = Clock::now();
  stage_bytes_end = lctx->GetStats()->sent_bytes.load();
  result.expert_ms = ElapsedMs(stage_begin, stage_end);
  result.expert_bytes = stage_bytes_end - stage_bytes_begin;

  stage_begin = Clock::now();
  stage_bytes_begin = lctx->GetStats()->sent_bytes.load();

  auto output = DenseCombine(
      &ctx, expert_outputs, routing.indices, routing.weights);

  stage_end = Clock::now();
  stage_bytes_end = lctx->GetStats()->sent_bytes.load();
  result.combine_ms = ElapsedMs(stage_begin, stage_end);
  result.combine_bytes = stage_bytes_end - stage_bytes_begin;

  SPU_ENFORCE(output.isSecret(), "dense output must remain secret");
  SPU_ENFORCE(
      output.shape() ==
          spu::Shape({static_cast<int64_t>(token_embeddings.shape()[0]),
                      static_cast<int64_t>(token_embeddings.shape()[1])}),
      "unexpected dense output shape");

  const auto total_end = Clock::now();
  const uint64_t total_bytes_end =
      lctx->GetStats()->sent_bytes.load();

  result.total_ms = ElapsedMs(total_begin, total_end);
  result.total_bytes = total_bytes_end - total_bytes_begin;

  return result;
}

}  // namespace

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "CryptoMoE experiment runner\n");

  SPU_ENFORCE(cli_method == "cryptomoe" || cli_method == "dense",
              "method must be cryptomoe or dense");
  SPU_ENFORCE(cli_num_experts > 0, "num_experts must be positive");
  SPU_ENFORCE(cli_top_k > 0 && cli_top_k <= cli_num_experts,
              "top_k must be in [1, num_experts]");
  SPU_ENFORCE(cli_num_tokens > 0, "num_tokens must be positive");
  SPU_ENFORCE(cli_hidden_dim > 0, "hidden_dim must be positive");
  SPU_ENFORCE(cli_intermediate_dim > 0,
              "intermediate_dim must be positive");
  SPU_ENFORCE(cli_alpha > 0.0, "alpha must be positive");
  SPU_ENFORCE(cli_iterations > 0, "iterations must be positive");

  const int64_t m = cli_num_tokens;
  const int64_t n = cli_num_experts;
  const int64_t k = cli_top_k;
  const int64_t d = cli_hidden_dim;
  const int64_t h = cli_intermediate_dim;

  int64_t capacity = static_cast<int64_t>(
      std::ceil(cli_alpha * static_cast<double>(m * k) /
                static_cast<double>(n)));
  capacity = std::max<int64_t>(1, std::min<int64_t>(capacity, m));

  const auto tokens = MakeTokens(m, d);
  const auto router_weight = MakeRouterWeight(d, n);
  const auto gate_weight = MakeExpertWeight(n, d, h, 3);
  const auto up_weight = MakeExpertWeight(n, d, h, 5);
  const auto down_weight = MakeExpertWeight(n, h, d, 7);

  std::cout << "CryptoMoE runner configuration\n"
            << "method=" << cli_method << "\n"
            << "field=FM64\n"
            << "protocol=CHEETAH\n"
            << "num_experts=" << n << "\n"
            << "top_k=" << k << "\n"
            << "num_tokens=" << m << "\n"
            << "hidden_dim=" << d << "\n"
            << "intermediate_dim=" << h << "\n"
            << "alpha=" << cli_alpha << "\n"
            << "capacity=" << capacity << "\n"
            << "iterations=" << cli_iterations << "\n";

  for (int iteration = 0; iteration < cli_iterations; ++iteration) {
    auto party_results = spu::mpc::utils::simulate(
        2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          lctx->SetRecvTimeout(120 * 1000);
          if (cli_method == "cryptomoe") {
            return RunCryptoMoE(
                lctx, tokens, router_weight, gate_weight,
                up_weight, down_weight, n, k, capacity);
          }
          return RunDense(
              lctx, tokens, router_weight, gate_weight,
              up_weight, down_weight, n, k);
        });

    SPU_ENFORCE(party_results.size() == 2,
                "expected exactly two party results");

    BenchResult merged;
    merged.gate_ms =
        std::max(party_results[0].gate_ms, party_results[1].gate_ms);
    merged.dispatch_ms =
        std::max(party_results[0].dispatch_ms, party_results[1].dispatch_ms);
    merged.expert_ms =
        std::max(party_results[0].expert_ms, party_results[1].expert_ms);
    merged.combine_ms =
        std::max(party_results[0].combine_ms, party_results[1].combine_ms);
    merged.total_ms =
        std::max(party_results[0].total_ms, party_results[1].total_ms);

    merged.gate_bytes =
        party_results[0].gate_bytes + party_results[1].gate_bytes;
    merged.dispatch_bytes =
        party_results[0].dispatch_bytes + party_results[1].dispatch_bytes;
    merged.expert_bytes =
        party_results[0].expert_bytes + party_results[1].expert_bytes;
    merged.combine_bytes =
        party_results[0].combine_bytes + party_results[1].combine_bytes;
    merged.total_bytes =
        party_results[0].total_bytes + party_results[1].total_bytes;

    constexpr double kMiB = 1024.0 * 1024.0;

    std::cout << std::fixed << std::setprecision(3)
              << "\niteration=" << iteration + 1 << "\n"
              << "gate_ms=" << merged.gate_ms << "\n"
              << "dispatch_ms=" << merged.dispatch_ms << "\n"
              << "expert_ms=" << merged.expert_ms << "\n"
              << "combine_ms=" << merged.combine_ms << "\n"
              << "total_ms=" << merged.total_ms << "\n"
              << "gate_comm_mib=" << merged.gate_bytes / kMiB << "\n"
              << "dispatch_comm_mib="
              << merged.dispatch_bytes / kMiB << "\n"
              << "expert_comm_mib=" << merged.expert_bytes / kMiB << "\n"
              << "combine_comm_mib=" << merged.combine_bytes / kMiB << "\n"
              << "total_comm_mib=" << merged.total_bytes / kMiB << "\n"
              << "latency_ms_per_token=" << merged.total_ms / m << "\n"
              << "comm_mib_per_token="
              << (merged.total_bytes / kMiB) / m << "\n";
  }

  return 0;
}
