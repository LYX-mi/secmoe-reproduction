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

#include "libspu/kernel/hlo/cryptomoe_router.h"

#include <array>
#include <vector>

#include "libspu/kernel/hal/intrinsic/nn/activation.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hlo/rank.h"
#include "libspu/kernel/hlo/reduce.h"

namespace spu::kernel::hlo {

namespace {

spu::Value ReduceLastDimMax(SPUContext* ctx, const spu::Value& input) {
  std::array<spu::Value, 1> inputs = {input};
  auto reducer = [ctx](absl::Span<const spu::Value> lhs,
                       absl::Span<const spu::Value> rhs) {
    return std::vector<spu::Value>{hal::max(ctx, lhs[0], rhs[0])};
  };
  return TreeReduce(ctx, inputs, input.shape().size() - 1, reducer)[0];
}

spu::Value ReduceLastDimSum(SPUContext* ctx, const spu::Value& input) {
  std::array<spu::Value, 1> inputs = {input};
  auto reducer = [ctx](absl::Span<const spu::Value> lhs,
                       absl::Span<const spu::Value> rhs) {
    return std::vector<spu::Value>{hal::add(ctx, lhs[0], rhs[0])};
  };
  return TreeReduce(ctx, inputs, input.shape().size() - 1, reducer)[0];
}

spu::Value BumbleBeeSoftmaxLastDim(SPUContext* ctx,
                                   const spu::Value& logits) {
  // Match BumbleBee's stable softmax:
  //   x_max = max(x, axis=-1, keepdims=True)
  //   nexp = spu_neg_exp(x - x_max)
  //   divisor = sum(nexp, axis=-1, keepdims=True)
  //   return nexp / divisor
  //
  // In the Python path BumbleBee enables
  // enable_optimize_denominator_with_broadcast, which rewrites the final
  // division into one reciprocal per reduced row followed by a broadcasted
  // multiplication. Reproduce that optimization explicitly here.
  auto row_max = ReduceLastDimMax(ctx, logits);
  auto row_max_b = hal::broadcast_to(ctx, row_max, logits.shape());
  auto shifted = hal::sub(ctx, logits, row_max_b);

  auto nexp = hal::intrinsic::nn::f_neg_exp_taylor(ctx, shifted);
  auto divisor = ReduceLastDimSum(ctx, nexp);
  auto inv_divisor = hal::reciprocal(ctx, divisor);
  auto inv_divisor_b =
      hal::broadcast_to(ctx, inv_divisor, nexp.shape());

  return hal::mul(ctx, nexp, inv_divisor_b);
}

}  // namespace

CryptoMoERouterResult CryptoMoERoute(SPUContext* ctx,
                                     const spu::Value& tokens,
                                     const spu::Value& router_weight,
                                     int64_t top_k) {
  SPU_ENFORCE(tokens.isSecret(), "tokens must be secret");
  SPU_ENFORCE(router_weight.isSecret(), "router_weight must be secret");
  SPU_ENFORCE(tokens.isFxp(), "tokens must have fixed-point dtype");
  SPU_ENFORCE(router_weight.isFxp(),
              "router_weight must have fixed-point dtype");

  SPU_ENFORCE(tokens.shape().size() == 2,
              "tokens must have shape [m, d]");
  SPU_ENFORCE(router_weight.shape().size() == 2,
              "router_weight must have shape [d, n]");
  SPU_ENFORCE(tokens.shape()[0] > 0,
              "number of tokens must be positive");
  SPU_ENFORCE(tokens.shape()[1] > 0,
              "token hidden dimension must be positive");
  SPU_ENFORCE(router_weight.shape()[0] == tokens.shape()[1],
              "router_weight first dimension must match token hidden "
              "dimension, got d={} and weight rows={}",
              tokens.shape()[1], router_weight.shape()[0]);

  const int64_t num_experts = router_weight.shape()[1];
  SPU_ENFORCE(num_experts > 0, "number of experts must be positive");
  SPU_ENFORCE(top_k > 0 && top_k <= num_experts,
              "top_k must be in [1, n], got top_k={}, n={}", top_k,
              num_experts);

  // Gate Routing, Equation (1):
  //   G(x) = Softmax(Linear(x)).
  // The gate linear layer maps [m, d] x [d, n] -> [m, n].
  auto logits = hal::matmul(ctx, tokens, router_weight);
  auto probabilities = BumbleBeeSoftmaxLastDim(ctx, logits);

  //   [[W]], [[K]] = Pi_topk([[G(x)]], k).
  // HLO TopK operates along the last dimension, so for [m, n] it performs
  // an independent top-k selection over the n experts for every token.
  auto topk_out = TopK(ctx, probabilities, top_k,
                       /*k_hi=*/-1,
                       /*largest=*/true,
                       /*value_only=*/false);
  SPU_ENFORCE(topk_out.size() == 2,
              "CryptoMoE gate routing requires TopK indices");

  auto routing_weights = topk_out[0];
  auto routing_indices = topk_out[1];

  SPU_ENFORCE(routing_weights.isSecret(),
              "routing weights must remain secret");
  SPU_ENFORCE(routing_indices.isSecret(),
              "routing indices must remain secret");

  return {routing_weights, routing_indices};
}

}  // namespace spu::kernel::hlo
