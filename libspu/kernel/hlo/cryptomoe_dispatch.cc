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

#include "libspu/kernel/hlo/cryptomoe_dispatch.h"

#include <limits>

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hlo/rank.h"

namespace spu::kernel::hlo {

CryptoMoEDispatchResult CryptoMoEDispatchWithAux(
    SPUContext* ctx, const spu::Value& routing_indices,
    const spu::Value& routing_weights, const spu::Value& tokens,
    int64_t expert_id, int64_t capacity) {
  SPU_ENFORCE(routing_indices.isSecret(),
              "routing_indices must be secret");
  SPU_ENFORCE(routing_weights.isSecret(),
              "routing_weights must be secret");
  SPU_ENFORCE(tokens.isSecret(), "tokens must be secret");

  SPU_ENFORCE(routing_indices.dtype() == DT_I32 ||
                  routing_indices.dtype() == DT_I64,
              "routing_indices must have DT_I32 or DT_I64 dtype");
  SPU_ENFORCE(routing_weights.isFxp(),
              "routing_weights must have fixed-point dtype");
  SPU_ENFORCE(tokens.isFxp(), "tokens must have fixed-point dtype");

  SPU_ENFORCE(routing_indices.shape().size() == 2,
              "routing_indices must have shape [m, k]");
  SPU_ENFORCE(routing_weights.shape() == routing_indices.shape(),
              "routing_weights must have the same shape as routing_indices");
  SPU_ENFORCE(tokens.shape().size() == 2,
              "tokens must have shape [m, d]");

  const int64_t num_tokens = routing_indices.shape()[0];
  const int64_t routed_experts_per_token = routing_indices.shape()[1];

  SPU_ENFORCE(expert_id >= 0, "expert_id must be non-negative");
  SPU_ENFORCE(num_tokens > 0, "number of tokens must be positive");
  SPU_ENFORCE(routed_experts_per_token > 0,
              "number of routed experts per token must be positive");
  SPU_ENFORCE(tokens.shape()[0] == num_tokens,
              "tokens and routing tensors must have the same token count");
  SPU_ENFORCE(capacity > 0 && capacity <= num_tokens,
              "capacity must be in [1, m], got capacity={}, m={}", capacity,
              num_tokens);

  const int64_t flattened_size =
      num_tokens * routed_experts_per_token;

  auto make_index_constant =
      [&](int64_t value, DataType dtype, const Shape& shape) {
        if (dtype == DT_I32) {
          SPU_ENFORCE(value <= std::numeric_limits<int32_t>::max(),
                      "index constant {} does not fit DT_I32", value);
          return hal::constant(ctx, static_cast<int32_t>(value), dtype, shape);
        }
        SPU_ENFORCE(dtype == DT_I64,
                    "routing index dtype must be DT_I32 or DT_I64");
        return hal::constant(ctx, value, dtype, shape);
      };

  // Algorithm 1, line 2: flatten K and W.
  auto k_flat =
      hal::reshape(ctx, routing_indices, {flattened_size});
  auto w_flat =
      hal::reshape(ctx, routing_weights, {flattened_size});

  // Algorithm 1, line 3:
  // [[M_i]]^B = Pi_equal([[K]], i).
  auto expert_ids =
      make_index_constant(expert_id, k_flat.dtype(), k_flat.shape());
  auto mask = hal::equal(ctx, k_flat, expert_ids);

  // Algorithm 1, line 4:
  // [[S_i]] = Pi_mux([[M_i]]^B, [[W]]).
  auto zeros =
      hal::zeros(ctx, w_flat.dtype(), w_flat.shape());
  auto scores = hal::select(ctx, mask, w_flat, zeros);

  // Algorithm 1, line 5:
  // [[K_i]], [[S'_i]] = Pi_topk([[S_i]], t).
  auto topk_out = TopK(ctx, scores, capacity);
  SPU_ENFORCE(topk_out.size() == 2,
              "CryptoMoE dispatch requires TopK indices");
  auto selected_scores = topk_out[0];
  auto selected_indices = topk_out[1];

  // Algorithm 1, line 6:
  // [[K'_i]] = [[K_i]] // k.
  auto routing_k =
      make_index_constant(routed_experts_per_token,
                          selected_indices.dtype(),
                          selected_indices.shape());
  auto token_indices =
      hal::div(ctx, selected_indices, routing_k);

  // Algorithm 1, line 7:
  // [[O_i]][r][j] = Pi_equal([[K'_i]][r], j).
  auto token_candidates =
      hal::iota(ctx, selected_indices.dtype(), num_tokens);

  auto token_indices_matrix =
      hal::broadcast_to(ctx, token_indices,
                        {capacity, num_tokens}, {0});
  auto token_candidates_matrix =
      hal::broadcast_to(ctx, token_candidates,
                        {capacity, num_tokens}, {1});

  auto onehot =
      hal::equal(ctx, token_indices_matrix, token_candidates_matrix);

  // Algorithm 1, line 8:
  // [[X_i]] = [[O_i]] * [[x]].
  //
  // Pi_equal yields Boolean shares. Convert the one-hot matrix to arithmetic
  // sharing before invoking Cheetah's secret-secret MatMul.
  auto onehot_a = hal::_prefer_a(ctx, onehot);
  auto dispatched = hal::matmul(ctx, onehot_a, tokens);

  return {
    dispatched,
    onehot,
    selected_scores,
};
}

spu::Value CryptoMoEDispatch(SPUContext* ctx,
                             const spu::Value& routing_indices,
                             const spu::Value& routing_weights,
                             const spu::Value& tokens, int64_t expert_id,
                             int64_t capacity) {
  return CryptoMoEDispatchWithAux(ctx, routing_indices, routing_weights, tokens,
                                  expert_id, capacity)
      .tokens;
}

std::vector<spu::Value> CryptoMoEDispatchAll(
    SPUContext* ctx, const spu::Value& routing_indices,
    const spu::Value& routing_weights, const spu::Value& tokens,
    int64_t num_experts, int64_t capacity) {
  SPU_ENFORCE(num_experts > 0,
              "num_experts must be positive, got {}", num_experts);

  std::vector<spu::Value> dispatched;
  dispatched.reserve(num_experts);

  // Algorithm 1, line 1: dispatch independently for every expert.
  for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    dispatched.push_back(CryptoMoEDispatch(
        ctx, routing_indices, routing_weights, tokens, expert_id, capacity));
  }

  return dispatched;
}

}  // namespace spu::kernel::hlo
