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

#pragma once

#include <cstdint>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hlo {

struct CryptoMoEDispatchResult {
  // X_i = O_i * x
  spu::Value tokens;

  // O_i, shape [capacity, num_tokens]
  spu::Value onehot;

  // S'_i, selected routing scores, shape [capacity]
  spu::Value scores;
};

// CryptoMoE Algorithm 1: dispatch tokens to one expert.
//
// routing_indices and routing_weights have shape [m, k], where m is the
// number of tokens and k is the number of routed experts per token.
// tokens has shape [m, d].
//
// The function performs Algorithm 1 lines 2-8 for public expert_id and
// capacity t, and returns the dispatched tokens X_i together with O_i and
// S'_i required by Pi_combine.
CryptoMoEDispatchResult CryptoMoEDispatchWithAux(
    SPUContext* ctx, const spu::Value& routing_indices,
    const spu::Value& routing_weights, const spu::Value& tokens,
    int64_t expert_id, int64_t capacity);

spu::Value CryptoMoEDispatch(SPUContext* ctx,
                             const spu::Value& routing_indices,
                             const spu::Value& routing_weights,
                             const spu::Value& tokens, int64_t expert_id,
                             int64_t capacity);

// CryptoMoE Algorithm 1, line 1: dispatch tokens to every expert.
//
// Returns one secret [t, d] dispatched token matrix for each expert,
// ordered by expert id in [0, num_experts).
std::vector<spu::Value> CryptoMoEDispatchAll(
    SPUContext* ctx, const spu::Value& routing_indices,
    const spu::Value& routing_weights, const spu::Value& tokens,
    int64_t num_experts, int64_t capacity);

}  // namespace spu::kernel::hlo
