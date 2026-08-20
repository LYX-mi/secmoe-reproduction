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

#include "libspu/kernel/hlo/cryptomoe_expert.h"

#include <cstdint>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/kernel/hal/intrinsic/nn/activation.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hlo {

std::vector<spu::Value> CryptoMoEExpertCompute(
    SPUContext* ctx, const std::vector<spu::Value>& expert_inputs,
    const spu::Value& gate_weight, const spu::Value& up_weight,
    const spu::Value& down_weight) {
  SPU_ENFORCE(!expert_inputs.empty(), "expert_inputs must not be empty");

  const int64_t num_experts = expert_inputs.size();
  const auto& first = expert_inputs.front();

  SPU_ENFORCE(first.isSecret(), "expert inputs must be secret");
  SPU_ENFORCE(first.isFxp(), "expert inputs must have fixed-point dtype");
  SPU_ENFORCE(first.shape().size() == 2,
              "each expert input must have shape [t, d]");

  const int64_t capacity = first.shape()[0];
  const int64_t hidden_dim = first.shape()[1];
  SPU_ENFORCE(capacity > 0, "expert capacity must be positive");
  SPU_ENFORCE(hidden_dim > 0, "expert hidden dimension must be positive");

  for (int64_t expert = 0; expert < num_experts; ++expert) {
    const auto& input = expert_inputs[expert];
    SPU_ENFORCE(input.isSecret(), "expert input {} must be secret", expert);
    SPU_ENFORCE(input.isFxp(),
                "expert input {} must have fixed-point dtype", expert);
    SPU_ENFORCE(input.dtype() == first.dtype(),
                "all expert inputs must have the same dtype");
    SPU_ENFORCE(input.shape() == first.shape(),
                "all expert inputs must have shape [t, d]");
  }

  SPU_ENFORCE(gate_weight.isPrivate(),
              "gate_weight must be server-private");
  SPU_ENFORCE(up_weight.isPrivate(), "up_weight must be server-private");
  SPU_ENFORCE(down_weight.isPrivate(),
              "down_weight must be server-private");
  SPU_ENFORCE(gate_weight.isFxp() && up_weight.isFxp() &&
                  down_weight.isFxp(),
              "expert weights must have fixed-point dtype");
  SPU_ENFORCE(gate_weight.dtype() == first.dtype() &&
                  up_weight.dtype() == first.dtype() &&
                  down_weight.dtype() == first.dtype(),
              "expert inputs and weights must have the same dtype");

  SPU_ENFORCE(gate_weight.shape().size() == 3,
              "gate_weight must have shape [n, d, h]");
  SPU_ENFORCE(up_weight.shape().size() == 3,
              "up_weight must have shape [n, d, h]");
  SPU_ENFORCE(down_weight.shape().size() == 3,
              "down_weight must have shape [n, h, d]");

  SPU_ENFORCE(gate_weight.shape()[0] == num_experts,
              "gate_weight expert dimension mismatch");
  SPU_ENFORCE(up_weight.shape()[0] == num_experts,
              "up_weight expert dimension mismatch");
  SPU_ENFORCE(down_weight.shape()[0] == num_experts,
              "down_weight expert dimension mismatch");
  SPU_ENFORCE(gate_weight.shape()[1] == hidden_dim,
              "gate_weight input dimension mismatch");
  SPU_ENFORCE(up_weight.shape()[1] == hidden_dim,
              "up_weight input dimension mismatch");

  const int64_t intermediate_dim = gate_weight.shape()[2];
  SPU_ENFORCE(intermediate_dim > 0,
              "expert intermediate dimension must be positive");
  SPU_ENFORCE(up_weight.shape()[2] == intermediate_dim,
              "gate/up intermediate dimensions must match");
  SPU_ENFORCE(down_weight.shape()[1] == intermediate_dim,
              "down_weight intermediate dimension mismatch");
  SPU_ENFORCE(down_weight.shape()[2] == hidden_dim,
              "down_weight output dimension mismatch");

  // Stack n dispatched expert inputs [t, d] into [n, t, d].
  std::vector<spu::Value> batched_inputs;
  batched_inputs.reserve(num_experts);
  for (const auto& input : expert_inputs) {
    batched_inputs.push_back(
        hal::reshape(ctx, input, {1, capacity, hidden_dim}));
  }
  auto x = hal::concatenate(ctx, batched_inputs, 0);

  // SwiGLU:
  //   gate = X W_gate
  //   up   = X W_up
  //   h    = SiLU(gate) * up
  //   y    = h W_down
  auto gate = hal::batch_matmul(ctx, x, gate_weight);
  SPU_ENFORCE(gate.has_value(),
              "CryptoMoE gate projection requires Batch MatMul");

  auto up = hal::batch_matmul(ctx, x, up_weight);
  SPU_ENFORCE(up.has_value(),
              "CryptoMoE up projection requires Batch MatMul");

  auto activated_gate =
      hal::intrinsic::nn::f_seg4_silu(ctx, *gate);
  auto intermediate = hal::mul(ctx, activated_gate, *up);

  auto output = hal::batch_matmul(ctx, intermediate, down_weight);
  SPU_ENFORCE(output.has_value(),
              "CryptoMoE down projection requires Batch MatMul");
  SPU_ENFORCE(output->isSecret(),
              "expert outputs must remain secret");
  SPU_ENFORCE(output->isFxp(),
              "expert outputs must have fixed-point dtype");
  SPU_ENFORCE(output->shape() ==
                  Shape({num_experts, capacity, hidden_dim}),
              "unexpected batched expert output shape");

  // Split [n, t, d] back into n values [t, d] for CryptoMoECombine.
  std::vector<spu::Value> expert_outputs;
  expert_outputs.reserve(num_experts);
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    auto slice = hal::slice(
        ctx, *output, {expert, 0, 0},
        {expert + 1, capacity, hidden_dim});
    expert_outputs.push_back(
        hal::reshape(ctx, slice, {capacity, hidden_dim}));
  }

  return expert_outputs;
}

}  // namespace spu::kernel::hlo
