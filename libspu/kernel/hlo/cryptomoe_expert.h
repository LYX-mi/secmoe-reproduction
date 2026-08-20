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

#include <vector>

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hlo {

// Secure CryptoMoE expert computation using SwiGLU:
//
//   gate = X * W_gate
//   up   = X * W_up
//   h    = SiLU(gate) * up
//   y    = h * W_down
//
// The n experts are evaluated together through CryptoMoE/Cheetah Batch MatMul.
//
// Inputs:
// - expert_inputs: n secret values, each shape [t, d].
// - gate_weight:   server-private [n, d, h].
// - up_weight:     server-private [n, d, h].
// - down_weight:   server-private [n, h, d].
//
// Returns:
// - n secret expert outputs, each shape [t, d].
//
// RuntimeConfig must enable experimental_enable_bmm.
std::vector<spu::Value> CryptoMoEExpertCompute(
    SPUContext* ctx, const std::vector<spu::Value>& expert_inputs,
    const spu::Value& gate_weight, const spu::Value& up_weight,
    const spu::Value& down_weight);

}  // namespace spu::kernel::hlo
