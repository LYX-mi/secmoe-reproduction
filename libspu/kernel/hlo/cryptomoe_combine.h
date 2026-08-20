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

#include "absl/types/span.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hlo {

// CryptoMoE Algorithm 2: securely combine all expert outputs.
//
// expert_outputs[i] has shape [t, d], onehots[i] has shape [t, m], and
// scores[i] has shape [t]. All inputs are secret shared.
//
// Returns the secret MoE layer output with shape [m, d].
spu::Value CryptoMoECombine(
    SPUContext* ctx, absl::Span<const spu::Value> expert_outputs,
    absl::Span<const spu::Value> onehots,
    absl::Span<const spu::Value> scores);

}  // namespace spu::kernel::hlo
