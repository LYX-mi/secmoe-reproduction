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

#include "libspu/kernel/hlo/cryptomoe_combine.h"

#include <cstddef>

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hlo {

spu::Value CryptoMoECombine(
    SPUContext* ctx, absl::Span<const spu::Value> expert_outputs,
    absl::Span<const spu::Value> onehots,
    absl::Span<const spu::Value> scores) {
  SPU_ENFORCE(!expert_outputs.empty(), "expert_outputs must not be empty");
  SPU_ENFORCE(expert_outputs.size() == onehots.size(),
              "expert_outputs and onehots size mismatch: {} vs {}",
              expert_outputs.size(), onehots.size());
  SPU_ENFORCE(expert_outputs.size() == scores.size(),
              "expert_outputs and scores size mismatch: {} vs {}",
              expert_outputs.size(), scores.size());

  const auto& first_output = expert_outputs.front();
  const auto& first_onehot = onehots.front();
  const auto& first_scores = scores.front();

  SPU_ENFORCE(first_output.shape().ndim() == 2,
              "expert output must be rank 2");
  SPU_ENFORCE(first_onehot.shape().ndim() == 2, "onehot must be rank 2");
  SPU_ENFORCE(first_scores.shape().ndim() == 1, "scores must be rank 1");

  const int64_t capacity = first_output.shape()[0];
  const int64_t hidden_dim = first_output.shape()[1];
  const int64_t num_tokens = first_onehot.shape()[1];

  auto combine_one_expert = [&](size_t i) {
    const auto& expert_output = expert_outputs[i];
    const auto& onehot = onehots[i];
    const auto& selected_scores = scores[i];

    SPU_ENFORCE(expert_output.isSecret(), "expert output must be secret");
    SPU_ENFORCE(onehot.isSecret(), "onehot must be secret");
    SPU_ENFORCE(selected_scores.isSecret(), "scores must be secret");
    SPU_ENFORCE(expert_output.isFxp(), "expert output must be fixed-point");
    SPU_ENFORCE(selected_scores.isFxp(), "scores must be fixed-point");
    SPU_ENFORCE(onehot.dtype() == DT_I1, "onehot must have DT_I1 dtype");

    SPU_ENFORCE(expert_output.shape() == Shape({capacity, hidden_dim}),
                "expert output shape mismatch: expected [{}, {}], got {}",
                capacity, hidden_dim, expert_output.shape());
    SPU_ENFORCE(onehot.shape() == Shape({capacity, num_tokens}),
                "onehot shape mismatch: expected [{}, {}], got {}", capacity,
                num_tokens, onehot.shape());
    SPU_ENFORCE(selected_scores.shape() == Shape({capacity}),
                "scores shape mismatch: expected [{}], got {}", capacity,
                selected_scores.shape());

    // Algorithm 2, line 2: transpose O_i from [t, m] to [m, t].
    auto onehot_t = hal::transpose(ctx, onehot, Axes{1, 0});

    // Algorithm 2, line 3: broadcast S'_i from [t] to [m, t], then bind each
    // selected score to its one-hot token position. Keeping O_i as a 1-bit
    // Boolean share lets Cheetah dispatch this multiplication to MulA1B.
    auto scores_matrix =
        hal::broadcast_to(ctx, selected_scores, {num_tokens, capacity}, {1});
    auto scored_onehot = hal::mul(ctx, scores_matrix, onehot_t);

    // Algorithm 2, line 4: y'_Ei = R_i * y_Ei.
    return hal::matmul(ctx, scored_onehot, expert_output);
  };

  auto output = combine_one_expert(0);

  // Algorithm 2, lines 5-6: sum all expert contributions.
  for (size_t i = 1; i < expert_outputs.size(); ++i) {
    output = hal::add(ctx, output, combine_one_expert(i));
  }

  return output;
}

}  // namespace spu::kernel::hlo
