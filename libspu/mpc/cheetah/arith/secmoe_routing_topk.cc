// Copyright 2026
//
// SecMoE secure Top-1 routing bridge.

#include "libspu/mpc/cheetah/arith/secmoe_routing_topk.h"

#include <utility>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hlo/rank.h"

namespace spu::mpc::cheetah {

SecMoETop1RoutingResult SecMoESecretTop1Routing(
    SPUContext* context,
    const Value& secret_scores,
    int64_t number_of_experts) {
  SPU_ENFORCE(
      context != nullptr,
      "SPU context must not be null");

  SPU_ENFORCE(
      secret_scores.isSecret(),
      "routing scores must be secret");

  SPU_ENFORCE(
      secret_scores.isFxp(),
      "routing scores must be fixed-point values");

  SPU_ENFORCE(
      secret_scores.shape().ndim() == 1U,
      "routing scores must be one-dimensional");

  SPU_ENFORCE(
      number_of_experts > 0,
      "number_of_experts must be positive");

  SPU_ENFORCE(
      secret_scores.numel() == number_of_experts,
      "routing score count must equal number_of_experts");

  // Reuse OpenBumbleBee's secure TopK implementation.
  //
  // result[0]: secret Top-1 value, shape {1}
  // result[1]: secret Top-1 index, shape {1}
  auto topk_result =
      kernel::hlo::TopK(
          context,
          secret_scores,
          1,
          1,
          true,
          false);

  SPU_ENFORCE(
      topk_result.size() == 2U,
      "TopK must return value and index");

  auto top_value =
      std::move(topk_result[0]);

  auto top_index =
      std::move(topk_result[1]);

  SPU_ENFORCE(
      top_value.isSecret(),
      "Top-1 value must remain secret");

  SPU_ENFORCE(
      top_index.isSecret(),
      "Top-1 index must remain secret");

  SPU_ENFORCE(
      top_value.numel() == 1,
      "Top-1 value must contain one element");

  SPU_ENFORCE(
      top_index.numel() == 1,
      "Top-1 index must contain one element");

  // hal::equal requires both operands to have identical shapes.
  //
  // Expand:
  //   [secret_index]
  //
  // to:
  //   [secret_index, ..., secret_index]
  std::vector<Value> repeated_top_indices;

  repeated_top_indices.reserve(
      number_of_experts);

  for (int64_t expert = 0;
       expert < number_of_experts;
       ++expert) {
    repeated_top_indices.emplace_back(
        top_index.clone());
  }

  auto expanded_top_index =
      kernel::hal::concatenate(
          context,
          repeated_top_indices,
          0);

  // Public expert labels:
  // [0, 1, ..., number_of_experts - 1]
  auto public_expert_indices =
      kernel::hal::iota(
          context,
          top_index.dtype(),
          number_of_experts);

  // Secret integer equality produces a secret DT_I1
  // Boolean one-hot vector.
  auto boolean_one_hot =
      kernel::hal::equal(
          context,
          expanded_top_index,
          public_expert_indices);

  SPU_ENFORCE(
      boolean_one_hot.isSecret(),
      "Boolean one-hot must remain secret");

  SPU_ENFORCE(
      boolean_one_hot.dtype() == DT_I1,
      "Boolean one-hot must use DT_I1");

  SPU_ENFORCE(
      boolean_one_hot.numel() == number_of_experts,
      "Boolean one-hot has invalid length");

  // Reuse the existing OpenBumbleBee B2A bridge.
  //
  // The semantic dtype remains DT_I1, while the storage
  // representation becomes an arithmetic share.
  auto arithmetic_one_hot =
      kernel::hal::_prefer_a(
          context,
          boolean_one_hot);

  SPU_ENFORCE(
      arithmetic_one_hot.isSecret(),
      "Arithmetic one-hot must remain secret");

  SPU_ENFORCE(
      arithmetic_one_hot.dtype() == DT_I1,
      "Arithmetic one-hot must retain DT_I1");

  SPU_ENFORCE(
      arithmetic_one_hot.numel() == number_of_experts,
      "Arithmetic one-hot has invalid length");

  return {
      std::move(top_value),
      std::move(top_index),
      std::move(boolean_one_hot),
      std::move(arithmetic_one_hot),
  };
}

}  // namespace spu::mpc::cheetah
