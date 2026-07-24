// Copyright 2026

#include "libspu/mpc/cheetah/arith/secmoe_router_linear.h"

#include "libspu/core/prelude.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc::cheetah {

Value SecMoERouterLinear(
    SPUContext* context,
    const Value& secret_input,
    const Value& private_router_weights) {
  SPU_ENFORCE(
      context != nullptr,
      "SecMoERouterLinear requires a valid SPUContext");

  SPU_ENFORCE(
      secret_input.isSecret(),
      "router input must be secret");

  SPU_ENFORCE(
      secret_input.storage_type().isa<AShrTy>(),
      "router input must use CHEETAH arithmetic shares, got {}",
      secret_input.storage_type());

  SPU_ENFORCE(
      private_router_weights.storage_type().isa<Priv2kTy>(),
      "router weights must be a private value, got {}",
      private_router_weights.storage_type());

  SPU_ENFORCE(
      secret_input.isFxp()
          && private_router_weights.isFxp(),
      "router input and weights must be fixed-point values");

  SPU_ENFORCE(
      secret_input.dtype()
          == private_router_weights.dtype(),
      "router input and weights must use the same dtype");

  SPU_ENFORCE(
      secret_input.shape().ndim() == 2,
      "router input must be rank 2, got shape {}",
      secret_input.shape());

  SPU_ENFORCE(
      private_router_weights.shape().ndim() == 2,
      "router weights must be rank 2, got shape {}",
      private_router_weights.shape());

  SPU_ENFORCE(
      secret_input.shape()[1]
          == private_router_weights.shape()[0],
      "router matrix dimension mismatch: input={}, weights={}",
      secret_input.shape(),
      private_router_weights.shape());

  auto routing_scores =
      kernel::hal::matmul(
          context,
          secret_input,
          private_router_weights);

  SPU_ENFORCE(
      routing_scores.isSecret()
          && routing_scores.isFxp(),
      "router output must remain secret fixed point");

  SPU_ENFORCE(
      routing_scores.storage_type().isa<AShrTy>(),
      "router output must use arithmetic shares, got {}",
      routing_scores.storage_type());

  return routing_scores;
}

}  // namespace spu::mpc::cheetah
