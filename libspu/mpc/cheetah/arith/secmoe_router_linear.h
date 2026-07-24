// Copyright 2026
//
// SecMoE private router linear layer.

#pragma once

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::mpc::cheetah {

// Computes:
//
//   secret_input [tokens, model_dimension]
//   x
//   private_router_weights [model_dimension, number_of_experts]
//
// and returns secret fixed-point routing scores:
//
//   [tokens, number_of_experts]
//
// Both inputs must use the same fixed-point dtype. The underlying HAL
// fixed-point matrix multiplication performs secure truncation after the
// ring matrix multiplication.
Value SecMoERouterLinear(
    SPUContext* context,
    const Value& secret_input,
    const Value& private_router_weights);

}  // namespace spu::mpc::cheetah
