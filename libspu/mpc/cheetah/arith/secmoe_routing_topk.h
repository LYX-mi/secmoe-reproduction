// Copyright 2026
//
// SecMoE secure Top-1 routing bridge.

#pragma once

#include <cstdint>

#include "libspu/core/value.h"

namespace spu {

class SPUContext;

}  // namespace spu

namespace spu::mpc::cheetah {

struct SecMoETop1RoutingResult {
  // Secret fixed-point Top-1 value, shape {1}.
  Value top_value;

  // Secret integer Top-1 index, shape {1}.
  Value top_index;

  // Secret DT_I1 Boolean one-hot.
  Value boolean_one_hot;

  // Secret DT_I1 arithmetic one-hot after B2A.
  Value arithmetic_one_hot;
};

// secret routing scores
// -> secure Top-1
// -> Boolean one-hot
// -> arithmetic one-hot through B2A
SecMoETop1RoutingResult SecMoESecretTop1Routing(
    SPUContext* context,
    const Value& secret_scores,
    int64_t number_of_experts);

}  // namespace spu::mpc::cheetah
