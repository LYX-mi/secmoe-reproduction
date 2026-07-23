// Copyright 2026
//
// SecMoE Protocol 2 six-segment quadratic GeLU.

#pragma once

#include "libspu/core/value.h"

namespace spu {

class SPUContext;

}  // namespace spu

namespace spu::mpc::cheetah {

// SecMoE Protocol 2 functional implementation.
//
// Current form:
//   five separate secret/public comparisons;
//   six-segment coefficient selection;
//   one secret square;
//   quadratic fixed-point evaluation.
//
// The batched-comparison optimization is not included yet.
Value SecMoEProtocol2GeLU(
    SPUContext* context,
    const Value& x);

}  // namespace spu::mpc::cheetah
