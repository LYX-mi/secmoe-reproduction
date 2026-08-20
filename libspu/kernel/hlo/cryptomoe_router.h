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

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hlo {

// Outputs of CryptoMoE gate routing.
//
// Given secret token embeddings [[x]], the router computes
//   [[G(x)]] = Softmax(Linear([[x]]))
// and then selects the top-k experts for every token.
//
// - weights: [[W]], shape [m, k], selected routing scores.
// - indices: [[K]], shape [m, k], selected expert indices.
struct CryptoMoERouterResult {
  spu::Value weights;
  spu::Value indices;
};

// CryptoMoE gate routing:
//   [[W]], [[K]] = TopK(Softmax(Linear([[x]])), k)
//
// Inputs:
// - tokens:        secret [m, d] token embeddings.
// - router_weight: secret [d, n] gate linear-layer weights.
// - top_k:         number of experts selected for each token.
//
// All returned values remain secret-shared.
CryptoMoERouterResult CryptoMoERoute(SPUContext* ctx,
                                     const spu::Value& tokens,
                                     const spu::Value& router_weight,
                                     int64_t top_k);

}  // namespace spu::kernel::hlo
