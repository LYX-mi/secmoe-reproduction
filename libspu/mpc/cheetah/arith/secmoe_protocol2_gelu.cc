// Copyright 2026
//
// SecMoE Protocol 2 six-segment quadratic GeLU.

#include "libspu/mpc/cheetah/arith/secmoe_protocol2_gelu.h"

#include <cstddef>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/ring.h"

namespace spu::mpc::cheetah {

Value SecMoEProtocol2GeLU(
    SPUContext* context,
    const Value& x) {
  SPU_ENFORCE(
      context != nullptr);

  SPU_ENFORCE(
      x.isSecret());

  SPU_ENFORCE(
      x.isFxp());

  // Public breakpoints:
  //
  // (-inf, -5), [-5, -3), [-3, -1),
  // [-1, 1), [1, 3), [3, inf)
  //
  // Exact boundary ownership depends on f_less semantics
  // and will be covered by a separate boundary test.
  static constexpr float
      breakpoints[5] = {
          -5.0F,
          -3.0F,
          -1.0F,
          1.0F,
          3.0F,
      };

  std::vector<Value> less_than;
  less_than.reserve(5);

  for (const float breakpoint :
       breakpoints) {
    less_than.emplace_back(
        kernel::hal::f_less(
            context,
            x,
            kernel::hal::constant(
                context,
                breakpoint,
                x.dtype(),
                x.shape())));
  }

  // Consecutive XOR converts the monotone comparison
  // vector into six mutually exclusive segment bits.
  const auto one =
      kernel::hal::_constant(
          context,
          1,
          x.shape());

  std::vector<Value> segment_bits;
  segment_bits.reserve(6);

  segment_bits.emplace_back(
      less_than[0]);

  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          less_than[1],
          less_than[0]));

  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          less_than[2],
          less_than[1]));

  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          less_than[3],
          less_than[2]));

  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          less_than[4],
          less_than[3]));

  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          less_than[4],
          one));

  // Coefficients are ordered as:
  //
  // {constant, linear, quadratic}.
  static constexpr float
      coefficients[6][3] = {
          {
              0.0F,
              0.0F,
              0.0F,
          },
          {
              -0.02986296F,
              -0.01380208F,
              -0.00158297F,
          },
          {
              -0.36497047F,
              -0.23581369F,
              -0.03840320F,
          },
          {
              0.00485947F,
              0.50000716F,
              0.34826040F,
          },
          {
              -0.36491015F,
              1.23575599F,
              -0.03839009F,
          },
          {
              0.0F,
              1.0F,
              0.0F,
          },
      };

  const auto zero =
      kernel::hal::constant(
          context,
          0.0F,
          x.dtype(),
          x.shape());

  auto select_coefficient =
      [&](size_t coefficient_index) {
        Value selected = zero;

        for (size_t segment = 0;
             segment < 6;
             ++segment) {
          const auto public_coefficient =
              kernel::hal::constant(
                  context,
                  coefficients[segment]
                              [coefficient_index],
                  x.dtype(),
                  x.shape());

          auto selected_term =
              kernel::hal::_mux(
                  context,
                  segment_bits[segment],
                  public_coefficient,
                  zero)
                  .setDtype(
                      x.dtype());

          selected =
              kernel::hal::f_add(
                  context,
                  selected,
                  selected_term);
        }

        return selected;
      };

  const auto selected_constant =
      select_coefficient(0);

  const auto selected_linear =
      select_coefficient(1);

  const auto selected_quadratic =
      select_coefficient(2);

  const auto x_squared =
      kernel::hal::f_square(
          context,
          x);

  const auto quadratic_term =
      kernel::hal::f_mul(
          context,
          x_squared,
          selected_quadratic);

  const auto linear_term =
      kernel::hal::f_mul(
          context,
          x,
          selected_linear);

  return kernel::hal::f_add(
      context,
      kernel::hal::f_add(
          context,
          quadratic_term,
          linear_term),
      selected_constant);
}

}  // namespace spu::mpc::cheetah
