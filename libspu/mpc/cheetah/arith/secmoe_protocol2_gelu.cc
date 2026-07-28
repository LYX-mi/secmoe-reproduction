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

std::vector<Value> SecMoEProtocol2SegmentBits(
    SPUContext* context,
    const Value& x) {
  SPU_ENFORCE(
      context != nullptr);

  SPU_ENFORCE(
      x.isSecret());

  SPU_ENFORCE(
      x.isFxp());

  // Protocol 2 right-closed intervals:
  //
  // (-inf, -5], (-5, -3], (-3, -1],
  // (-1, 1], (1, 3], (3, inf).
  //
  // Each predicate is:
  //
  //   breakpoint < x
  //
  // so equality remains in the segment on the left.
  static constexpr float
      breakpoints[5] = {
          -5.0F,
          -3.0F,
          -1.0F,
          1.0F,
          3.0F,
      };

  std::vector<Value> greater_than;
  greater_than.reserve(5);

  for (const float breakpoint :
       breakpoints) {
    const auto public_breakpoint =
        kernel::hal::constant(
            context,
            breakpoint,
            x.dtype(),
            x.shape());

    greater_than.emplace_back(
        kernel::hal::f_less(
            context,
            public_breakpoint,
            x));
  }

  const auto one =
      kernel::hal::_constant(
          context,
          1,
          x.shape());

  std::vector<Value> segment_bits;
  segment_bits.reserve(6);

  // x <= -5
  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          greater_than[0],
          one));

  // -5 < x <= -3
  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          greater_than[0],
          greater_than[1]));

  // -3 < x <= -1
  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          greater_than[1],
          greater_than[2]));

  // -1 < x <= 1
  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          greater_than[2],
          greater_than[3]));

  // 1 < x <= 3
  segment_bits.emplace_back(
      kernel::hal::_xor(
          context,
          greater_than[3],
          greater_than[4]));

  // 3 < x
  segment_bits.emplace_back(
      greater_than[4]);

  // The low-level XOR operator returns raw ring values.
  // Mark every segment predicate explicitly as a Boolean value
  // so reveal/dump and later Boolean operations see DT_I1.
  for (auto& segment_bit : segment_bits) {
    segment_bit = segment_bit.setDtype(DT_I1);
  }

  return segment_bits;
}

Value SecMoEProtocol2GeLU(
    SPUContext* context,
    const Value& x) {
  auto segment_bits =
      SecMoEProtocol2SegmentBits(
          context,
          x);

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
