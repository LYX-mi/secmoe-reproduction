// Copyright 2026
//
// Minimal SecMoE arithmetic-share to CHEETAH fixed-point
// Runtime bridge test.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/cheetah/arith/secmoe_protocol2_gelu.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

TEST(
    SecMoENonlinearBridgeTest,
    ArithmeticSharesToFixedPointMultiplication) {
  constexpr FieldType kField =
      FieldType::FM64;

  constexpr int64_t kFractionBits = 18;

  const Shape shape = {2};

  // Clear Q18 values, used only to construct and check
  // the test shares.
  //
  // x = [0.5, -0.25]
  // y = [0.25, 0.5]
  auto clear_x =
      ring_zeros(
          kField,
          shape);

  auto clear_y =
      ring_zeros(
          kField,
          shape);

  clear_x.at<uint64_t>(0) =
      static_cast<uint64_t>(
          int64_t{131072});

  clear_x.at<uint64_t>(1) =
      static_cast<uint64_t>(
          int64_t{-65536});

  clear_y.at<uint64_t>(0) =
      static_cast<uint64_t>(
          int64_t{65536});

  clear_y.at<uint64_t>(1) =
      static_cast<uint64_t>(
          int64_t{131072});

  // Build arbitrary additive shares:
  //
  // x0 + x1 = x mod 2^64
  // y0 + y1 = y mod 2^64
  auto x_share_0 =
      ring_zeros(
          kField,
          shape);

  auto x_share_1 =
      ring_zeros(
          kField,
          shape);

  auto y_share_0 =
      ring_zeros(
          kField,
          shape);

  auto y_share_1 =
      ring_zeros(
          kField,
          shape);

  x_share_0.at<uint64_t>(0) = 5U;
  x_share_0.at<uint64_t>(1) = 7U;

  y_share_0.at<uint64_t>(0) = 11U;
  y_share_0.at<uint64_t>(1) = 13U;

  for (int64_t index = 0;
       index < shape.numel();
       ++index) {
    x_share_1.at<uint64_t>(index) =
        clear_x.at<uint64_t>(index)
        - x_share_0.at<uint64_t>(index);

    y_share_1.at<uint64_t>(index) =
        clear_y.at<uint64_t>(index)
        - y_share_0.at<uint64_t>(index);
  }

  // Each simulated party receives only its own local
  // arithmetic shares.
  auto product_shares =
      utils::simulate(
          2,
          [&](const std::shared_ptr<
                  yacl::link::Context>& link_context)
              -> NdArrayRef {
            RuntimeConfig runtime_config;

            runtime_config.set_protocol(
                ProtocolKind::CHEETAH);

            runtime_config.set_field(
                kField);

            runtime_config.set_fxp_fraction_bits(
                kFractionBits);

            runtime_config
                .mutable_cheetah_2pc_config()
                ->set_enable_mul_lsb_error(
                    true);

            SPUContext context(
                runtime_config,
                link_context);

            Factory::RegisterProtocol(
                &context,
                link_context);

            const NdArrayRef& local_x_ring =
                link_context->Rank() == 0
                    ? x_share_0
                    : x_share_1;

            const NdArrayRef& local_y_ring =
                link_context->Rank() == 0
                    ? y_share_0
                    : y_share_1;

            // This is the actual bridge:
            //
            // local RingTy share
            // -> CHEETAH arithmetic secret-share type.
            auto local_x_secret =
                local_x_ring.as(
                    makeType<AShrTy>(
                        kField));

            auto local_y_secret =
                local_y_ring.as(
                    makeType<AShrTy>(
                        kField));

            // DT_F64 marks the ring values as fixed-point
            // semantic values. The scale is supplied by the
            // RuntimeConfig: 18 fractional bits.
            Value x_value(
                local_x_secret,
                DT_F64);

            Value y_value(
                local_y_secret,
                DT_F64);

            SPU_ENFORCE(
                x_value.isSecret());

            SPU_ENFORCE(
                y_value.isSecret());

            // f_mul performs secret multiplication followed
            // by fixed-point truncation back to Q18.
            auto product =
                kernel::hal::f_mul(
                    &context,
                    x_value,
                    y_value);

            SPU_ENFORCE(
                product.isSecret());

            SPU_ENFORCE_EQ(
                product.dtype(),
                DT_F64);

            return product.data().clone();
          });

  ASSERT_EQ(
      product_shares.size(),
      2U);

  ASSERT_EQ(
      product_shares[0].numel(),
      2);

  ASSERT_EQ(
      product_shares[1].numel(),
      2);

  // Expected Q18 results:
  //
  // 0.5 * 0.25 = 0.125  ->  32768
  // -0.25 * 0.5 = -0.125 -> -32768
  const std::array<uint64_t, 2>
      expected = {
          static_cast<uint64_t>(
              int64_t{32768}),
          static_cast<uint64_t>(
              int64_t{-32768}),
      };

  uint64_t maximum_difference = 0U;

  std::array<uint64_t, 2>
      reconstructed{};

  for (int64_t index = 0;
       index < 2;
       ++index) {
    reconstructed[index] =
        product_shares[0]
            .at<uint64_t>(index)
        + product_shares[1]
              .at<uint64_t>(index);

    const uint64_t forward =
        reconstructed[index]
        - expected[index];

    const uint64_t backward =
        expected[index]
        - reconstructed[index];

    const uint64_t difference =
        std::min(
            forward,
            backward);

    maximum_difference =
        std::max(
            maximum_difference,
            difference);
  }

  std::cout
      << "SECMOE_NONLINEAR_SHARE_BRIDGE"
      << " expected_0="
      << expected[0]
      << " reconstructed_0="
      << reconstructed[0]
      << " expected_1="
      << expected[1]
      << " reconstructed_1="
      << reconstructed[1]
      << " maximum_raw_difference="
      << maximum_difference
      << " maximum_real_error_q18="
      << static_cast<double>(
             maximum_difference)
             / static_cast<double>(
                   1ULL << kFractionBits)
      << std::endl;

  // Approximate truncation may introduce a few low bits.
  EXPECT_LE(
      maximum_difference,
      8U);
}


TEST(
    SecMoENonlinearBridgeTest,
    H2AStyleQ36SharesToQ18ThenMultiply) {
  constexpr FieldType kField =
      FieldType::FM64;

  constexpr int64_t kFractionBits = 18;

  const Shape shape = {2};

  // -------------------------------------------------------
  // Simulated linear-layer H2A outputs.
  //
  // These values are Q36 because they are products of two
  // Q18 inputs:
  //
  // xW1 = [-0.25, 0.09375]
  // xV  = [-0.09375, 0.71875]
  //
  // Q36 encodings:
  //
  // -0.25    * 2^36 = -17179869184
  //  0.09375 * 2^36 =   6442450944
  // -0.09375 * 2^36 =  -6442450944
  //  0.71875 * 2^36 =  49392123904
  // -------------------------------------------------------

  auto clear_w1_q36 =
      ring_zeros(
          kField,
          shape);

  auto clear_v_q36 =
      ring_zeros(
          kField,
          shape);

  clear_w1_q36.at<uint64_t>(0) =
      static_cast<uint64_t>(
          int64_t{-17179869184LL});

  clear_w1_q36.at<uint64_t>(1) =
      static_cast<uint64_t>(
          int64_t{6442450944LL});

  clear_v_q36.at<uint64_t>(0) =
      static_cast<uint64_t>(
          int64_t{-6442450944LL});

  clear_v_q36.at<uint64_t>(1) =
      static_cast<uint64_t>(
          int64_t{49392123904LL});

  // Arbitrary additive shares, matching the form returned
  // by H2A:
  //
  // party0_share + party1_share = clear value mod 2^64.
  auto w1_share_0 =
      ring_zeros(
          kField,
          shape);

  auto w1_share_1 =
      ring_zeros(
          kField,
          shape);

  auto v_share_0 =
      ring_zeros(
          kField,
          shape);

  auto v_share_1 =
      ring_zeros(
          kField,
          shape);

  w1_share_0.at<uint64_t>(0) = 101U;
  w1_share_0.at<uint64_t>(1) = 103U;

  v_share_0.at<uint64_t>(0) = 107U;
  v_share_0.at<uint64_t>(1) = 109U;

  for (int64_t index = 0;
       index < shape.numel();
       ++index) {
    w1_share_1.at<uint64_t>(index) =
        clear_w1_q36.at<uint64_t>(index)
        - w1_share_0.at<uint64_t>(index);

    v_share_1.at<uint64_t>(index) =
        clear_v_q36.at<uint64_t>(index)
        - v_share_0.at<uint64_t>(index);
  }

  auto product_shares =
      utils::simulate(
          2,
          [&](const std::shared_ptr<
                  yacl::link::Context>& link_context)
              -> NdArrayRef {
            RuntimeConfig runtime_config;

            runtime_config.set_protocol(
                ProtocolKind::CHEETAH);

            runtime_config.set_field(
                kField);

            runtime_config.set_fxp_fraction_bits(
                kFractionBits);

            runtime_config
                .mutable_cheetah_2pc_config()
                ->set_enable_mul_lsb_error(
                    true);

            SPUContext context(
                runtime_config,
                link_context);

            Factory::RegisterProtocol(
                &context,
                link_context);

            const NdArrayRef& local_w1_q36 =
                link_context->Rank() == 0
                    ? w1_share_0
                    : w1_share_1;

            const NdArrayRef& local_v_q36 =
                link_context->Rank() == 0
                    ? v_share_0
                    : v_share_1;

            auto local_w1_secret =
                local_w1_q36.as(
                    makeType<AShrTy>(
                        kField));

            auto local_v_secret =
                local_v_q36.as(
                    makeType<AShrTy>(
                        kField));

            // The raw H2A outputs are Q36 integers.
            // They must not yet be labelled as DT_F64/Q18.
            Value raw_w1_q36(
                local_w1_secret,
                DT_I64);

            Value raw_v_q36(
                local_v_secret,
                DT_I64);

            // Securely truncate both arithmetic shares:
            //
            // Q36 >> 18 = Q18.
            auto w1_q18 =
                kernel::hal::_trunc(
                    &context,
                    raw_w1_q36,
                    kFractionBits,
                    SignType::Unknown);

            auto v_q18 =
                kernel::hal::_trunc(
                    &context,
                    raw_v_q36,
                    kFractionBits,
                    SignType::Unknown);

            w1_q18.setDtype(DT_F64);
            v_q18.setDtype(DT_F64);

            SPU_ENFORCE(
                w1_q18.isSecret());

            SPU_ENFORCE(
                v_q18.isSecret());

            // Secure element-wise multiplication:
            //
            // Q18 * Q18 -> Q36 -> internal truncation
            // -> Q18.
            auto product_q18 =
                kernel::hal::f_mul(
                    &context,
                    w1_q18,
                    v_q18);

            SPU_ENFORCE(
                product_q18.isSecret());

            SPU_ENFORCE_EQ(
                product_q18.dtype(),
                DT_F64);

            return product_q18
                .data()
                .clone();
          });

  ASSERT_EQ(
      product_shares.size(),
      2U);

  // Expected real results:
  //
  // (-0.25) * (-0.09375)
  //     = 0.0234375
  //     = 6144 / 2^18
  //
  // 0.09375 * 0.71875
  //     = 0.0673828125
  //     = 17664 / 2^18
  const std::array<uint64_t, 2>
      expected_q18 = {
          6144U,
          17664U,
      };

  std::array<uint64_t, 2>
      reconstructed{};

  uint64_t maximum_difference = 0U;

  for (int64_t index = 0;
       index < shape.numel();
       ++index) {
    reconstructed[index] =
        product_shares[0]
            .at<uint64_t>(index)
        + product_shares[1]
              .at<uint64_t>(index);

    const uint64_t forward =
        reconstructed[index]
        - expected_q18[index];

    const uint64_t backward =
        expected_q18[index]
        - reconstructed[index];

    const uint64_t difference =
        std::min(
            forward,
            backward);

    maximum_difference =
        std::max(
            maximum_difference,
            difference);
  }

  std::cout
      << "SECMOE_Q36_TRUNCATION_BRIDGE"
      << " expected_0="
      << expected_q18[0]
      << " reconstructed_0="
      << reconstructed[0]
      << " expected_1="
      << expected_q18[1]
      << " reconstructed_1="
      << reconstructed[1]
      << " maximum_raw_difference="
      << maximum_difference
      << " maximum_real_error_q18="
      << static_cast<double>(
             maximum_difference)
             / static_cast<double>(
                   1ULL << kFractionBits)
      << std::endl;

  // Diagnostic allowance for two secure truncations and
  // the multiplication truncation.
  EXPECT_LE(
      maximum_difference,
      64U);
}



TEST(
    SecMoENonlinearBridgeTest,
    SecureProtocol2PiecewiseQuadraticGeLU) {
  constexpr FieldType kField =
      FieldType::FM64;

  constexpr int64_t kFractionBits = 18;

  // Cover the interior of all six segments.
  // Exact Q18 breakpoint ownership is verified separately
  // by the segment-one-hot boundary test.
  static constexpr double clear_inputs[8] = {
      -6.0,
      -4.0,
      -2.0,
      -0.25,
      0.0,
      0.09375,
      2.0,
      4.0,
  };

  constexpr int64_t kInputCount = 8;
  const Shape shape = {kInputCount};

  auto clear_x =
      ring_zeros(
          kField,
          shape);

  for (int64_t index = 0;
       index < kInputCount;
       ++index) {
    const int64_t encoded =
        static_cast<int64_t>(
            std::llround(
                clear_inputs[index]
                * static_cast<double>(
                      1ULL
                      << kFractionBits)));

    clear_x.at<uint64_t>(index) =
        static_cast<uint64_t>(
            encoded);
  }

  // Arbitrary two-party additive shares.
  auto x_share_0 =
      ring_zeros(
          kField,
          shape);

  auto x_share_1 =
      ring_zeros(
          kField,
          shape);

  for (int64_t index = 0;
       index < kInputCount;
       ++index) {
    x_share_0.at<uint64_t>(index) =
        static_cast<uint64_t>(
            17 + index * 13);

    x_share_1.at<uint64_t>(index) =
        clear_x.at<uint64_t>(index)
        - x_share_0.at<uint64_t>(index);
  }

  auto gelu_shares =
      utils::simulate(
          2,
          [&](const std::shared_ptr<
                  yacl::link::Context>&
                  link_context)
              -> NdArrayRef {
            RuntimeConfig runtime_config;

            runtime_config.set_protocol(
                ProtocolKind::CHEETAH);

            runtime_config.set_field(
                kField);

            runtime_config.set_fxp_fraction_bits(
                kFractionBits);

            runtime_config
                .mutable_cheetah_2pc_config()
                ->set_enable_mul_lsb_error(
                    true);

            SPUContext context(
                runtime_config,
                link_context);

            Factory::RegisterProtocol(
                &context,
                link_context);

            const NdArrayRef& local_share =
                link_context->Rank() == 0
                    ? x_share_0
                    : x_share_1;

            auto local_secret =
                local_share.as(
                    makeType<AShrTy>(
                        kField));

            Value x_value(
                local_secret,
                DT_F64);

            auto gelu_value =
                SecMoEProtocol2GeLU(
                    &context,
                    x_value);

            SPU_ENFORCE(
                gelu_value.isSecret());

            SPU_ENFORCE_EQ(
                gelu_value.dtype(),
                DT_F64);

            return gelu_value
                .data()
                .clone();
          });

  ASSERT_EQ(
      gelu_shares.size(),
      2U);

  ASSERT_EQ(
      gelu_shares[0].numel(),
      kInputCount);

  ASSERT_EQ(
      gelu_shares[1].numel(),
      kInputCount);

  static constexpr double coefficients[6][3] = {
      {0.0, 0.0, 0.0},
      {-0.02986296, -0.01380208, -0.00158297},
      {-0.36497047, -0.23581369, -0.03840320},
      {0.00485947, 0.50000716, 0.34826040},
      {-0.36491015, 1.23575599, -0.03839009},
      {0.0, 1.0, 0.0},
  };

  auto segment_for =
      [](double value) {
        if (value <= -5.0) {
          return 0;
        }

        if (value <= -3.0) {
          return 1;
        }

        if (value <= -1.0) {
          return 2;
        }

        if (value <= 1.0) {
          return 3;
        }

        if (value <= 3.0) {
          return 4;
        }

        return 5;
      };

  uint64_t maximum_difference = 0U;

  for (int64_t index = 0;
       index < kInputCount;
       ++index) {
    const int segment =
        segment_for(
            clear_inputs[index]);

    const double expected_real =
        coefficients[segment][0]
        + coefficients[segment][1]
              * clear_inputs[index]
        + coefficients[segment][2]
              * clear_inputs[index]
              * clear_inputs[index];

    const int64_t expected_signed =
        static_cast<int64_t>(
            std::llround(
                expected_real
                * static_cast<double>(
                      1ULL
                      << kFractionBits)));

    const uint64_t expected =
        static_cast<uint64_t>(
            expected_signed);

    const uint64_t reconstructed =
        gelu_shares[0]
            .at<uint64_t>(index)
        + gelu_shares[1]
              .at<uint64_t>(index);

    const uint64_t forward =
        reconstructed
        - expected;

    const uint64_t backward =
        expected
        - reconstructed;

    const uint64_t difference =
        std::min(
            forward,
            backward);

    maximum_difference =
        std::max(
            maximum_difference,
            difference);

    std::cout
        << "SECMOE_PROTOCOL2_GELU_POINT"
        << " index=" << index
        << " x=" << clear_inputs[index]
        << " segment=" << segment
        << " expected=" << expected
        << " reconstructed=" << reconstructed
        << " raw_difference=" << difference
        << std::endl;
  }

  std::cout
      << "SECMOE_PROTOCOL2_GELU_SUMMARY"
      << " points="
      << kInputCount
      << " maximum_raw_difference="
      << maximum_difference
      << " maximum_real_error_q18="
      << static_cast<double>(
             maximum_difference)
             / static_cast<double>(
                   1ULL << kFractionBits)
      << std::endl;

  // Includes comparison, coefficient MUX, square and two
  // secure fixed-point multiplications.
  EXPECT_LE(
      maximum_difference,
      1024U);
}


}  // namespace spu::mpc::cheetah::test
