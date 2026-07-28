// Copyright 2026
//
// Exact Q18 boundary ownership test for SecMoE Protocol 2.

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/mpc/cheetah/arith/secmoe_protocol2_gelu.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

TEST(
    SecMoEProtocol2BoundaryTest,
    ExactRightClosedQ18Intervals) {
  constexpr FieldType kField =
      FieldType::FM64;

  constexpr int64_t kFractionBits = 18;
  constexpr int64_t kPointCount = 15;
  constexpr int64_t kSegmentCount = 6;

  // For every breakpoint:
  //
  // breakpoint - 1 Q18 LSB,
  // breakpoint,
  // breakpoint + 1 Q18 LSB.
  static constexpr std::array<
      int64_t,
      kPointCount>
      encoded_inputs = {
          -1310721,
          -1310720,
          -1310719,

          -786433,
          -786432,
          -786431,

          -262145,
          -262144,
          -262143,

          262143,
          262144,
          262145,

          786431,
          786432,
          786433,
      };

  static constexpr std::array<
      int64_t,
      kPointCount>
      expected_segments = {
          0,
          0,
          1,

          1,
          1,
          2,

          2,
          2,
          3,

          3,
          3,
          4,

          4,
          4,
          5,
      };

  const Shape shape = {
      kPointCount,
  };

  auto clear_x =
      ring_zeros(
          kField,
          shape);

  auto x_share_0 =
      ring_zeros(
          kField,
          shape);

  auto x_share_1 =
      ring_zeros(
          kField,
          shape);

  for (int64_t index = 0;
       index < kPointCount;
       ++index) {
    clear_x.at<uint64_t>(index) =
        static_cast<uint64_t>(
            encoded_inputs[index]);

    x_share_0.at<uint64_t>(index) =
        static_cast<uint64_t>(
            401 + 29 * index);

    x_share_1.at<uint64_t>(index) =
        clear_x.at<uint64_t>(index)
        - x_share_0.at<uint64_t>(index);
  }

  utils::simulate(
      2,
      [&](const std::shared_ptr<
              yacl::link::Context>& link_context) {
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

        Value secret_x(
            local_share.as(
                makeType<AShrTy>(
                    kField)),
            DT_F64);

        auto segment_bits =
            SecMoEProtocol2SegmentBits(
                &context,
                secret_x);

        ASSERT_EQ(
            segment_bits.size(),
            kSegmentCount);

        std::array<
            std::array<bool, kPointCount>,
            kSegmentCount>
            public_bits{};

        for (int64_t segment = 0;
             segment < kSegmentCount;
             ++segment) {
          ASSERT_TRUE(
              segment_bits[segment]
                  .isSecret());

          auto revealed =
              kernel::hal::reveal(
                  &context,
                  segment_bits[segment]);

          const auto revealed_bits =
              kernel::hal::dump_public_as<bool>(
                  &context,
                  revealed);

          ASSERT_EQ(
              revealed_bits.size(),
              kPointCount);

          for (int64_t index = 0;
               index < kPointCount;
               ++index) {
            public_bits[segment][index] =
                static_cast<bool>(
                    revealed_bits[index]);
          }
        }

        for (int64_t index = 0;
             index < kPointCount;
             ++index) {
          int64_t one_hot_sum = 0;
          int64_t actual_segment = -1;

          for (int64_t segment = 0;
               segment < kSegmentCount;
               ++segment) {
            if (public_bits[segment][index]) {
              ++one_hot_sum;
              actual_segment = segment;
            }
          }

          EXPECT_EQ(
              one_hot_sum,
              1);

          EXPECT_EQ(
              actual_segment,
              expected_segments[index]);

          if (link_context->Rank() == 0) {
            std::cout
                << "SECMOE_PROTOCOL2_BOUNDARY"
                << " encoded_x="
                << encoded_inputs[index]
                << " expected_segment="
                << expected_segments[index]
                << " actual_segment="
                << actual_segment
                << std::endl;
          }
        }
      });
}

}  // namespace spu::mpc::cheetah::test
