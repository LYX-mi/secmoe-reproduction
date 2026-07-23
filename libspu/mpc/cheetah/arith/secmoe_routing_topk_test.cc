// Copyright 2026
//
// SecMoE routing bridge:
//
// secret routing scores
// -> secure Top-1
// -> Boolean one-hot shares
// -> arithmetic one-hot shares via B2A.

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/rank.h"
#include "libspu/mpc/cheetah/arith/secmoe_routing_topk.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

TEST(
    SecMoERoutingTopKTest,
    SecretScoresToBooleanOneHotAndB2A) {
  constexpr FieldType kField =
      FieldType::FM64;

  constexpr int64_t kFractionBits = 18;
  constexpr int64_t kNumberOfExperts = 8;
  constexpr int64_t kExpectedExpert = 3;

  const Shape score_shape = {
      kNumberOfExperts,
  };

  // Routing scores:
  //
  // [0.10, -0.20, 0.35, 0.80,
  //  0.05,  0.40, -0.10, 0.25]
  //
  // The unique maximum is expert 3.
  //
  // Values below are Q18 encodings.
  static constexpr std::array<
      int64_t,
      kNumberOfExperts>
      clear_scores_q18 = {
          26214,
          -52429,
          91750,
          209715,
          13107,
          104858,
          -26214,
          65536,
      };

  auto clear_scores =
      ring_zeros(
          kField,
          score_shape);

  auto score_share_0 =
      ring_zeros(
          kField,
          score_shape);

  auto score_share_1 =
      ring_zeros(
          kField,
          score_shape);

  for (int64_t index = 0;
       index < kNumberOfExperts;
       ++index) {
    clear_scores.at<uint64_t>(index) =
        static_cast<uint64_t>(
            clear_scores_q18[index]);

    // Deterministic arbitrary additive share held by
    // party 0. It reveals nothing without party 1.
    score_share_0.at<uint64_t>(index) =
        static_cast<uint64_t>(
            1009 + index * 37);

    score_share_1.at<uint64_t>(index) =
        clear_scores.at<uint64_t>(index)
        - score_share_0.at<uint64_t>(index);
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

        const NdArrayRef& local_score_share =
            link_context->Rank() == 0
                ? score_share_0
                : score_share_1;

        // Wrap each party's local additive share as a
        // CHEETAH secret fixed-point Value.
        Value secret_scores(
            local_score_share.as(
                makeType<AShrTy>(
                    kField)),
            DT_F64);

        ASSERT_TRUE(
            secret_scores.isSecret());

        ASSERT_TRUE(
            secret_scores.storage_type()
                .isa<AShrTy>());

        // Call the reusable SecMoE routing bridge:
        //
        // secret scores
        // -> secure Top-1
        // -> Boolean one-hot
        // -> arithmetic one-hot through B2A.
        auto routing =
            SecMoESecretTop1Routing(
                &context,
                secret_scores,
                kNumberOfExperts);

        const auto& top_value =
            routing.top_value;

        const auto& top_index =
            routing.top_index;

        const auto& boolean_one_hot =
            routing.boolean_one_hot;

        const auto& arithmetic_one_hot =
            routing.arithmetic_one_hot;

        ASSERT_TRUE(
            arithmetic_one_hot.isSecret());

        ASSERT_TRUE(
            arithmetic_one_hot.storage_type()
                .isa<AShrTy>());

        // _prefer_a converts the underlying storage from
        // Boolean share to arithmetic share, while the
        // semantic dtype correctly remains DT_I1.
        //
        // Do not call setDtype(DT_I64): setDtype is not a
        // semantic integer-width conversion. The revealed
        // DT_I1 values can be exported as int64_t directly
        // by dump_public_as<int64_t>.

        // Reveal only inside the test for verification.
        auto revealed_top_value =
            kernel::hal::reveal(
                &context,
                top_value);

        auto revealed_top_index =
            kernel::hal::reveal(
                &context,
                top_index);

        auto revealed_boolean_one_hot =
            kernel::hal::reveal(
                &context,
                boolean_one_hot);

        auto revealed_arithmetic_one_hot =
            kernel::hal::reveal(
                &context,
                arithmetic_one_hot);

        auto public_top_value =
            kernel::hal::dump_public_as<float>(
                &context,
                revealed_top_value);

        auto public_top_index =
            kernel::hal::dump_public_as<int64_t>(
                &context,
                revealed_top_index);

        auto public_boolean_one_hot =
            kernel::hal::dump_public_as<bool>(
                &context,
                revealed_boolean_one_hot);

        auto public_arithmetic_one_hot =
            kernel::hal::dump_public_as<int64_t>(
                &context,
                revealed_arithmetic_one_hot);

        ASSERT_EQ(
            public_top_index.size(),
            1U);

        ASSERT_EQ(
            public_top_index[0],
            kExpectedExpert);

        EXPECT_NEAR(
            public_top_value[0],
            0.80F,
            1.0e-4F);

        int64_t boolean_sum = 0;
        int64_t arithmetic_sum = 0;

        for (int64_t expert = 0;
             expert < kNumberOfExperts;
             ++expert) {
          const bool expected =
              expert == kExpectedExpert;

          EXPECT_EQ(
              static_cast<bool>(
                  public_boolean_one_hot[expert]),
              expected);

          EXPECT_EQ(
              public_arithmetic_one_hot[expert],
              expected ? 1 : 0);

          boolean_sum +=
              public_boolean_one_hot[expert]
                  ? 1
                  : 0;

          arithmetic_sum +=
              public_arithmetic_one_hot[expert];
        }

        EXPECT_EQ(
            boolean_sum,
            1);

        EXPECT_EQ(
            arithmetic_sum,
            1);

        if (link_context->Rank() == 0) {
          std::cout
              << "SECMOE_ROUTING_TOPK_ONEHOT"
              << " top_value="
              << public_top_value[0]
              << " top_index="
              << public_top_index[0]
              << " boolean_sum="
              << boolean_sum
              << " arithmetic_sum="
              << arithmetic_sum
              << " boolean_storage_is_bshare=1"
              << " arithmetic_storage_is_ashare=1"
              << " boolean_onehot=";

          for (int64_t expert = 0;
               expert < kNumberOfExperts;
               ++expert) {
            std::cout
                << (
                    public_boolean_one_hot[expert]
                        ? 1
                        : 0);
          }

          std::cout
              << " arithmetic_onehot=";

          for (int64_t expert = 0;
               expert < kNumberOfExperts;
               ++expert) {
            std::cout
                << public_arithmetic_one_hot[
                       expert];
          }

          std::cout
              << std::endl;
        }
      });
}

}  // namespace spu::mpc::cheetah::test
