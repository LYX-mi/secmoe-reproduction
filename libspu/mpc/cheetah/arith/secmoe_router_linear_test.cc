// Copyright 2026
//
// SecMoE router linear layer:
//
// secret input x
// x private router weights Wg
// -> secret Q18 routing scores
// -> secure Top-1
// -> secret one-hot.

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>

#include "gtest/gtest.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/mpc/cheetah/arith/secmoe_routing_topk.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

TEST(
    SecMoERouterLinearTest,
    SecretInputTimesPrivateWeightsToTop1) {
  constexpr FieldType kField =
      FieldType::FM64;

  constexpr int64_t kFractionBits = 18;
  constexpr int64_t kModelDimension = 3;
  constexpr int64_t kNumberOfExperts = 8;
  constexpr int64_t kWeightOwner = 0;
  constexpr int64_t kExpectedExpert = 3;

  const Shape input_shape = {
      1,
      kModelDimension,
  };

  const Shape weight_shape = {
      kModelDimension,
      kNumberOfExperts,
  };

  // x = [0.5, -0.25, 0.75], encoded in Q18.
  static constexpr std::array<
      int64_t,
      kModelDimension>
      input_q18 = {
          131072,
          -65536,
          196608,
      };

  // Only the first row is nonzero.
  //
  // Since x[0] = 0.5, the resulting routing scores are:
  //
  // [0.10, -0.20, 0.35, 0.80,
  //  0.05,  0.40, -0.10, 0.25]
  //
  // The unique maximum is expert 3.
  static constexpr std::array<
      int64_t,
      kModelDimension * kNumberOfExperts>
      router_weights_q18 = {
          52429,
          -104858,
          183501,
          419430,
          26214,
          209715,
          -52429,
          131072,

          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,

          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
      };

  static constexpr std::array<
      float,
      kNumberOfExperts>
      expected_scores = {
          0.10F,
          -0.20F,
          0.35F,
          0.80F,
          0.05F,
          0.40F,
          -0.10F,
          0.25F,
      };

  auto clear_input =
      ring_zeros(
          kField,
          input_shape);

  auto input_share_0 =
      ring_zeros(
          kField,
          input_shape);

  auto input_share_1 =
      ring_zeros(
          kField,
          input_shape);

  for (int64_t index = 0;
       index < kModelDimension;
       ++index) {
    clear_input.at<uint64_t>(index) =
        static_cast<uint64_t>(
            input_q18[index]);

    const uint64_t party_0_share =
        static_cast<uint64_t>(
            3001 + 97 * index);

    input_share_0.at<uint64_t>(index) =
        party_0_share;

    input_share_1.at<uint64_t>(index) =
        clear_input.at<uint64_t>(index)
        - party_0_share;
  }

  auto clear_router_weights =
      ring_zeros(
          kField,
          weight_shape);

  for (int64_t index = 0;
       index
           < kModelDimension
                 * kNumberOfExperts;
       ++index) {
    clear_router_weights.at<uint64_t>(index) =
        static_cast<uint64_t>(
            router_weights_q18[index]);
  }

  const auto private_weight_type =
      makeType<Priv2kTy>(
          kField,
          kWeightOwner);

  // This follows CHEETAH private-input semantics:
  // the owner stores the actual data, while the other
  // party stores a same-shaped zero placeholder.
  auto owner_router_weights =
      clear_router_weights.as(
          private_weight_type);

  auto non_owner_router_weights =
      ring_zeros(
          kField,
          weight_shape)
          .as(private_weight_type);

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

        const NdArrayRef& local_input_share =
            link_context->Rank() == 0
                ? input_share_0
                : input_share_1;

        const NdArrayRef& local_router_weights =
            link_context->Rank()
                    == kWeightOwner
                ? owner_router_weights
                : non_owner_router_weights;

        Value secret_input(
            local_input_share.as(
                makeType<AShrTy>(
                    kField)),
            DT_F64);

        Value private_router_weights(
            local_router_weights,
            DT_F64);

        ASSERT_TRUE(
            secret_input.isSecret());

        ASSERT_TRUE(
            secret_input.storage_type()
                .isa<AShrTy>());

        ASSERT_TRUE(
            private_router_weights
                .storage_type()
                .isa<Priv2kTy>());

        // DT_F64 x DT_F64 dispatches to f_mmul.
        //
        // f_mmul performs:
        //   _mmul -> secure _trunc -> Q18 output.
        auto routing_scores_2d =
            kernel::hal::matmul(
                &context,
                secret_input,
                private_router_weights);

        ASSERT_TRUE(
            routing_scores_2d.isSecret());

        ASSERT_TRUE(
            routing_scores_2d.isFxp());

        ASSERT_TRUE(
            routing_scores_2d
                .storage_type()
                .isa<AShrTy>());

        ASSERT_EQ(
            routing_scores_2d.shape().ndim(),
            2U);

        ASSERT_EQ(
            routing_scores_2d.shape()[0],
            1);

        ASSERT_EQ(
            routing_scores_2d.shape()[1],
            kNumberOfExperts);

        auto routing_scores =
            kernel::hal::reshape(
                &context,
                routing_scores_2d,
                {
                    kNumberOfExperts,
                });

        auto routing =
            SecMoESecretTop1Routing(
                &context,
                routing_scores,
                kNumberOfExperts);

        ASSERT_TRUE(
            routing.top_index.isSecret());

        ASSERT_TRUE(
            routing.arithmetic_one_hot
                .storage_type()
                .isa<AShrTy>());

        // Reveal only inside this independent test.
        auto public_scores =
            kernel::hal::dump_public_as<float>(
                &context,
                kernel::hal::reveal(
                    &context,
                    routing_scores));

        auto public_top_index =
            kernel::hal::dump_public_as<int64_t>(
                &context,
                kernel::hal::reveal(
                    &context,
                    routing.top_index));

        auto public_top_value =
            kernel::hal::dump_public_as<float>(
                &context,
                kernel::hal::reveal(
                    &context,
                    routing.top_value));

        auto public_one_hot =
            kernel::hal::dump_public_as<int64_t>(
                &context,
                kernel::hal::reveal(
                    &context,
                    routing.arithmetic_one_hot));

        ASSERT_EQ(
            public_scores.size(),
            kNumberOfExperts);

        ASSERT_EQ(
            public_top_index.size(),
            1U);

        ASSERT_EQ(
            public_top_value.size(),
            1U);

        ASSERT_EQ(
            public_one_hot.size(),
            kNumberOfExperts);

        for (int64_t expert = 0;
             expert < kNumberOfExperts;
             ++expert) {
          EXPECT_NEAR(
              public_scores[expert],
              expected_scores[expert],
              5.0e-4F);

          EXPECT_EQ(
              public_one_hot[expert],
              expert == kExpectedExpert
                  ? 1
                  : 0);
        }

        EXPECT_EQ(
            public_top_index[0],
            kExpectedExpert);

        EXPECT_NEAR(
            public_top_value[0],
            0.80F,
            5.0e-4F);

        if (link_context->Rank() == 0) {
          std::cout
              << "SECMOE_ROUTER_LINEAR"
              << " top_index="
              << public_top_index[0]
              << " top_value="
              << public_top_value[0]
              << " score_count="
              << public_scores.size()
              << " output_is_secret=1"
              << " output_scale=Q18"
              << std::endl;
        }
      });
}

}  // namespace spu::mpc::cheetah::test
