// Copyright 2026
// SecMoE vs Naive Benchmark - with communication stats

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/mpc/cheetah/arith/secmoe_protocol2_gelu.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"
#include "yacl/link/context.h"

namespace spu::mpc::cheetah::test {

void RunGELUBenchmark(int64_t num_experts, int64_t repeats, const std::string& label) {
  constexpr FieldType kField = FieldType::FM64;
  constexpr int64_t kFractionBits = 18;
  const Shape shape = {64};

  auto clear = ring_zeros(kField, shape);
  auto share0 = ring_zeros(kField, shape);
  auto share1 = ring_zeros(kField, shape);
  for (int64_t i = 0; i < 64; ++i) {
    uint64_t v = static_cast<uint64_t>((i * 101) % 131072);
    uint64_t s0 = static_cast<uint64_t>((7 + i * 13) % 1048576);
    clear.at<uint64_t>(i) = v;
    share0.at<uint64_t>(i) = s0;
    share1.at<uint64_t>(i) = v - s0;
  }

  double total_send_mib = 0;
  double total_recv_mib = 0;

  auto start = std::chrono::steady_clock::now();

  for (int64_t rep = 0; rep < repeats; ++rep) {
    utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& link) {
      RuntimeConfig cfg;
      cfg.set_protocol(ProtocolKind::CHEETAH);
      cfg.set_field(kField);
      cfg.set_fxp_fraction_bits(kFractionBits);
      cfg.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);

      SPUContext ctx(cfg, link);
      Factory::RegisterProtocol(&ctx, link);

      const NdArrayRef& local = (link->Rank() == 0) ? share0 : share1;
      Value secret_x(local.as(makeType<AShrTy>(kField)), DT_F64);

      auto stats_before = link->GetStats();
      size_t send_before = stats_before->sent_bytes;
      size_t recv_before = stats_before->recv_bytes;

      for (int64_t e = 0; e < num_experts; ++e) {
        auto result = SecMoEProtocol2GeLU(&ctx, secret_x);
        (void)result;
      }

      auto stats_after = link->GetStats();
      size_t send_bytes = stats_after->sent_bytes - send_before;
      size_t recv_bytes = stats_after->recv_bytes - recv_before;

      if (link->Rank() == 0) {
        total_send_mib += static_cast<double>(send_bytes) / (1024.0 * 1024.0);
        total_recv_mib += static_cast<double>(recv_bytes) / (1024.0 * 1024.0);
      }
    });
  }

  auto end = std::chrono::steady_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  auto avg_ms = static_cast<double>(total_ms) / repeats;

  std::cout << label << " experts=" << num_experts
            << " repeats=" << repeats
            << " total_ms=" << total_ms
            << " avg_ms=" << avg_ms
            << " send_MiB=" << total_send_mib
            << " recv_MiB=" << total_recv_mib
            << std::endl;
}

}  // namespace

TEST(SecMoEBenchmark, CompareNaiveVsSecMoE) {
  constexpr int64_t kRepeats = 10;
  int64_t expert_sizes[] = {4, 8, 16};

  for (int ei = 0; ei < 3; ++ei) {
    int64_t E = expert_sizes[ei];

    spu::mpc::cheetah::test::RunGELUBenchmark(E, kRepeats, "[NAIVE]");
    spu::mpc::cheetah::test::RunGELUBenchmark(1, kRepeats, "[SECMOE]");
  }
}
