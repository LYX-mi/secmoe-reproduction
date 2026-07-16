"""Multi-case correctness tests for the secure Top-1 MoE.

This program evaluates the secure MoE with multiple random seeds,
batch sizes, and expert counts.
"""

from dataclasses import dataclass

import numpy as np

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim

from secure_moe_correctness import create_simulator
from secure_moe_correctness import plaintext_top1_moe
from secure_moe_correctness import secure_top1_moe
from switch_moe_reference import MoEConfig
from switch_moe_reference import initialize_parameters


@dataclass(frozen=True)
class TestCase:
    """One secure MoE correctness test case."""

    name: str
    batch_size: int
    d_model: int
    d_ff: int
    num_experts: int
    seed: int
TEST_CASES = (
    TestCase(
        name="two_experts_single_token",
        batch_size=1,
        d_model=16,
        d_ff=32,
        num_experts=2,
        seed=2026,
    ),
    TestCase(
        name="four_experts_seed_2027",
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=4,
        seed=2027,
    ),
    TestCase(
        name="four_experts_four_tokens",
        batch_size=4,
        d_model=16,
        d_ff=32,
        num_experts=4,
        seed=2028,
    ),
    TestCase(
        name="eight_experts_seed_2029",
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=8,
        seed=2029,
    ),
)

def run_test_case(test_case: TestCase):
    """Run one plaintext-versus-secure MoE correctness case."""

    config = MoEConfig(
        batch_size=test_case.batch_size,
        d_model=test_case.d_model,
        d_ff=test_case.d_ff,
        num_experts=test_case.num_experts,
        seed=test_case.seed,
        tolerance=1e-3,
    )

    x, router_w, w1, v, w2 = initialize_parameters(
        config
    )

    (
        plain_output,
        plain_logits,
        plain_probabilities,
        plain_selected,
    ) = plaintext_top1_moe(
        x,
        router_w,
        w1,
        v,
        w2,
    )

    simulator = create_simulator()

    compiler_options = spu_pb2.CompilerOptions()

    compiler_options.enable_optimize_denominator_with_broadcast = True

    secure_moe = ppsim.sim_jax(
        simulator,
        secure_top1_moe,
        copts=compiler_options,
    )

    (
        secure_output,
        secure_logits,
        secure_probabilities,
        secure_selected,
    ) = secure_moe(
        x,
        router_w,
        w1,
        v,
        w2,
    )

    plain_output_np = np.asarray(
        plain_output
    )

    secure_output_np = np.asarray(
        secure_output
    )

    plain_logits_np = np.asarray(
        plain_logits
    )

    secure_logits_np = np.asarray(
        secure_logits
    )

    plain_probabilities_np = np.asarray(
        plain_probabilities
    )

    secure_probabilities_np = np.asarray(
        secure_probabilities
    )

    plain_selected_np = np.asarray(
        plain_selected
    )

    secure_selected_np = np.asarray(
        secure_selected
    )

    selection_match = bool(
        np.array_equal(
            plain_selected_np,
            secure_selected_np,
        )
    )

    max_logit_error = float(
        np.max(
            np.abs(
                plain_logits_np
                - secure_logits_np
            )
        )
    )

    max_probability_error = float(
        np.max(
            np.abs(
                plain_probabilities_np
                - secure_probabilities_np
            )
        )
    )

    max_output_error = float(
        np.max(
            np.abs(
                plain_output_np
                - secure_output_np
            )
        )
    )

    logit_tolerance = 1e-3
    probability_tolerance = 5e-3
    output_tolerance = 1e-2

    passed = (
        selection_match
        and max_logit_error < logit_tolerance
        and max_probability_error
        < probability_tolerance
        and max_output_error
        < output_tolerance
    )

    print()
    print(
        f"===== Test Case: {test_case.name} ====="
    )
    print(f"batch_size: {test_case.batch_size}")
    print(f"d_model: {test_case.d_model}")
    print(f"d_ff: {test_case.d_ff}")
    print(f"num_experts: {test_case.num_experts}")
    print(f"seed: {test_case.seed}")
    print(
        "plain_selected_experts:",
        plain_selected_np.tolist(),
    )
    print(
        "secure_selected_experts:",
        secure_selected_np.tolist(),
    )
    print(f"selection_match: {selection_match}")
    print(
        f"max_logit_error: "
        f"{max_logit_error:.10f}"
    )
    print(
        f"max_probability_error: "
        f"{max_probability_error:.10f}"
    )
    print(
        f"max_output_error: "
        f"{max_output_error:.10f}"
    )
    print(f"passed: {passed}")

    return {
        "name": test_case.name,
        "passed": passed,
        "selection_match": selection_match,
        "max_logit_error": max_logit_error,
        "max_probability_error": max_probability_error,
        "max_output_error": max_output_error,
    }

def run_all_tests() -> None:
    """Run all secure MoE correctness test cases."""

    results = []

    for test_case in TEST_CASES:
        result = run_test_case(test_case)
        results.append(result)

    total_cases = len(results)

    passed_cases = sum(
        int(result["passed"])
        for result in results
    )

    overall_passed = (
        passed_cases == total_cases
    )

    worst_logit_error = max(
        result["max_logit_error"]
        for result in results
    )

    worst_probability_error = max(
        result["max_probability_error"]
        for result in results
    )

    worst_output_error = max(
        result["max_output_error"]
        for result in results
    )

    print()
    print("===== RQ1 Multi-Case Summary =====")
    print(f"total_cases: {total_cases}")
    print(f"passed_cases: {passed_cases}")
    print(f"failed_cases: {total_cases - passed_cases}")
    print(
        f"worst_logit_error: "
        f"{worst_logit_error:.10f}"
    )
    print(
        f"worst_probability_error: "
        f"{worst_probability_error:.10f}"
    )
    print(
        f"worst_output_error: "
        f"{worst_output_error:.10f}"
    )
    print(f"overall_passed: {overall_passed}")

    for result in results:
        print(
            f"{result['name']}: "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    if not overall_passed:
        raise AssertionError(
            "One or more secure MoE "
            "correctness cases failed."
        )


if __name__ == "__main__":
    run_all_tests()
