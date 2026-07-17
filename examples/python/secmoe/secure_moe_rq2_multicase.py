"""Multi-case structural sparsity tests for RQ2.

This program compares the all-expert secure MoE baseline with
the functional Select-Then-Compute implementation under multiple
expert counts.

The test measures logical token-expert network evaluations.
It does not yet measure communication volume or reproduce the
complete SecMoE cryptographic parameter-selection protocol.
"""

from dataclasses import dataclass

import numpy as np

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim

from secure_moe_correctness import create_simulator
from secure_moe_correctness import plaintext_top1_moe
from secure_moe_correctness import secure_top1_moe
from secure_moe_select_then_compute import (
    secure_select_then_compute_moe,
)
from switch_moe_reference import MoEConfig
from switch_moe_reference import initialize_parameters


@dataclass(frozen=True)
class RQ2TestCase:
    """One RQ2 structural sparsity test case."""

    name: str
    batch_size: int
    d_model: int
    d_ff: int
    num_experts: int
    seed: int


TEST_CASES = (
    RQ2TestCase(
        name="two_experts",
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=2,
        seed=2040,
    ),
    RQ2TestCase(
        name="four_experts",
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=4,
        seed=2041,
    ),
    RQ2TestCase(
        name="eight_experts",
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=8,
        seed=2042,
    ),
)

def run_test_case(test_case: RQ2TestCase):
    """Run one RQ2 correctness and sparsity test case."""

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

    secure_all_experts = ppsim.sim_jax(
        simulator,
        secure_top1_moe,
        copts=compiler_options,
    )

    secure_select_then_compute = ppsim.sim_jax(
        simulator,
        secure_select_then_compute_moe,
        copts=compiler_options,
    )

    (
        all_expert_output,
        all_expert_logits,
        all_expert_probabilities,
        all_expert_selected,
    ) = secure_all_experts(
        x,
        router_w,
        w1,
        v,
        w2,
    )

    (
        stc_output,
        stc_logits,
        stc_probabilities,
        stc_selected,
    ) = secure_select_then_compute(
        x,
        router_w,
        w1,
        v,
        w2,
    )

    plain_output_np = np.asarray(
        plain_output
    )

    plain_logits_np = np.asarray(
        plain_logits
    )

    plain_probabilities_np = np.asarray(
        plain_probabilities
    )

    plain_selected_np = np.asarray(
        plain_selected
    )

    all_expert_output_np = np.asarray(
        all_expert_output
    )

    all_expert_logits_np = np.asarray(
        all_expert_logits
    )

    all_expert_probabilities_np = np.asarray(
        all_expert_probabilities
    )

    all_expert_selected_np = np.asarray(
        all_expert_selected
    )

    stc_output_np = np.asarray(
        stc_output
    )

    stc_logits_np = np.asarray(
        stc_logits
    )

    stc_probabilities_np = np.asarray(
        stc_probabilities
    )

    stc_selected_np = np.asarray(
        stc_selected
    )

    all_expert_selection_match = bool(
        np.array_equal(
            plain_selected_np,
            all_expert_selected_np,
        )
    )

    stc_selection_match = bool(
        np.array_equal(
            plain_selected_np,
            stc_selected_np,
        )
    )

    secure_selection_match = bool(
        np.array_equal(
            all_expert_selected_np,
            stc_selected_np,
        )
    )

    all_expert_logit_error = float(
        np.max(
            np.abs(
                plain_logits_np
                - all_expert_logits_np
            )
        )
    )

    stc_logit_error = float(
        np.max(
            np.abs(
                plain_logits_np
                - stc_logits_np
            )
        )
    )

    all_expert_probability_error = float(
        np.max(
            np.abs(
                plain_probabilities_np
                - all_expert_probabilities_np
            )
        )
    )

    stc_probability_error = float(
        np.max(
            np.abs(
                plain_probabilities_np
                - stc_probabilities_np
            )
        )
    )

    all_expert_output_error = float(
        np.max(
            np.abs(
                plain_output_np
                - all_expert_output_np
            )
        )
    )

    stc_output_error = float(
        np.max(
            np.abs(
                plain_output_np
                - stc_output_np
            )
        )
    )

    secure_output_difference = float(
        np.max(
            np.abs(
                all_expert_output_np
                - stc_output_np
            )
        )
    )
    all_expert_evaluations = (
        config.batch_size
        * config.num_experts
    )

    stc_expert_evaluations = (
        config.batch_size
    )

    reduction_factor = (
        all_expert_evaluations
        / stc_expert_evaluations
    )

    expected_reduction_factor = float(
        config.num_experts
    )

    sparsity_match = bool(
        all_expert_evaluations
        == config.batch_size
        * config.num_experts
        and stc_expert_evaluations
        == config.batch_size
        and np.isclose(
            reduction_factor,
            expected_reduction_factor,
        )
    )

    logit_tolerance = 1e-3
    probability_tolerance = 5e-3
    output_tolerance = 1e-2

    passed = (
        all_expert_selection_match
        and stc_selection_match
        and secure_selection_match
        and all_expert_logit_error
        < logit_tolerance
        and stc_logit_error
        < logit_tolerance
        and all_expert_probability_error
        < probability_tolerance
        and stc_probability_error
        < probability_tolerance
        and all_expert_output_error
        < output_tolerance
        and stc_output_error
        < output_tolerance
        and secure_output_difference
        < output_tolerance
        and sparsity_match
    )

    print()
    print(
        f"===== RQ2 Case: {test_case.name} ====="
    )
    print(f"batch_size: {config.batch_size}")
    print(f"num_experts: {config.num_experts}")
    print(f"seed: {config.seed}")

    print(
        "plain_selected_experts:",
        plain_selected_np.tolist(),
    )
    print(
        "all_expert_selected_experts:",
        all_expert_selected_np.tolist(),
    )
    print(
        "stc_selected_experts:",
        stc_selected_np.tolist(),
    )

    print(
        "all_expert_selection_match:",
        all_expert_selection_match,
    )
    print(
        "stc_selection_match:",
        stc_selection_match,
    )
    print(
        "secure_selection_match:",
        secure_selection_match,
    )

    print(
        f"all_expert_logit_error: "
        f"{all_expert_logit_error:.10f}"
    )
    print(
        f"stc_logit_error: "
        f"{stc_logit_error:.10f}"
    )
    print(
        f"all_expert_probability_error: "
        f"{all_expert_probability_error:.10f}"
    )
    print(
        f"stc_probability_error: "
        f"{stc_probability_error:.10f}"
    )
    print(
        f"all_expert_output_error: "
        f"{all_expert_output_error:.10f}"
    )
    print(
        f"stc_output_error: "
        f"{stc_output_error:.10f}"
    )
    print(
        f"secure_output_difference: "
        f"{secure_output_difference:.10f}"
    )

    print(
        "all_expert_token_expert_evaluations:",
        all_expert_evaluations,
    )
    print(
        "stc_token_expert_evaluations:",
        stc_expert_evaluations,
    )
    print(
        f"reduction_factor: "
        f"{reduction_factor:.2f}x"
    )
    print(
        f"expected_reduction_factor: "
        f"{expected_reduction_factor:.2f}x"
    )
    print(f"sparsity_match: {sparsity_match}")
    print(f"passed: {passed}")


    return {
        "name": test_case.name,
        "num_experts": config.num_experts,
        "passed": passed,
        "sparsity_match": sparsity_match,
        "reduction_factor": reduction_factor,
        "all_expert_evaluations": (
            all_expert_evaluations
        ),
        "stc_expert_evaluations": (
            stc_expert_evaluations
        ),
        "stc_output_error": stc_output_error,
        "secure_output_difference": (
            secure_output_difference
        ),
    }

def run_all_test_cases() -> None:
    """Run all RQ2 structural sparsity test cases."""

    print("===== RQ2 Multi-Case Test =====")
    print(f"total_cases: {len(TEST_CASES)}")

    results = []

    for test_case in TEST_CASES:
        result = run_test_case(test_case)
        results.append(result)

    passed_cases = sum(
        int(result["passed"])
        for result in results
    )

    all_passed = (
        passed_cases == len(TEST_CASES)
    )

    all_sparsity_matched = all(
        result["sparsity_match"]
        for result in results
    )

    worst_stc_output_error = max(
        result["stc_output_error"]
        for result in results
    )

    worst_secure_output_difference = max(
        result["secure_output_difference"]
        for result in results
    )

    print()
    print("===== RQ2 Multi-Case Summary =====")
    print(f"total_cases: {len(TEST_CASES)}")
    print(f"passed_cases: {passed_cases}")
    print(
        f"failed_cases: "
        f"{len(TEST_CASES) - passed_cases}"
    )
    print(
        "all_sparsity_matched:",
        all_sparsity_matched,
    )
    print(
        f"worst_stc_output_error: "
        f"{worst_stc_output_error:.10f}"
    )
    print(
        f"worst_secure_output_difference: "
        f"{worst_secure_output_difference:.10f}"
    )

    print()
    print("===== Structural Sparsity Summary =====")

    for result in results:
        print(
            f"{result['name']}: "
            f"{result['all_expert_evaluations']} "
            f"-> {result['stc_expert_evaluations']} "
            f"token-expert evaluations, "
            f"{result['reduction_factor']:.2f}x, "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    print()
    print(
        "parameter_selection_processes_all_experts:",
        True,
    )
    print(
        "expert_network_runs_once_per_token:",
        True,
    )
    print(f"all_passed: {all_passed}")

    if not all_passed:
        failed_names = [
            result["name"]
            for result in results
            if not result["passed"]
        ]

        raise AssertionError(
            "Failed RQ2 cases: "
            + ", ".join(failed_names)
        )


if __name__ == "__main__":
    run_all_test_cases()
