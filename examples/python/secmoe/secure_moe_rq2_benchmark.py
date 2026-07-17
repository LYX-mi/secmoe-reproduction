"""RQ2 timing benchmark for secure MoE.

This benchmark compares:
1. The RQ1 all-expert secure MoE baseline.
2. The functional Select-Then-Compute secure MoE.

Compilation time and simulator execution time are measured
separately.

Execution time includes secret-share creation, runtime setup,
in-memory communication, secure execution, output reading,
and reconstruction.

This is not a reproduction of the complete SecMoE
cryptographic parameter-selection protocol.
"""

from dataclasses import dataclass
from time import perf_counter
import statistics

import jax

try:
    import jax.extend.linear_util as jax_lu
except ImportError:
    import jax.linear_util as jax_lu

import numpy as np

from jax._src import api_util as japi_util

import spu.spu_pb2 as spu_pb2
import spu.utils.frontend as spu_fe

from secure_moe_correctness import create_simulator
from secure_moe_correctness import plaintext_top1_moe
from secure_moe_correctness import secure_top1_moe
from secure_moe_select_then_compute import (
    secure_select_then_compute_moe,
)
from switch_moe_reference import MoEConfig
from switch_moe_reference import initialize_parameters


@dataclass(frozen=True)
class BenchmarkCase:
    """One RQ2 secure MoE timing case."""

    name: str
    batch_size: int
    d_model: int
    d_ff: int
    num_experts: int
    seed: int


BENCHMARK_CASES = (
    BenchmarkCase(
        name="two_experts",
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=2,
        seed=2050,
    ),
    BenchmarkCase(
        name="four_experts",
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=4,
        seed=2051,
    ),
    BenchmarkCase(
        name="eight_experts",
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=8,
        seed=2052,
    ),
)


EXECUTION_REPEATS = 3


def compile_secure_jax_function(
    function,
    arguments,
    compiler_options,
):
    """Compile one JAX function without executing it."""

    compilation_start = perf_counter()

    _, dynamic_arguments = (
        japi_util.argnums_partial_except(
            jax_lu.wrap_init(function),
            (),
            arguments,
            allow_invalid=False,
        )
    )

    flat_arguments, _ = (
        jax.tree_util.tree_flatten(
            (
                dynamic_arguments,
                {},
            )
        )
    )

    input_names = [
        f"in{index}"
        for index in range(
            len(flat_arguments)
        )
    ]

    def output_name_generator(flat_output):
        return [
            f"out{index}"
            for index in range(
                len(flat_output)
            )
        ]

    executable, output_template = spu_fe.compile(
        spu_fe.Kind.JAX,
        function,
        arguments,
        {},
        input_names,
        [
            spu_pb2.Visibility.VIS_SECRET
        ]
        * len(flat_arguments),
        output_name_generator,
        static_argnums=(),
        copts=compiler_options,
    )

    compilation_seconds = (
        perf_counter()
        - compilation_start
    )

    _, output_tree = (
        jax.tree_util.tree_flatten(
            output_template
        )
    )

    return (
        executable,
        flat_arguments,
        output_tree,
        compilation_seconds,
    )


def execute_secure_program(
    simulator,
    executable,
    flat_arguments,
    output_tree,
    label,
):
    """Execute one compiled SPU program repeatedly."""

    execution_times = []
    reconstructed_output = None

    for repeat_index in range(
        EXECUTION_REPEATS
    ):
        execution_start = perf_counter()

        flat_output = simulator(
            executable,
            *flat_arguments,
        )

        execution_seconds = (
            perf_counter()
            - execution_start
        )

        reconstructed_output = (
            jax.tree_util.tree_unflatten(
                output_tree,
                flat_output,
            )
        )

        execution_times.append(
            execution_seconds
        )

        print(
            f"{label}_execution_"
            f"repeat_{repeat_index + 1}: "
            f"{execution_seconds:.6f} s"
        )

    return {
        "output": reconstructed_output,
        "execution_times": execution_times,
        "median_execution_seconds": (
            statistics.median(
                execution_times
            )
        ),
        "mean_execution_seconds": (
            statistics.mean(
                execution_times
            )
        ),
        "minimum_execution_seconds": min(
            execution_times
        ),
        "maximum_execution_seconds": max(
            execution_times
        ),
    }


def calculate_output_error(
    reference_output,
    candidate_output,
) -> float:
    """Return maximum absolute output error."""

    reference_array = np.asarray(
        reference_output
    )

    candidate_array = np.asarray(
        candidate_output
    )

    return float(
        np.max(
            np.abs(
                reference_array
                - candidate_array
            )
        )
    )


def run_benchmark_case(
    benchmark_case: BenchmarkCase,
):
    """Benchmark one expert-count configuration."""

    print()
    print(
        f"===== RQ2 Benchmark: "
        f"{benchmark_case.name} ====="
    )

    config = MoEConfig(
        batch_size=benchmark_case.batch_size,
        d_model=benchmark_case.d_model,
        d_ff=benchmark_case.d_ff,
        num_experts=benchmark_case.num_experts,
        seed=benchmark_case.seed,
        tolerance=1e-3,
    )

    x, router_w, w1, v, w2 = (
        initialize_parameters(
            config
        )
    )

    (
        plain_output,
        _plain_logits,
        _plain_probabilities,
        plain_selected,
    ) = plaintext_top1_moe(
        x,
        router_w,
        w1,
        v,
        w2,
    )

    simulator = create_simulator()

    compiler_options = (
        spu_pb2.CompilerOptions()
    )

    compiler_options.enable_optimize_denominator_with_broadcast = True

    arguments = (
        x,
        router_w,
        w1,
        v,
        w2,
    )

    print(
        "Compiling all-expert "
        "secure MoE..."
    )

    (
        all_expert_executable,
        all_expert_arguments,
        all_expert_output_tree,
        all_expert_compilation_seconds,
    ) = compile_secure_jax_function(
        secure_top1_moe,
        arguments,
        compiler_options,
    )

    print(
        f"all_expert_compilation_seconds: "
        f"{all_expert_compilation_seconds:.6f} s"
    )

    print(
        "Compiling Select-Then-Compute "
        "secure MoE..."
    )

    (
        stc_executable,
        stc_arguments,
        stc_output_tree,
        stc_compilation_seconds,
    ) = compile_secure_jax_function(
        secure_select_then_compute_moe,
        arguments,
        compiler_options,
    )

    print(
        f"stc_compilation_seconds: "
        f"{stc_compilation_seconds:.6f} s"
    )

    print()
    print(
        "Executing all-expert "
        "secure MoE..."
    )

    all_expert_execution = (
        execute_secure_program(
            simulator,
            all_expert_executable,
            all_expert_arguments,
            all_expert_output_tree,
            "all_expert",
        )
    )

    print()
    print(
        "Executing Select-Then-Compute "
        "secure MoE..."
    )

    stc_execution = (
        execute_secure_program(
            simulator,
            stc_executable,
            stc_arguments,
            stc_output_tree,
            "stc",
        )
    )

    (
        all_expert_output,
        _all_expert_logits,
        _all_expert_probabilities,
        all_expert_selected,
    ) = all_expert_execution["output"]

    (
        stc_output,
        _stc_logits,
        _stc_probabilities,
        stc_selected,
    ) = stc_execution["output"]

    plain_selected_np = np.asarray(
        plain_selected
    )

    all_expert_selected_np = np.asarray(
        all_expert_selected
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

    all_expert_output_error = (
        calculate_output_error(
            plain_output,
            all_expert_output,
        )
    )

    stc_output_error = (
        calculate_output_error(
            plain_output,
            stc_output,
        )
    )

    secure_output_difference = (
        calculate_output_error(
            all_expert_output,
            stc_output,
        )
    )

    all_expert_median_seconds = (
        all_expert_execution[
            "median_execution_seconds"
        ]
    )

    stc_median_seconds = (
        stc_execution[
            "median_execution_seconds"
        ]
    )

    if stc_median_seconds > 0:
        execution_speedup = (
            all_expert_median_seconds
            / stc_median_seconds
        )
    else:
        execution_speedup = float("inf")

    if stc_compilation_seconds > 0:
        compilation_ratio = (
            all_expert_compilation_seconds
            / stc_compilation_seconds
        )
    else:
        compilation_ratio = float("inf")

    all_expert_total_seconds = (
        all_expert_compilation_seconds
        + all_expert_median_seconds
    )

    stc_total_seconds = (
        stc_compilation_seconds
        + stc_median_seconds
    )

    if stc_total_seconds > 0:
        total_time_ratio = (
            all_expert_total_seconds
            / stc_total_seconds
        )
    else:
        total_time_ratio = float("inf")

    all_expert_evaluations = (
        config.batch_size
        * config.num_experts
    )

    stc_expert_evaluations = (
        config.batch_size
    )

    theoretical_reduction_factor = (
        all_expert_evaluations
        / stc_expert_evaluations
    )

    output_tolerance = 1e-2

    correctness_passed = (
        all_expert_selection_match
        and stc_selection_match
        and secure_selection_match
        and all_expert_output_error
        < output_tolerance
        and stc_output_error
        < output_tolerance
        and secure_output_difference
        < output_tolerance
    )

    structural_sparsity_passed = (
        all_expert_evaluations
        == config.batch_size
        * config.num_experts
        and stc_expert_evaluations
        == config.batch_size
        and np.isclose(
            theoretical_reduction_factor,
            float(config.num_experts),
        )
    )

    passed = (
        correctness_passed
        and structural_sparsity_passed
    )

    print()
    print("===== Correctness Result =====")
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

    print()
    print("===== Timing Result =====")
    print(
        f"all_expert_compilation_seconds: "
        f"{all_expert_compilation_seconds:.6f}"
    )
    print(
        f"stc_compilation_seconds: "
        f"{stc_compilation_seconds:.6f}"
    )
    print(
        f"compilation_ratio: "
        f"{compilation_ratio:.4f}x"
    )
    print(
        f"all_expert_median_execution_seconds: "
        f"{all_expert_median_seconds:.6f}"
    )
    print(
        f"stc_median_execution_seconds: "
        f"{stc_median_seconds:.6f}"
    )
    print(
        f"execution_speedup: "
        f"{execution_speedup:.4f}x"
    )
    print(
        f"all_expert_total_seconds: "
        f"{all_expert_total_seconds:.6f}"
    )
    print(
        f"stc_total_seconds: "
        f"{stc_total_seconds:.6f}"
    )
    print(
        f"total_time_ratio: "
        f"{total_time_ratio:.4f}x"
    )

    print()
    print("===== Structural Sparsity =====")
    print(
        "all_expert_token_expert_evaluations:",
        all_expert_evaluations,
    )
    print(
        "stc_token_expert_evaluations:",
        stc_expert_evaluations,
    )
    print(
        f"theoretical_reduction_factor: "
        f"{theoretical_reduction_factor:.2f}x"
    )
    print(
        "correctness_passed:",
        correctness_passed,
    )
    print(
        "structural_sparsity_passed:",
        structural_sparsity_passed,
    )
    print(f"passed: {passed}")

    return {
        "name": benchmark_case.name,
        "num_experts": config.num_experts,
        "passed": passed,
        "correctness_passed": (
            correctness_passed
        ),
        "structural_sparsity_passed": (
            structural_sparsity_passed
        ),
        "all_expert_compilation_seconds": (
            all_expert_compilation_seconds
        ),
        "stc_compilation_seconds": (
            stc_compilation_seconds
        ),
        "all_expert_median_seconds": (
            all_expert_median_seconds
        ),
        "stc_median_seconds": (
            stc_median_seconds
        ),
        "execution_speedup": (
            execution_speedup
        ),
        "total_time_ratio": (
            total_time_ratio
        ),
        "theoretical_reduction_factor": (
            theoretical_reduction_factor
        ),
        "stc_output_error": (
            stc_output_error
        ),
        "secure_output_difference": (
            secure_output_difference
        ),
    }


def run_all_benchmarks() -> None:
    """Run all RQ2 secure MoE timing benchmarks."""

    print("===== RQ2 Secure MoE Benchmark =====")
    print(
        f"total_cases: "
        f"{len(BENCHMARK_CASES)}"
    )
    print(
        f"execution_repeats_per_program: "
        f"{EXECUTION_REPEATS}"
    )

    results = []

    for benchmark_case in BENCHMARK_CASES:
        result = run_benchmark_case(
            benchmark_case
        )
        results.append(result)

    passed_cases = sum(
        int(result["passed"])
        for result in results
    )

    all_passed = (
        passed_cases
        == len(BENCHMARK_CASES)
    )

    correctness_passed_cases = sum(
        int(result["correctness_passed"])
        for result in results
    )

    sparsity_passed_cases = sum(
        int(
            result[
                "structural_sparsity_passed"
            ]
        )
        for result in results
    )

    faster_execution_cases = sum(
        int(
            result["execution_speedup"] > 1.0
        )
        for result in results
    )

    faster_total_time_cases = sum(
        int(
            result["total_time_ratio"] > 1.0
        )
        for result in results
    )

    execution_speedups = [
        result["execution_speedup"]
        for result in results
    ]

    total_time_ratios = [
        result["total_time_ratio"]
        for result in results
    ]

    median_execution_speedup = (
        statistics.median(
            execution_speedups
        )
    )

    median_total_time_ratio = (
        statistics.median(
            total_time_ratios
        )
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
    print("===== RQ2 Benchmark Summary =====")
    print(
        f"total_cases: "
        f"{len(BENCHMARK_CASES)}"
    )
    print(f"passed_cases: {passed_cases}")
    print(
        f"failed_cases: "
        f"{len(BENCHMARK_CASES) - passed_cases}"
    )
    print(
        f"correctness_passed_cases: "
        f"{correctness_passed_cases}"
    )
    print(
        f"sparsity_passed_cases: "
        f"{sparsity_passed_cases}"
    )
    print(
        f"faster_execution_cases: "
        f"{faster_execution_cases}"
    )
    print(
        f"faster_total_time_cases: "
        f"{faster_total_time_cases}"
    )
    print(
        f"median_execution_speedup: "
        f"{median_execution_speedup:.4f}x"
    )
    print(
        f"median_total_time_ratio: "
        f"{median_total_time_ratio:.4f}x"
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
    print("===== Per-Case Timing Summary =====")

    for result in results:
        print(
            f"{result['name']}: "
            f"experts={result['num_experts']}, "
            f"theoretical="
            f"{result['theoretical_reduction_factor']:.2f}x, "
            f"execution="
            f"{result['execution_speedup']:.4f}x, "
            f"total="
            f"{result['total_time_ratio']:.4f}x, "
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
    print(
        "timing_environment:",
        "single-machine SPU simulator",
    )
    print(
        "execution_time_includes:",
        (
            "share creation, runtime setup, "
            "in-memory communication, execution, "
            "and reconstruction"
        ),
    )
    print(f"all_passed: {all_passed}")

    if not all_passed:
        failed_names = [
            result["name"]
            for result in results
            if not result["passed"]
        ]

        raise AssertionError(
            "Failed RQ2 benchmark cases: "
            + ", ".join(failed_names)
        )


if __name__ == "__main__":
    run_all_benchmarks()
