import csv
import statistics
import time
from pathlib import Path

import numpy as np

from examples.python.secmoe.secure_moe_correctness import (
    secure_top1_moe,
)
from examples.python.secmoe.secure_moe_select_then_compute import (
    secure_select_then_compute_moe,
)
from examples.python.secmoe.secure_moe_protocol1_select_then_compute import (
    CLIENT_RANK,
    SERVER_RANK,
    RoleAwareSimulator,
    compile_protocol1_function,
    create_protocol1_test_data,
    protocol1_select_then_compute_moe,
)


METHODS = {
    "baseline_all_experts": secure_top1_moe,
    "rq2_v1_select_then_compute": secure_select_then_compute_moe,
    "rq2_v2_protocol1": protocol1_select_then_compute_moe,
}


def max_absolute_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    return float(
        np.max(
            np.abs(actual - expected)
        )
    )


def execute_once(
    simulator,
    executable,
    flat_arguments,
    owner_ranks,
):
    start_time = time.perf_counter()

    outputs, storage_types = simulator.execute(
        executable,
        flat_arguments,
        owner_ranks,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    return (
        outputs,
        storage_types,
        elapsed_seconds,
    )


def benchmark_one_method(
    method_name,
    function,
    arguments,
    owner_ranks,
    repetitions,
):
    print()
    print("-" * 72)
    print(f"Compiling method: {method_name}")
    print("-" * 72)

    compile_start = time.perf_counter()

    (
        executable,
        flat_arguments,
        _output_tree,
    ) = compile_protocol1_function(
        function,
        arguments,
    )

    compile_seconds = (
        time.perf_counter()
        - compile_start
    )

    print(
        f"Compilation time: "
        f"{compile_seconds:.6f} s"
    )

    simulator = RoleAwareSimulator()

    print("Running one warm-up execution...")

    (
        warmup_outputs,
        storage_types,
        warmup_seconds,
    ) = execute_once(
        simulator,
        executable,
        flat_arguments,
        owner_ranks,
    )

    print(
        f"Warm-up time: "
        f"{warmup_seconds:.6f} s"
    )

    measured_times = []
    final_outputs = None

    for repetition_index in range(
        repetitions
    ):
        (
            outputs,
            _storage_types,
            elapsed_seconds,
        ) = execute_once(
            simulator,
            executable,
            flat_arguments,
            owner_ranks,
        )

        measured_times.append(
            elapsed_seconds
        )

        final_outputs = outputs

        print(
            f"Run {repetition_index + 1}/"
            f"{repetitions}: "
            f"{elapsed_seconds:.6f} s"
        )

    median_seconds = statistics.median(
        measured_times
    )

    mean_seconds = statistics.mean(
        measured_times
    )

    minimum_seconds = min(
        measured_times
    )

    maximum_seconds = max(
        measured_times
    )

    return {
        "method": method_name,
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "median_seconds": median_seconds,
        "mean_seconds": mean_seconds,
        "minimum_seconds": minimum_seconds,
        "maximum_seconds": maximum_seconds,
        "times": measured_times,
        "outputs": final_outputs,
        "storage_types": storage_types,
    }


def main():
    batch_size = 2
    d_model = 16
    d_ff = 32

    expert_counts = [
        2,
        4,
        8,
    ]

    repetitions = 3
    seed = 2026

    output_directory = (
        Path.home()
        / "SecMoE"
        / "results"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / "protocol1_comparison.csv"
    )

    all_rows = []

    print(
        "===== Unified SecMoE Benchmark ====="
    )

    print(
        f"batch_size: {batch_size}"
    )

    print(
        f"d_model: {d_model}"
    )

    print(
        f"d_ff: {d_ff}"
    )

    print(
        f"expert_counts: {expert_counts}"
    )

    print(
        f"repetitions: {repetitions}"
    )

    for number_of_experts in expert_counts:
        print()
        print("=" * 80)
        print(
            "Number of experts: "
            f"{number_of_experts}"
        )
        print("=" * 80)

        arguments = create_protocol1_test_data(
            seed=seed,
            batch_size=batch_size,
            d_model=d_model,
            d_ff=d_ff,
            number_of_experts=(
                number_of_experts
            ),
        )

        owner_ranks = [
            None,
            SERVER_RANK,
            SERVER_RANK,
            SERVER_RANK,
            SERVER_RANK,
        ]

        results = {}

        for method_name, function in (
            METHODS.items()
        ):
            result = benchmark_one_method(
                method_name=method_name,
                function=function,
                arguments=arguments,
                owner_ranks=owner_ranks,
                repetitions=repetitions,
            )

            results[method_name] = result

        baseline_output = np.asarray(
            results[
                "baseline_all_experts"
            ]["outputs"][0]
        )

        baseline_median = results[
            "baseline_all_experts"
        ]["median_seconds"]

        print()
        print(
            "===== Correctness and Speedup ====="
        )

        for method_name, result in (
            results.items()
        ):
            current_output = np.asarray(
                result["outputs"][0]
            )

            output_error = max_absolute_error(
                current_output,
                baseline_output,
            )

            speedup = (
                baseline_median
                / result["median_seconds"]
            )

            storage_types = result[
                "storage_types"
            ]

            server_private = all(
                "Priv2k" in storage_type
                for storage_type
                in storage_types[1:]
            )

            print()
            print(
                f"method: {method_name}"
            )

            print(
                "median_seconds: "
                f"{result['median_seconds']:.6f}"
            )

            print(
                "mean_seconds: "
                f"{result['mean_seconds']:.6f}"
            )

            print(
                "speedup_vs_baseline: "
                f"{speedup:.4f}x"
            )

            print(
                "output_error_vs_baseline: "
                f"{output_error:.10f}"
            )

            print(
                "server_weights_private: "
                f"{server_private}"
            )

            all_rows.append(
                {
                    "number_of_experts": (
                        number_of_experts
                    ),
                    "method": method_name,
                    "compile_seconds": (
                        result[
                            "compile_seconds"
                        ]
                    ),
                    "warmup_seconds": (
                        result[
                            "warmup_seconds"
                        ]
                    ),
                    "median_seconds": (
                        result[
                            "median_seconds"
                        ]
                    ),
                    "mean_seconds": (
                        result[
                            "mean_seconds"
                        ]
                    ),
                    "minimum_seconds": (
                        result[
                            "minimum_seconds"
                        ]
                    ),
                    "maximum_seconds": (
                        result[
                            "maximum_seconds"
                        ]
                    ),
                    "speedup_vs_baseline": (
                        speedup
                    ),
                    "output_error_vs_baseline": (
                        output_error
                    ),
                    "server_weights_private": (
                        server_private
                    ),
                }
            )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "number_of_experts",
            "method",
            "compile_seconds",
            "warmup_seconds",
            "median_seconds",
            "mean_seconds",
            "minimum_seconds",
            "maximum_seconds",
            "speedup_vs_baseline",
            "output_error_vs_baseline",
            "server_weights_private",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    print()
    print("=" * 80)
    print("Benchmark completed.")
    print(
        "CSV saved to:",
        csv_path,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
