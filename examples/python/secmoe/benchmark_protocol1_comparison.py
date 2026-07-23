import csv
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from spu import intrinsic as si

from examples.python.secmoe.secure_moe_correctness import (
    secure_expert_forward,
    secure_softmax,
)
from examples.python.secmoe.secure_moe_protocol1_select_then_compute import (
    SERVER_RANK,
    RoleAwareSimulator,
    compile_protocol1_function,
    create_protocol1_test_data,
    protocol1_select_then_compute_moe,
)


def secure_top1_index(router_logits):
    """Use the same secure Top-1 operator in every method."""

    _, top_indices = jax.lax.top_k(
        router_logits,
        k=1,
    )

    return jnp.squeeze(
        top_indices,
        axis=-1,
    )


def baseline_all_experts_topk(
    x,
    router_w,
    w1,
    v,
    w2,
):
    """Compute every expert, then select one output."""

    router_logits = x @ router_w

    router_probabilities = secure_softmax(
        router_logits,
        axis=-1,
    )

    selected_experts = secure_top1_index(
        router_logits
    )

    expert_outputs = []

    for expert_id in range(w1.shape[0]):
        expert_outputs.append(
            secure_expert_forward(
                x,
                w1[expert_id],
                v[expert_id],
                w2[expert_id],
            )
        )

    stacked_outputs = jnp.stack(
        expert_outputs,
        axis=1,
    )

    selection_mask = jax.nn.one_hot(
        selected_experts,
        w1.shape[0],
        dtype=x.dtype,
    )

    selected_output = jnp.sum(
        stacked_outputs
        * selection_mask[:, :, None],
        axis=1,
    )

    selected_gate = jnp.sum(
        router_probabilities
        * selection_mask,
        axis=-1,
    )

    return (
        selected_output
        * selected_gate[:, None]
    )


def select_then_compute_topk(
    x,
    router_w,
    w1,
    v,
    w2,
):
    """Select parameters with one-hot, then run one expert."""

    router_logits = x @ router_w

    router_probabilities = secure_softmax(
        router_logits,
        axis=-1,
    )

    selected_experts = secure_top1_index(
        router_logits
    )

    selection_mask = jax.nn.one_hot(
        selected_experts,
        w1.shape[0],
        dtype=x.dtype,
    )

    parameter_mask = selection_mask[
        :,
        :,
        None,
        None,
    ]

    selected_w1 = jnp.sum(
        parameter_mask
        * w1[None, :, :, :],
        axis=1,
    )

    selected_v = jnp.sum(
        parameter_mask
        * v[None, :, :, :],
        axis=1,
    )

    selected_w2 = jnp.sum(
        parameter_mask
        * w2[None, :, :, :],
        axis=1,
    )

    first_projection = jnp.squeeze(
        jnp.matmul(
            x[:, None, :],
            selected_w1,
        ),
        axis=1,
    )

    gate_projection = jnp.squeeze(
        jnp.matmul(
            x[:, None, :],
            selected_v,
        ),
        axis=1,
    )

    hidden = (
        si.spu_gelu(first_projection)
        * gate_projection
    )

    expert_output = jnp.squeeze(
        jnp.matmul(
            hidden[:, None, :],
            selected_w2,
        ),
        axis=1,
    )

    selected_gate = jnp.sum(
        router_probabilities
        * selection_mask,
        axis=-1,
    )

    return (
        expert_output
        * selected_gate[:, None]
    )


def protocol1_output_only(
    x,
    router_w,
    w1,
    v,
    w2,
):
    """Run the Protocol-1-aligned implementation.

    Only final_output is returned so that diagnostic
    outputs do not affect the benchmark.
    """

    return protocol1_select_then_compute_moe(
        x,
        router_w,
        w1,
        v,
        w2,
    )[0]


METHODS = {
    "baseline_all_experts_topk": (
        baseline_all_experts_topk
    ),
    "select_then_compute_topk": (
        select_then_compute_topk
    ),
    "protocol1_output_only": (
        protocol1_output_only
    ),
}


def compile_method(
    method_name,
    function,
    arguments,
):
    """Compile one method and prepare its simulator."""

    start_time = time.perf_counter()

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
        - start_time
    )

    return {
        "name": method_name,
        "executable": executable,
        "flat_arguments": flat_arguments,
        "simulator": RoleAwareSimulator(),
        "compile_seconds": compile_seconds,
        "warmup_seconds": None,
        "times": [],
        "last_output": None,
        "storage_types": None,
    }


def execute_once(
    compiled_entry,
    owner_ranks,
):
    """Execute one two-party secure computation."""

    start_time = time.perf_counter()

    (
        outputs,
        storage_types,
    ) = compiled_entry[
        "simulator"
    ].execute(
        compiled_entry["executable"],
        compiled_entry["flat_arguments"],
        owner_ranks,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    return (
        np.asarray(outputs[0]),
        storage_types,
        elapsed_seconds,
    )


def maximum_absolute_error(
    actual,
    expected,
):
    """Calculate maximum output difference."""

    actual = np.asarray(actual)
    expected = np.asarray(expected)

    return float(
        np.max(
            np.abs(
                actual
                - expected
            )
        )
    )


def main():
    batch_size = 2
    d_model = 16
    d_ff = 32

    expert_counts = [
        2,
        4,
        8,
    ]

    repetitions = 7
    seed = 2026

    result_directory = (
        Path.home()
        / "SecMoE"
        / "results"
    )

    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        result_directory
        / "protocol1_fair_summary.csv"
    )

    raw_path = (
        result_directory
        / "protocol1_fair_raw.csv"
    )

    # x uses ordinary arithmetic secret sharing.
    # All model parameters belong to server rank 1.
    owner_ranks = [
        None,
        SERVER_RANK,
        SERVER_RANK,
        SERVER_RANK,
        SERVER_RANK,
    ]

    summary_rows = []
    raw_rows = []

    print(
        "===== Fair SecMoE "
        "Protocol-1 Benchmark ====="
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

    print(
        "All methods use the same secure "
        "TopK and return only final_output."
    )

    for number_of_experts in expert_counts:
        print(
            "\n"
            + "=" * 80
        )

        print(
            "Number of experts: "
            f"{number_of_experts}"
        )

        print(
            "=" * 80
        )

        arguments = create_protocol1_test_data(
            seed=seed,
            batch_size=batch_size,
            d_model=d_model,
            d_ff=d_ff,
            number_of_experts=(
                number_of_experts
            ),
        )

        compiled_methods = {
            method_name: compile_method(
                method_name,
                function,
                arguments,
            )
            for method_name, function
            in METHODS.items()
        }

        print(
            "\n===== Warm-up ====="
        )

        for method_name, entry in (
            compiled_methods.items()
        ):
            (
                output,
                storage_types,
                warmup_seconds,
            ) = execute_once(
                entry,
                owner_ranks,
            )

            entry["last_output"] = output

            entry["storage_types"] = (
                storage_types
            )

            entry["warmup_seconds"] = (
                warmup_seconds
            )

            print(
                f"{method_name}: "
                f"{warmup_seconds:.6f} s "
                f"(compile "
                f"{entry['compile_seconds']:.6f} s)"
            )

        method_names = list(
            METHODS.keys()
        )

        # Rotate method order in every round.
        # This prevents one method from always running
        # first or last.
        for round_index in range(
            repetitions
        ):
            offset = (
                round_index
                % len(method_names)
            )

            round_order = (
                method_names[offset:]
                + method_names[:offset]
            )

            print(
                f"\nRound "
                f"{round_index + 1}/"
                f"{repetitions}"
            )

            print(
                "Execution order:",
                round_order,
            )

            for method_name in round_order:
                entry = compiled_methods[
                    method_name
                ]

                (
                    output,
                    storage_types,
                    elapsed_seconds,
                ) = execute_once(
                    entry,
                    owner_ranks,
                )

                entry["times"].append(
                    elapsed_seconds
                )

                entry["last_output"] = (
                    output
                )

                entry["storage_types"] = (
                    storage_types
                )

                raw_rows.append(
                    {
                        "number_of_experts": (
                            number_of_experts
                        ),
                        "round": (
                            round_index + 1
                        ),
                        "method": (
                            method_name
                        ),
                        "elapsed_seconds": (
                            elapsed_seconds
                        ),
                    }
                )

                print(
                    f"{method_name}: "
                    f"{elapsed_seconds:.6f} s"
                )

        baseline_name = (
            "baseline_all_experts_topk"
        )

        baseline_output = (
            compiled_methods[
                baseline_name
            ]["last_output"]
        )

        baseline_median = (
            statistics.median(
                compiled_methods[
                    baseline_name
                ]["times"]
            )
        )

        print(
            "\n===== Summary ====="
        )

        for method_name, entry in (
            compiled_methods.items()
        ):
            execution_times = entry[
                "times"
            ]

            median_seconds = (
                statistics.median(
                    execution_times
                )
            )

            mean_seconds = (
                statistics.mean(
                    execution_times
                )
            )

            stdev_seconds = (
                statistics.stdev(
                    execution_times
                )
            )

            speedup = (
                baseline_median
                / median_seconds
            )

            output_error = (
                maximum_absolute_error(
                    entry["last_output"],
                    baseline_output,
                )
            )

            server_weights_private = all(
                "Priv2k" in storage_type
                for storage_type
                in entry[
                    "storage_types"
                ][1:]
            )

            print(
                f"{method_name}: "
                f"median={median_seconds:.6f} s, "
                f"mean={mean_seconds:.6f} s, "
                f"stdev={stdev_seconds:.6f} s, "
                f"speedup={speedup:.4f}x, "
                f"error={output_error:.10f}, "
                f"private="
                f"{server_weights_private}"
            )

            summary_rows.append(
                {
                    "number_of_experts": (
                        number_of_experts
                    ),
                    "method": (
                        method_name
                    ),
                    "compile_seconds": (
                        entry[
                            "compile_seconds"
                        ]
                    ),
                    "warmup_seconds": (
                        entry[
                            "warmup_seconds"
                        ]
                    ),
                    "median_seconds": (
                        median_seconds
                    ),
                    "mean_seconds": (
                        mean_seconds
                    ),
                    "stdev_seconds": (
                        stdev_seconds
                    ),
                    "minimum_seconds": (
                        min(execution_times)
                    ),
                    "maximum_seconds": (
                        max(execution_times)
                    ),
                    "speedup_vs_baseline": (
                        speedup
                    ),
                    "output_error_vs_baseline": (
                        output_error
                    ),
                    "server_weights_private": (
                        server_weights_private
                    ),
                }
            )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as raw_file:
        writer = csv.DictWriter(
            raw_file,
            fieldnames=list(
                raw_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            raw_rows
        )

    print(
        "\nBenchmark completed."
    )

    print(
        "Summary CSV:",
        summary_path,
    )

    print(
        "Raw CSV:",
        raw_path,
    )


if __name__ == "__main__":
    main()
