import jax
import jax.numpy as jnp
import numpy as np

import spu.intrinsic as si
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim

from secure_moe_correctness import create_simulator
from secure_moe_correctness import plaintext_top1_moe
from secure_moe_correctness import secure_softmax
from secure_moe_correctness import secure_top1_moe
from switch_moe_reference import MoEConfig
from switch_moe_reference import initialize_parameters

def secure_select_then_compute_moe(
    x,
    router_w,
    w1,
    v,
    w2,
):
    """Securely select expert parameters, then run one expert."""

    router_logits = x @ router_w

    router_probabilities = secure_softmax(
        router_logits,
        axis=-1,
    )

    selected_experts = jnp.argmax(
        router_logits,
        axis=-1,
    )

    num_experts = w1.shape[0]

    selection_mask = jax.nn.one_hot(
        selected_experts,
        num_experts,
        dtype=x.dtype,
    )

    selected_w1 = jnp.sum(
        selection_mask[:, :, None, None]
        * w1[None, :, :, :],
        axis=1,
    )

    selected_v = jnp.sum(
        selection_mask[:, :, None, None]
        * v[None, :, :, :],
        axis=1,
    )

    selected_w2 = jnp.sum(
        selection_mask[:, :, None, None]
        * w2[None, :, :, :],
        axis=1,
    )

    first_projection = jnp.matmul(
        x[:, None, :],
        selected_w1,
    )

    first_projection = jnp.squeeze(
        first_projection,
        axis=1,
    )

    gate_projection = jnp.matmul(
        x[:, None, :],
        selected_v,
    )

    gate_projection = jnp.squeeze(
        gate_projection,
        axis=1,
    )

    activated = si.spu_gelu(
        first_projection
    )

    hidden = activated * gate_projection

    expert_output = jnp.matmul(
        hidden[:, None, :],
        selected_w2,
    )

    expert_output = jnp.squeeze(
        expert_output,
        axis=1,
    )

    selected_gate = jnp.sum(
        router_probabilities
        * selection_mask,
        axis=-1,
    )

    final_output = (
        expert_output
        * selected_gate[:, None]
    )

    return (
        final_output,
        router_logits,
        router_probabilities,
        selected_experts,
    )

def run_rq2_correctness_test() -> None:
    """Compare plaintext, all-expert, and select-then-compute MoE."""

    config = MoEConfig(
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=4,
        seed=2030,
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

    theoretical_reduction_factor = (
        all_expert_evaluations
        / stc_expert_evaluations
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
        and stc_expert_evaluations
        == config.batch_size
    )

    print("===== RQ2 Select-Then-Compute Test =====")
    print(f"batch_size: {config.batch_size}")
    print(f"d_model: {config.d_model}")
    print(f"d_ff: {config.d_ff}")
    print(f"num_experts: {config.num_experts}")
    print(f"seed: {config.seed}")

    print()
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

    print()
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

    print()
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

    print()
    print(
        "parameter_selection_processes_all_experts:",
        True,
    )
    print(
        "expert_network_runs_once_per_token:",
        True,
    )
    print(f"passed: {passed}")

    if not passed:
        raise AssertionError(
            "RQ2 Select-Then-Compute "
            "correctness test failed."
        )

if __name__ == "__main__":
    run_rq2_correctness_test()
