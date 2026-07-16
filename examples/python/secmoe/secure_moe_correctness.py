"""End-to-end secure MoE correctness test for SecMoE reproduction.

This program compares a plaintext Top-1 MoE with the same MoE
executed using the two-party CHEETAH protocol.
"""

import jax
import jax.numpy as jnp
import numpy as np

import spu.intrinsic as si
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim

from switch_moe_reference import MoEConfig
from switch_moe_reference import initialize_parameters


def make_test_config() -> MoEConfig:
    """Create a small configuration for the first secure MoE test."""

    return MoEConfig(
        batch_size=2,
        d_model=16,
        d_ff=32,
        num_experts=4,
        seed=2026,
        tolerance=1e-3,
    )

def plaintext_expert_forward(
    x: jnp.ndarray,
    w1: jnp.ndarray,
    v: jnp.ndarray,
    w2: jnp.ndarray,
) -> jnp.ndarray:
    """Compute one expert using ordinary JAX operations."""

    hidden_gelu = jax.nn.gelu(
        jnp.matmul(x, w1)
    )

    hidden_linear = jnp.matmul(
        x,
        v,
    )

    gated_hidden = (
        hidden_gelu * hidden_linear
    )

    output = jnp.matmul(
        gated_hidden,
        w2,
    )

    return output


def secure_expert_forward(
    x: jnp.ndarray,
    w1: jnp.ndarray,
    v: jnp.ndarray,
    w2: jnp.ndarray,
) -> jnp.ndarray:
    """Compute one expert using BumbleBee secure GeLU."""

    hidden_gelu = si.spu_gelu(
        jnp.matmul(x, w1)
    )

    hidden_linear = jnp.matmul(
        x,
        v,
    )

    gated_hidden = (
        hidden_gelu * hidden_linear
    )

    output = jnp.matmul(
        gated_hidden,
        w2,
    )

    return output

def plaintext_top1_moe(
    x: jnp.ndarray,
    router_w: jnp.ndarray,
    w1: jnp.ndarray,
    v: jnp.ndarray,
    w2: jnp.ndarray,
):
    """Compute the plaintext Top-1 MoE output."""

    router_logits = jnp.matmul(
        x,
        router_w,
    )

    router_probabilities = jax.nn.softmax(
        router_logits,
        axis=-1,
    )

    selected_experts = jnp.argmax(
        router_logits,
        axis=-1,
    )

    expert_outputs = []

    for expert_id in range(w1.shape[0]):
        expert_output = plaintext_expert_forward(
            x,
            w1[expert_id],
            v[expert_id],
            w2[expert_id],
        )

        expert_outputs.append(expert_output)

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
        axis=1,
    )

    final_output = (
        selected_output
        * selected_gate[:, None]
    )

    return (
        final_output,
        router_logits,
        router_probabilities,
        selected_experts,
    )

def secure_softmax(
    x: jnp.ndarray,
    axis: int = -1,
) -> jnp.ndarray:
    """Compute Softmax using BumbleBee secure exponential."""

    x_max = jnp.max(
        x,
        axis=axis,
        keepdims=True,
    )

    shifted_x = x - x_max

    exp_values = si.spu_neg_exp(
        shifted_x
    )

    denominator = jnp.sum(
        exp_values,
        axis=axis,
        keepdims=True,
    )

    probabilities = (
        exp_values / denominator
    )

    return probabilities

def secure_top1_moe(
    x: jnp.ndarray,
    router_w: jnp.ndarray,
    w1: jnp.ndarray,
    v: jnp.ndarray,
    w2: jnp.ndarray,
):
    """Compute the end-to-end Top-1 MoE inside SPU."""

    router_logits = jnp.matmul(
        x,
        router_w,
    )

    router_probabilities = secure_softmax(
        router_logits,
        axis=-1,
    )

    selected_experts = jnp.argmax(
        router_logits,
        axis=-1,
    )

    expert_outputs = []

    for expert_id in range(w1.shape[0]):
        expert_output = secure_expert_forward(
            x,
            w1[expert_id],
            v[expert_id],
            w2[expert_id],
        )

        expert_outputs.append(expert_output)

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
        axis=1,
    )

    final_output = (
        selected_output
        * selected_gate[:, None]
    )

    return (
        final_output,
        router_logits,
        router_probabilities,
        selected_experts,
    )

def create_simulator() -> ppsim.Simulator:
    """Create the two-party CHEETAH simulator."""

    runtime_config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH,
        field=spu_pb2.FieldType.FM64,
    )

    runtime_config.enable_hal_profile = True

    runtime_config.experimental_enable_colocated_optimization = False

    runtime_config.cheetah_2pc_config.enable_mul_lsb_error = True

    runtime_config.cheetah_2pc_config.approx_less_precision = 4

    return ppsim.Simulator(
        2,
        runtime_config,
    )

def run_end_to_end_test() -> None:
    """Compare plaintext and secure end-to-end MoE outputs."""

    config = make_test_config()

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

    logit_error = np.abs(
        plain_logits_np
        - secure_logits_np
    )

    probability_error = np.abs(
        plain_probabilities_np
        - secure_probabilities_np
    )

    output_error = np.abs(
        plain_output_np
        - secure_output_np
    )

    max_logit_error = float(
        np.max(logit_error)
    )

    mean_logit_error = float(
        np.mean(logit_error)
    )

    max_probability_error = float(
        np.max(probability_error)
    )

    mean_probability_error = float(
        np.mean(probability_error)
    )

    max_output_error = float(
        np.max(output_error)
    )

    mean_output_error = float(
        np.mean(output_error)
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

    print("===== Secure End-to-End MoE Test =====")
    print("protocol: CHEETAH")
    print("field: FM64")
    print("world_size: 2")
    print(f"batch_size: {config.batch_size}")
    print(f"d_model: {config.d_model}")
    print(f"d_ff: {config.d_ff}")
    print(f"num_experts: {config.num_experts}")
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
        f"mean_logit_error: "
        f"{mean_logit_error:.10f}"
    )
    print(
        f"max_probability_error: "
        f"{max_probability_error:.10f}"
    )
    print(
        f"mean_probability_error: "
        f"{mean_probability_error:.10f}"
    )
    print(
        f"max_output_error: "
        f"{max_output_error:.10f}"
    )
    print(
        f"mean_output_error: "
        f"{mean_output_error:.10f}"
    )
    print(
        "plain_output_shape:",
        tuple(plain_output_np.shape),
    )
    print(
        "secure_output_shape:",
        tuple(secure_output_np.shape),
    )
    print(
        f"logit_tolerance: "
        f"{logit_tolerance}"
    )
    print(
        f"probability_tolerance: "
        f"{probability_tolerance}"
    )
    print(
        f"output_tolerance: "
        f"{output_tolerance}"
    )
    print(f"passed: {passed}")

    if not passed:
        raise AssertionError(
            "Secure end-to-end MoE output "
            "does not match plaintext output."
        )

if __name__ == "__main__":
    run_end_to_end_test()
