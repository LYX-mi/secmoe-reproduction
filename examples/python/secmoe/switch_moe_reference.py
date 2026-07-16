"""Plaintext Switch-style MoE reference for SecMoE reproduction.

This program builds a small plaintext MoE model as the reference
for later secure SecMoE correctness and sparsity experiments.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MoEConfig:
    """Configuration of the small plaintext MoE experiment."""

    batch_size: int = 8
    d_model: int = 64
    d_ff: int = 128
    num_experts: int = 4
    seed: int = 2026
    tolerance: float = 1e-5

def initialize_parameters(config: MoEConfig):
    """Create deterministic inputs, router weights, and expert weights."""

    key = jax.random.PRNGKey(config.seed)
    x_key, router_key, w1_key, v_key, w2_key = jax.random.split(key, 5)

    x = jax.random.normal(
        x_key,
        (config.batch_size, config.d_model),
    ) * 0.05

    router_w = jax.random.normal(
        router_key,
        (config.d_model, config.num_experts),
    ) * 0.05

    w1 = jax.random.normal(
        w1_key,
        (config.num_experts, config.d_model, config.d_ff),
    ) * 0.05

    v = jax.random.normal(
        v_key,
        (config.num_experts, config.d_model, config.d_ff),
    ) * 0.05

    w2 = jax.random.normal(
        w2_key,
        (config.num_experts, config.d_ff, config.d_model),
    ) * 0.05

    return x, router_w, w1, v, w2

def expert_forward(
    x: jnp.ndarray,
    w1: jnp.ndarray,
    v: jnp.ndarray,
    w2: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the output of one gated MoE expert."""

    gelu_branch = jax.nn.gelu(x @ w1)
    linear_branch = x @ v
    gated_hidden = gelu_branch * linear_branch
    output = gated_hidden @ w2

    return output

def compute_router(
    x: jnp.ndarray,
    router_w: jnp.ndarray,
):
    """Compute router logits, probabilities, and Top-1 experts."""

    router_logits = x @ router_w
    router_probabilities = jax.nn.softmax(
        router_logits,
        axis=-1,
    )

    selected_from_logits = jnp.argmax(
        router_logits,
        axis=-1,
    )

    selected_from_probabilities = jnp.argmax(
        router_probabilities,
        axis=-1,
    )

    argmax_consistent = bool(
        jnp.all(
            selected_from_logits
            == selected_from_probabilities
        )
    )

    return (
        router_logits,
        router_probabilities,
        selected_from_probabilities,
        argmax_consistent,
    )

def sparse_top1_forward(
    x: jnp.ndarray,
    router_probabilities: jnp.ndarray,
    selected_experts: jnp.ndarray,
    w1: jnp.ndarray,
    v: jnp.ndarray,
    w2: jnp.ndarray,
):
    """Evaluate only the selected Top-1 expert for each token."""

    token_outputs = []

    for token_id in range(x.shape[0]):
        expert_id = int(selected_experts[token_id])

        expert_output = expert_forward(
            x[token_id : token_id + 1],
            w1[expert_id],
            v[expert_id],
            w2[expert_id],
        )

        gate_value = router_probabilities[
            token_id,
            expert_id,
        ]

        gated_output = gate_value * expert_output
        token_outputs.append(gated_output)

    output = jnp.concatenate(
        token_outputs,
        axis=0,
    )

    token_expert_evaluations = x.shape[0]

    return output, token_expert_evaluations

def all_experts_forward(
    x: jnp.ndarray,
    router_probabilities: jnp.ndarray,
    selected_experts: jnp.ndarray,
    w1: jnp.ndarray,
    v: jnp.ndarray,
    w2: jnp.ndarray,
):
    """Evaluate every expert, then select the Top-1 expert output."""

    all_outputs = []

    for expert_id in range(w1.shape[0]):
        expert_output = expert_forward(
            x,
            w1[expert_id],
            v[expert_id],
            w2[expert_id],
        )

        all_outputs.append(expert_output)

    stacked_outputs = jnp.stack(
        all_outputs,
        axis=1,
    )

    token_ids = jnp.arange(x.shape[0])

    selected_outputs = stacked_outputs[
        token_ids,
        selected_experts,
        :,
    ]

    selected_gates = router_probabilities[
        token_ids,
        selected_experts,
    ]

    output = selected_outputs * selected_gates[:, None]

    token_expert_evaluations = (
        x.shape[0] * w1.shape[0]
    )

    return output, token_expert_evaluations

def run_reference_test(config: MoEConfig) -> None:
    """Compare sparse Top-1 execution with all-expert execution."""

    x, router_w, w1, v, w2 = initialize_parameters(config)

    (
        router_logits,
        router_probabilities,
        selected_experts,
        argmax_consistent,
    ) = compute_router(
        x,
        router_w,
    )

    sparse_output, sparse_evaluations = sparse_top1_forward(
        x,
        router_probabilities,
        selected_experts,
        w1,
        v,
        w2,
    )

    naive_output, naive_evaluations = all_experts_forward(
        x,
        router_probabilities,
        selected_experts,
        w1,
        v,
        w2,
    )

    sparse_output.block_until_ready()
    naive_output.block_until_ready()

    absolute_error = jnp.abs(
        sparse_output - naive_output
    )

    max_error = float(
        jnp.max(absolute_error)
    )

    mean_error = float(
        jnp.mean(absolute_error)
    )

    expected_sparse_evaluations = config.batch_size

    expected_naive_evaluations = (
        config.batch_size * config.num_experts
    )

    passed = (
        argmax_consistent
        and sparse_evaluations
        == expected_sparse_evaluations
        and naive_evaluations
        == expected_naive_evaluations
        and max_error < config.tolerance
    )
    selected_gate_values = router_probabilities[
        jnp.arange(config.batch_size),
        selected_experts,
    ]

    print("===== Switch/SecMoE Plaintext Reference =====")
    print(f"seed: {config.seed}")
    print(f"batch_size: {config.batch_size}")
    print(f"d_model: {config.d_model}")
    print(f"d_ff: {config.d_ff}")
    print(f"num_experts: {config.num_experts}")
    print(
        "selected_experts:",
        np.asarray(selected_experts).tolist(),
    )
    print(
        "selected_gate_values:",
        np.asarray(selected_gate_values).round(6).tolist(),
    )
    print(f"argmax_consistent: {argmax_consistent}")
    print(
        "sparse_token_expert_evaluations:",
        sparse_evaluations,
    )
    print(
        "naive_token_expert_evaluations:",
        naive_evaluations,
    )
    print(f"max_error: {max_error:.10f}")
    print(f"mean_error: {mean_error:.10f}")
    print(f"passed: {passed}")

if __name__ == "__main__":
    run_reference_test(MoEConfig())
