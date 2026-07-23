import jax
import jax.numpy as jnp
import numpy as np
from spu import intrinsic as si

from examples.python.secmoe.secure_moe_correctness import (
    secure_softmax,
)
from examples.python.secmoe.secure_moe_protocol1_select_then_compute import (
    SERVER_RANK,
    RoleAwareSimulator,
    compile_protocol1_function,
    create_protocol1_test_data,
    maximum_absolute_error,
    plaintext_protocol1_reference,
)


def select_private_expert_parameters(
    selection_vector,
    expert_parameters,
):
    """Select one server-private expert parameter tensor.

    selection_vector:
        Arithmetic secret sharing, shape [B, E].

    expert_parameters:
        Server-private Priv2k tensor, shape [E, ...].

    The expert dimensions after E are flattened so that
    selection becomes one matrix multiplication:

        [B, E] @ [E, P] -> [B, P]

    At runtime this should dispatch to MatMulAV, whose
    implementation uses CheetahDot::DotOLE.
    """

    number_of_experts = (
        expert_parameters.shape[0]
    )

    flattened_parameters = jnp.reshape(
        expert_parameters,
        (
            number_of_experts,
            -1,
        ),
    )

    selected_flattened = jnp.matmul(
        selection_vector,
        flattened_parameters,
    )

    selected_shape = (
        selection_vector.shape[0],
        *expert_parameters.shape[1:],
    )

    return jnp.reshape(
        selected_flattened,
        selected_shape,
    )


def protocol1_ole_aligned_moe(
    x,
    router_w,
    w1,
    v,
    w2,
):
    """SecMoE Protocol-1 implementation using existing OLE kernels."""

    # Router calculation before Protocol 1.
    router_logits = jnp.matmul(
        x,
        router_w,
    )

    router_probabilities = secure_softmax(
        router_logits,
        axis=-1,
    )

    # Step 1:
    # Secure Top-1.
    _top_values, top_indices = (
        jax.lax.top_k(
            router_logits,
            k=1,
        )
    )

    selected_experts = jnp.squeeze(
        top_indices,
        axis=-1,
    )

    number_of_experts = w1.shape[0]

    # Steps 2-3:
    # Secret index -> Boolean one-hot -> arithmetic one-hot.
    selection_vector = jax.nn.one_hot(
        selected_experts,
        number_of_experts,
        dtype=x.dtype,
    )

    # Steps 4-5:
    # Arithmetic-shared selector multiplied by
    # server-private expert banks.
    #
    # Expected runtime dispatch:
    # MatMulAV -> CheetahDot::DotOLE.
    selected_w1 = (
        select_private_expert_parameters(
            selection_vector,
            w1,
        )
    )

    selected_v = (
        select_private_expert_parameters(
            selection_vector,
            v,
        )
    )

    selected_w2 = (
        select_private_expert_parameters(
            selection_vector,
            w2,
        )
    )

    # Steps 6-8:
    # Both x and selected weights are now arithmetic
    # shares. Batched matrix multiplication should
    # dispatch to BatchMatMulAA.
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

    # Step 9:
    # Reuse BumbleBee's secure segmented GeLU.
    activated = si.spu_gelu(
        first_projection
    )

    # Step 10:
    # Reuse secure multiplication and truncation.
    hidden = (
        activated
        * gate_projection
    )

    # Steps 11-14:
    # Shared GLU result multiplied by the selected
    # W2 shares, producing a shared expert output.
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
        * selection_vector,
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
        selection_vector,
    )


def main():
    seed = 2026
    batch_size = 2
    d_model = 16
    d_ff = 32
    number_of_experts = 4

    print(
        "===== SecMoE Protocol 1 "
        "OLE-Aligned Experiment ====="
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

    (
        x,
        router_w,
        w1,
        v,
        w2,
    ) = arguments

    (
        executable,
        flat_arguments,
        _output_tree,
    ) = compile_protocol1_function(
        protocol1_ole_aligned_moe,
        arguments,
    )

    owner_ranks = [
        None,
        SERVER_RANK,
        SERVER_RANK,
        SERVER_RANK,
        SERVER_RANK,
    ]

    simulator = RoleAwareSimulator()

    (
        secure_outputs,
        storage_types,
    ) = simulator.execute(
        executable,
        flat_arguments,
        owner_ranks,
    )

    (
        secure_final_output,
        secure_router_logits,
        secure_router_probabilities,
        secure_selected_experts,
        secure_selection_vector,
    ) = secure_outputs

    (
        plaintext_final_output,
        plaintext_router_logits,
        plaintext_router_probabilities,
        plaintext_selected_experts,
        plaintext_selection_vector,
    ) = plaintext_protocol1_reference(
        x,
        router_w,
        w1,
        v,
        w2,
    )

    final_output_error = (
        maximum_absolute_error(
            secure_final_output,
            plaintext_final_output,
        )
    )

    router_logit_error = (
        maximum_absolute_error(
            secure_router_logits,
            plaintext_router_logits,
        )
    )

    router_probability_error = (
        maximum_absolute_error(
            secure_router_probabilities,
            plaintext_router_probabilities,
        )
    )

    selected_experts_match = (
        np.array_equal(
            np.asarray(
                secure_selected_experts
            ).reshape(-1),
            np.asarray(
                plaintext_selected_experts
            ).reshape(-1),
        )
    )

    selection_vector_error = (
        maximum_absolute_error(
            secure_selection_vector,
            plaintext_selection_vector,
        )
    )

    server_weights_private = all(
        "Priv2k" in storage_type
        for storage_type
        in storage_types[1:]
    )

    output_matches = (
        final_output_error
        < 3.0e-2
    )

    print(
        "\n===== Runtime Storage Types ====="
    )

    names = [
        "client_input_x",
        "server_router_w",
        "server_w1",
        "server_v",
        "server_w2",
    ]

    for name, storage_type in zip(
        names,
        storage_types,
    ):
        print(
            f"{name}: {storage_type}"
        )

    print(
        "\n===== Routing Results ====="
    )

    print(
        "plaintext_selected_experts:",
        plaintext_selected_experts,
    )

    print(
        "secure_selected_experts:",
        np.asarray(
            secure_selected_experts
        ),
    )

    print(
        "\n===== Numerical Errors ====="
    )

    print(
        "router_logit_error: "
        f"{router_logit_error:.10f}"
    )

    print(
        "router_probability_error: "
        f"{router_probability_error:.10f}"
    )

    print(
        "selection_vector_error: "
        f"{selection_vector_error:.10f}"
    )

    print(
        "final_output_error: "
        f"{final_output_error:.10f}"
    )

    print(
        "\n===== Correctness Summary ====="
    )

    print(
        "server_weights_private:",
        server_weights_private,
    )

    print(
        "selected_experts_match:",
        selected_experts_match,
    )

    print(
        "output_matches:",
        output_matches,
    )

    all_passed = all(
        [
            server_weights_private,
            selected_experts_match,
            selection_vector_error
            < 1.0e-4,
            output_matches,
        ]
    )

    print(
        "all_passed:",
        all_passed,
    )

    pphlo_path = (
        "/tmp/"
        "secmoe_protocol1_ole_aligned.pphlo"
    )

    with open(
        pphlo_path,
        "w",
        encoding="utf-8",
    ) as pphlo_file:
        pphlo_file.write(
            executable.code.decode(
                "utf-8"
            )
        )

    print(
        "\nPPHLO saved to:",
        pphlo_path,
    )

    if not all_passed:
        raise RuntimeError(
            "OLE-aligned Protocol 1 "
            "correctness test failed."
        )


if __name__ == "__main__":
    main()
