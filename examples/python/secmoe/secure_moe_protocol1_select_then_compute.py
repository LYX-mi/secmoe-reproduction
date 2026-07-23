"""Protocol-1-aligned Select-Then-Compute secure MoE.

This experiment maps SecMoE Protocol 1 to existing
OpenBumbleBee/SPU/CHEETAH primitives.

Party roles:
- rank 0: client
- rank 1: model server

The client input is secret-shared.
The router and expert weights are owned by rank 1 and use
SPU colocated private-value optimization.

This version aligns the execution roles and existing SPU
subprotocols, but it does not expose SecMoE's internal HE
ciphertext objects directly at the Python level.
"""

from concurrent.futures import ThreadPoolExecutor

import jax

try:
    import jax.extend.linear_util as jax_lu
except ImportError:
    import jax.linear_util as jax_lu

import jax.numpy as jnp
import numpy as np

from jax._src import api_util as japi_util

import spu.api as spu_api
import spu.intrinsic as si
import spu.spu_pb2 as spu_pb2
import spu.utils.frontend as spu_fe

from spu import libspu

from secure_moe_correctness import (
    plaintext_top1_moe,
)
from secure_moe_correctness import (
    secure_softmax,
)
from switch_moe_reference import MoEConfig
from switch_moe_reference import (
    initialize_parameters,
)


CLIENT_RANK = 0
SERVER_RANK = 1
WORLD_SIZE = 2


def read_storage_type(share) -> str:
    """Read the SPU runtime storage type of one share."""

    metadata = spu_pb2.ValueMetaProto()
    metadata.ParseFromString(
        share.meta
    )

    return metadata.storage_type


class RoleAwareSimulator:
    """Run SPU with client-owned and server-owned inputs."""

    def __init__(self):
        runtime_config = spu_pb2.RuntimeConfig(
            protocol=(
                spu_pb2.ProtocolKind.CHEETAH
            ),
            field=spu_pb2.FieldType.FM64,
            fxp_fraction_bits=18,
            experimental_enable_colocated_optimization=True,
        )

        self.runtime_config = runtime_config
        self.io = spu_api.Io(
            WORLD_SIZE,
            runtime_config,
        )

    def make_argument_shares(
        self,
        flat_arguments,
        owner_ranks,
    ):
        """Create secret or single-party-owned inputs."""

        if len(flat_arguments) != len(
            owner_ranks
        ):
            raise ValueError(
                "flat_arguments and owner_ranks "
                "must have the same length"
            )

        parameters = []
        storage_types = []

        for argument, owner_rank in zip(
            flat_arguments,
            owner_ranks,
        ):
            argument_array = np.asarray(
                jnp.asarray(argument)
            )

            if owner_rank is None:
                shares = self.io.make_shares(
                    argument_array,
                    spu_pb2.Visibility.VIS_SECRET,
                )
            else:
                shares = self.io.make_shares(
                    argument_array,
                    spu_pb2.Visibility.VIS_SECRET,
                    owner_rank=owner_rank,
                )

            parameters.append(shares)
            storage_types.append(
                read_storage_type(
                    shares[0]
                )
            )

        return parameters, storage_types

    def execute(
        self,
        executable,
        flat_arguments,
        owner_ranks,
    ):
        """Execute one compiled program on two in-memory parties."""

        parameters, storage_types = (
            self.make_argument_shares(
                flat_arguments,
                owner_ranks,
            )
        )

        link_description = (
            libspu.link.Desc()
        )

        for rank in range(WORLD_SIZE):
            link_description.add_party(
                f"id_{rank}",
                f"thread_{rank}",
            )

        def run_party(rank):
            link_context = (
                libspu.link.create_mem(
                    link_description,
                    rank,
                )
            )

            rank_config = (
                spu_pb2.RuntimeConfig()
            )

            rank_config.CopyFrom(
                self.runtime_config
            )

            if rank != CLIENT_RANK:
                rank_config.enable_action_trace = False
                rank_config.enable_hal_profile = False
                rank_config.enable_pphlo_profile = False

            runtime = spu_api.Runtime(
                link_context,
                rank_config,
            )

            for argument_index, shares in enumerate(
                parameters
            ):
                runtime.set_var(
                    executable.input_names[
                        argument_index
                    ],
                    shares[rank],
                )

            runtime.run(executable)

            return [
                runtime.get_var(output_name)
                for output_name
                in executable.output_names
            ]

        with ThreadPoolExecutor(
            max_workers=WORLD_SIZE
        ) as executor:
            futures = [
                executor.submit(
                    run_party,
                    rank,
                )
                for rank in range(
                    WORLD_SIZE
                )
            ]

            party_outputs = [
                future.result()
                for future in futures
            ]

        grouped_outputs = zip(
            *party_outputs
        )

        reconstructed_outputs = [
            self.io.reconstruct(
                list(output_shares)
            )
            for output_shares
            in grouped_outputs
        ]

        return (
            reconstructed_outputs,
            storage_types,
        )


def compile_protocol1_function(
    function,
    arguments,
):
    """Compile a Protocol-1-aligned JAX function."""

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

    def output_name_generator(
        flat_outputs,
    ):
        return [
            f"out{index}"
            for index in range(
                len(flat_outputs)
            )
        ]

    compiler_options = (
        spu_pb2.CompilerOptions()
    )

    compiler_options.enable_optimize_denominator_with_broadcast = True

    executable, output_template = (
        spu_fe.compile(
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
    )


def protocol1_select_then_compute_moe(
    x,
    router_w,
    w1,
    v,
    w2,
):
    """Execute a Protocol-1-aligned Top-1 secure MoE.

    Protocol mapping:

    Steps 1-3:
        Secure TopK, one-hot generation and implicit B2A.

    Steps 4-5:
        Select W1, V and W2 from server-owned expert weights.

    Steps 6-12:
        Evaluate only one selected expert for each token.

    Steps 13-14:
        The SPU runtime keeps the result secret-shared.
        Reconstruction is performed only by the test harness.
    """

    # Router calculation before Protocol 1.
    router_logits = (
        x @ router_w
    )

    router_probabilities = (
        secure_softmax(
            router_logits,
            axis=-1,
        )
    )

    # Protocol 1, Step 1:
    # Securely obtain the largest router value
    # and its secret-shared expert index.
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

    # Protocol 1, Steps 2-3:
    # Convert the secret expert index into a
    # secret one-hot vector.
    #
    # Equality produces Boolean selection bits.
    # Converting the result to x.dtype causes
    # Boolean-to-arithmetic conversion when needed.
    selection_vector = (
        jax.nn.one_hot(
            selected_experts,
            number_of_experts,
            dtype=x.dtype,
        )
    )

    # Protocol 1, Steps 4-5:
    # Obliviously select the parameters of the
    # chosen expert.
    #
    # selection_mask shape:
    # [batch, experts, 1, 1]
    selection_mask = selection_vector[
        :,
        :,
        None,
        None,
    ]

    selected_w1 = jnp.sum(
        selection_mask
        * w1[None, :, :, :],
        axis=1,
    )

    selected_v = jnp.sum(
        selection_mask
        * v[None, :, :, :],
        axis=1,
    )

    selected_w2 = jnp.sum(
        selection_mask
        * w2[None, :, :, :],
        axis=1,
    )

    # Protocol 1, Steps 6-8:
    # Evaluate W1*x and V*x using only the
    # selected expert parameters.
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

    # Protocol 1, Step 9:
    # BumbleBee secure GeLU.
    activated = si.spu_gelu(
        first_projection
    )

    # Protocol 1, Step 10:
    # Secure element-wise multiplication and
    # fixed-point truncation.
    hidden = (
        activated
        * gate_projection
    )

    # Protocol 1, Steps 11-12:
    # Evaluate the selected expert's W2 layer.
    expert_output = jnp.matmul(
        hidden[:, None, :],
        selected_w2,
    )

    expert_output = jnp.squeeze(
        expert_output,
        axis=1,
    )

    # Use the probability of the selected expert
    # as the Top-1 router gate.
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



def plaintext_protocol1_reference(
    x,
    router_w,
    w1,
    v,
    w2,
):
    """Plaintext Top-1 MoE reference.

    The reference uses ordinary plaintext routing and
    evaluates only the selected expert.

    BumbleBee's secure GeLU is an approximation, so a
    small difference from the exact plaintext GeLU is
    expected.
    """

    x = jnp.asarray(
        x,
        dtype=jnp.float32,
    )

    router_w = jnp.asarray(
        router_w,
        dtype=jnp.float32,
    )

    w1 = jnp.asarray(
        w1,
        dtype=jnp.float32,
    )

    v = jnp.asarray(
        v,
        dtype=jnp.float32,
    )

    w2 = jnp.asarray(
        w2,
        dtype=jnp.float32,
    )

    router_logits = (
        x @ router_w
    )

    router_probabilities = (
        jax.nn.softmax(
            router_logits,
            axis=-1,
        )
    )

    selected_experts = jnp.argmax(
        router_logits,
        axis=-1,
    )

    batch_size = x.shape[0]

    outputs = []

    for token_index in range(
        batch_size
    ):
        expert_index = (
            selected_experts[
                token_index
            ]
        )

        token_x = x[
            token_index
        ]

        first_projection = (
            token_x
            @ w1[expert_index]
        )

        gate_projection = (
            token_x
            @ v[expert_index]
        )

        activated = jax.nn.gelu(
            first_projection,
            approximate=False,
        )

        hidden = (
            activated
            * gate_projection
        )

        expert_output = (
            hidden
            @ w2[expert_index]
        )

        selected_gate = (
            router_probabilities[
                token_index,
                expert_index,
            ]
        )

        outputs.append(
            expert_output
            * selected_gate
        )

    final_output = jnp.stack(
        outputs,
        axis=0,
    )

    selection_vector = (
        jax.nn.one_hot(
            selected_experts,
            w1.shape[0],
            dtype=x.dtype,
        )
    )

    return (
        np.asarray(final_output),
        np.asarray(router_logits),
        np.asarray(
            router_probabilities
        ),
        np.asarray(
            selected_experts
        ),
        np.asarray(
            selection_vector
        ),
    )


def create_protocol1_test_data(
    seed=2026,
    batch_size=2,
    d_model=16,
    d_ff=32,
    number_of_experts=4,
):
    """Create deterministic small MoE test data."""

    random_generator = (
        np.random.default_rng(
            seed
        )
    )

    # Keep the numerical range moderate so that
    # fixed-point and polynomial approximation errors
    # remain easy to observe.
    x = random_generator.normal(
        loc=0.0,
        scale=0.25,
        size=(
            batch_size,
            d_model,
        ),
    ).astype(
        np.float32
    )

    router_w = random_generator.normal(
        loc=0.0,
        scale=0.20,
        size=(
            d_model,
            number_of_experts,
        ),
    ).astype(
        np.float32
    )

    w1 = random_generator.normal(
        loc=0.0,
        scale=0.15,
        size=(
            number_of_experts,
            d_model,
            d_ff,
        ),
    ).astype(
        np.float32
    )

    v = random_generator.normal(
        loc=0.0,
        scale=0.15,
        size=(
            number_of_experts,
            d_model,
            d_ff,
        ),
    ).astype(
        np.float32
    )

    w2 = random_generator.normal(
        loc=0.0,
        scale=0.15,
        size=(
            number_of_experts,
            d_ff,
            d_model,
        ),
    ).astype(
        np.float32
    )

    return (
        x,
        router_w,
        w1,
        v,
        w2,
    )


def maximum_absolute_error(
    actual,
    expected,
):
    """Return maximum absolute numerical difference."""

    actual = np.asarray(
        actual
    )

    expected = np.asarray(
        expected
    )

    return float(
        np.max(
            np.abs(
                actual
                - expected
            )
        )
    )


def inspect_compiled_pphlo(
    executable,
):
    """Inspect whether important operations exist."""

    pphlo_text = (
        executable.code.decode(
            "utf-8"
        )
    )

    lowered_text = (
        pphlo_text.lower()
    )

    checks = {
        "contains_topk_or_sort": (
            "topk" in lowered_text
            or "sort" in lowered_text
        ),
        "contains_gelu": (
            "spu.gelu" in lowered_text
            or "gelu" in lowered_text
        ),
        "contains_dot_or_matmul": (
            "dot" in lowered_text
            or "matmul" in lowered_text
        ),
        "contains_secret_type": (
            "secret" in lowered_text
        ),
    }

    return (
        pphlo_text,
        checks,
    )


def main():
    """Run Protocol-1-aligned correctness experiment."""

    seed = 2026
    batch_size = 2
    d_model = 16
    d_ff = 32
    number_of_experts = 4

    print(
        "===== SecMoE Protocol 1 "
        "Aligned Experiment ====="
    )

    print(
        f"seed: {seed}"
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
        "number_of_experts: "
        f"{number_of_experts}"
    )

    (
        x,
        router_w,
        w1,
        v,
        w2,
    ) = create_protocol1_test_data(
        seed=seed,
        batch_size=batch_size,
        d_model=d_model,
        d_ff=d_ff,
        number_of_experts=(
            number_of_experts
        ),
    )

    arguments = (
        x,
        router_w,
        w1,
        v,
        w2,
    )

    print(
        "\nCompiling Protocol 1..."
    )

    (
        executable,
        flat_arguments,
        _output_tree,
    ) = compile_protocol1_function(
        protocol1_select_then_compute_moe,
        arguments,
    )

    print(
        "Compilation completed."
    )

    # x is ordinary additive secret sharing.
    #
    # Router and expert weights are owned by
    # the model server, rank 1.
    owner_ranks = [
        None,
        SERVER_RANK,
        SERVER_RANK,
        SERVER_RANK,
        SERVER_RANK,
    ]

    simulator = (
        RoleAwareSimulator()
    )

    print(
        "\nRunning two-party "
        "CHEETAH simulation..."
    )

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

    input_names = [
        "client_input_x",
        "server_router_w",
        "server_w1",
        "server_v",
        "server_w2",
    ]

    print(
        "\n===== Runtime Storage Types ====="
    )

    for name, storage_type in zip(
        input_names,
        storage_types,
    ):
        print(
            f"{name}: {storage_type}"
        )

    model_private_checks = [
        "Priv2k" in storage_type
        for storage_type
        in storage_types[1:]
    ]

    server_weights_private = all(
        model_private_checks
    )

    client_input_is_shared = (
        "AShr" in storage_types[0]
        or "Secret" in storage_types[0]
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

    onehot_match = (
        maximum_absolute_error(
            secure_selection_vector,
            plaintext_selection_vector,
        )
        < 1.0e-4
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

    final_output_error = (
        maximum_absolute_error(
            secure_final_output,
            plaintext_final_output,
        )
    )

    # Secure GeLU and secure Softmax use fixed-point
    # approximations. This threshold is intentionally
    # looser than the previous same-approximation test.
    output_tolerance = 3.0e-2

    output_matches = (
        final_output_error
        < output_tolerance
    )

    (
        pphlo_text,
        pphlo_checks,
    ) = inspect_compiled_pphlo(
        executable
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
        "secure_selection_vector:"
    )

    print(
        np.asarray(
            secure_selection_vector
        )
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
        "final_output_error: "
        f"{final_output_error:.10f}"
    )

    print(
        "output_tolerance: "
        f"{output_tolerance:.10f}"
    )

    naive_expert_evaluations = (
        batch_size
        * number_of_experts
    )

    protocol1_expert_evaluations = (
        batch_size
    )

    structural_reduction = (
        naive_expert_evaluations
        / protocol1_expert_evaluations
    )

    print(
        "\n===== Structural Sparsity ====="
    )

    print(
        "naive_all_expert_evaluations: "
        f"{naive_expert_evaluations}"
    )

    print(
        "protocol1_expert_evaluations: "
        f"{protocol1_expert_evaluations}"
    )

    print(
        "structural_reduction: "
        f"{structural_reduction:.2f}x"
    )

    print(
        "\n===== Compiled PPHLO Checks ====="
    )

    for check_name, check_value in (
        pphlo_checks.items()
    ):
        print(
            f"{check_name}: "
            f"{check_value}"
        )

    print(
        "\n===== Correctness Summary ====="
    )

    print(
        "client_input_is_shared:",
        client_input_is_shared,
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
        "onehot_match:",
        onehot_match,
    )

    print(
        "output_matches:",
        output_matches,
    )

    all_passed = all(
        [
            client_input_is_shared,
            server_weights_private,
            selected_experts_match,
            onehot_match,
            output_matches,
        ]
    )

    print(
        "all_passed:",
        all_passed,
    )

    # Save PPHLO for later protocol-level analysis.
    pphlo_output_path = (
        "/tmp/"
        "secmoe_protocol1.pphlo"
    )

    with open(
        pphlo_output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            pphlo_text
        )

    print(
        "\nPPHLO saved to:",
        pphlo_output_path,
    )

    if not all_passed:
        raise RuntimeError(
            "Protocol 1 aligned experiment failed. "
            "Check the summary above."
        )


if __name__ == "__main__":
    main()
