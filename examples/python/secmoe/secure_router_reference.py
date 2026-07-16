"""Secure Top-1 router reference for SecMoE reproduction.

This program compares the plaintext Switch router with a two-party
secure router executed by OpenBumbleBee/SPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim

from switch_moe_reference import MoEConfig
from switch_moe_reference import initialize_parameters

def router_top1(
    x: jnp.ndarray,
    router_w: jnp.ndarray,
):
    """Compute router logits and select one expert for each token."""

    router_logits = jnp.matmul(x, router_w)

    selected_experts = jnp.argmax(
        router_logits,
        axis=-1,
    )

    return router_logits, selected_experts

def create_simulator() -> ppsim.Simulator:
    """Create a two-party CHEETAH simulator over the 64-bit ring."""

    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH,
        field=spu_pb2.FieldType.FM64,
    )

    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = True

    return ppsim.Simulator(
        2,
        config,
    )

def run_secure_router_test() -> None:
    """Compare plaintext and secure Top-1 router results."""

    config = MoEConfig()

    x, router_w, _, _, _ = initialize_parameters(config)

    plain_logits, plain_selected = router_top1(
        x,
        router_w,
    )

    simulator = create_simulator()

    secure_router = ppsim.sim_jax(
        simulator,
        router_top1,
    )

    secure_logits, secure_selected = secure_router(
        x,
        router_w,
    )

    plain_logits_np = np.asarray(plain_logits)
    secure_logits_np = np.asarray(secure_logits)

    plain_selected_np = np.asarray(plain_selected)
    secure_selected_np = np.asarray(secure_selected)

    absolute_error = np.abs(
        plain_logits_np - secure_logits_np
    )

    max_logit_error = float(
        np.max(absolute_error)
    )

    mean_logit_error = float(
        np.mean(absolute_error)
    )

    selection_match = bool(
        np.array_equal(
            plain_selected_np,
            secure_selected_np,
        )
    )

    logit_tolerance = 1e-3

    passed = (
        selection_match
        and max_logit_error < logit_tolerance
    )

    print("===== Secure Top-1 Router Test =====")
    print(f"protocol: CHEETAH")
    print(f"field: FM64")
    print(f"world_size: 2")
    print(
        "plain_selected_experts:",
        plain_selected_np.tolist(),
    )
    print(
        "secure_selected_experts:",
        secure_selected_np.tolist(),
    )
    print(f"selection_match: {selection_match}")
    print(f"max_logit_error: {max_logit_error:.10f}")
    print(f"mean_logit_error: {mean_logit_error:.10f}")
    print(f"logit_tolerance: {logit_tolerance}")
    print(f"passed: {passed}")

    if not passed:
        raise AssertionError(
            "Secure router does not match plaintext router."
        )

if __name__ == "__main__":
    run_secure_router_test()
