"""Public Byte operator surface with stable manifest-builder exports."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from byte_server._operator_manifest import (
    GROUP,
    PLURAL,
    VERSION,
    build_deployment,
    build_inference_resources,
    build_memory_resources,
    build_persistent_volume_claim,
    build_server_args,
    build_service,
)
from byte_server._operator_reconcile import reconcile_resource

try:  # pragma: no cover
    import kopf
except ImportError:  # pragma: no cover
    kopf = None


LOGGER = logging.getLogger(__name__)

__all__ = [
    "build_deployment",
    "build_inference_resources",
    "build_memory_resources",
    "build_persistent_volume_claim",
    "build_server_args",
    "build_service",
    "main",
    "reconcile",
    "reconcile_resource",
]


def reconcile(
    spec: dict[str, Any],
    name: str,
    namespace: str,
    body: dict[str, Any],
    patch: Any,
    logger: Any,
    **_kwargs: Any,
) -> dict[str, str]:  # pragma: no cover
    """Kopf handler that reconciles a ByteCache CR into Kubernetes manifests."""
    logger.info("Reconciling ByteCache %s/%s", namespace, name)
    results = reconcile_resource(name, namespace, spec, body)
    patch.status["phase"] = "Ready"
    patch.status["resources"] = results
    patch.status["serviceName"] = name
    return results


if kopf is not None:  # pragma: no cover
    reconcile = kopf.on.create(GROUP, VERSION, PLURAL)(reconcile)
    reconcile = kopf.on.update(GROUP, VERSION, PLURAL)(reconcile)

    @kopf.on.delete(GROUP, VERSION, PLURAL)
    def cleanup(name: str, namespace: str, logger: Any, **_kwargs: Any) -> None:
        logger.info(
            "ByteCache %s/%s deleted; owned resources will be garbage-collected.", namespace, name
        )


def main() -> None:
    """Run the Kopf operator entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--namespace",
        default=os.getenv("BYTE_OPERATOR_NAMESPACE", ""),
        help="watch a single namespace; omit to run cluster-wide",
    )
    args = parser.parse_args()

    if kopf is None:  # pragma: no cover
        raise RuntimeError(
            "byte_operator requires the optional operator dependencies. "
            "Install the package with byte[operator]."
        )

    run_kwargs: dict[str, Any] = {}
    if args.namespace:
        run_kwargs["namespaces"] = [args.namespace]
    kopf.run(clusterwide=not bool(args.namespace), standalone=True, **run_kwargs)
