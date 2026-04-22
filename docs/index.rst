Byte
====

Byte is an OpenAI-compatible gateway and safe semantic cache for AI workloads.

It can run as:

- an embedded Python SDK through ``ByteClient``
- a single-node OpenAI-compatible gateway through ``byte start`` or ``byte_server``
- a split-service deployment with ``byte_server`` as the public control plane, ``byte_inference`` as the private worker pool, and ``byte_memory`` as the private memory service

.. include:: toc.rst
