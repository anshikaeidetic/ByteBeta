# Support Tiers

Byte distinguishes between core runtime surfaces, optional backends with active
test coverage, and experimental integrations that need further service-backed
validation before they should be treated as release-critical.

## Core runtime

These surfaces are expected to work on the base repository path and are part of
the default engineering bar:

- core cache runtime and gateway behavior
- benchmark planning and local comparison entrypoints
- provider-free benchmark plan imports

## Optional backends under active validation

These backends are supported, but they depend on optional extras and should be
validated in dedicated backend lanes instead of being inferred from the base
coverage number alone:

- Redis eviction
- S3 object storage
- Dynamo, Mongo, and Redis scalar stores
- FAISS, Milvus, pgvector, Qdrant, Usearch, Weaviate, and DocArray vector stores

Coverage and test reporting for these backends should be read together with the
matching optional dependency and service lane.

## Experimental or incomplete validation

A backend should remain in this tier until the repository has deterministic
tests or documented service-backed validation for it. Experimental status is a
support statement, not a claim that the backend is broken.

When a backend remains experimental, README and release notes should say so
explicitly instead of allowing the base coverage figure to imply production
validation.
