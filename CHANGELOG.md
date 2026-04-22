# Changelog

All notable changes to Byte are documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## [1.0.6] - 2026-04-09

### Added
- package-owned benchmark programs under `byte.benchmarking.programs`, with thin compatibility wrappers in `scripts/`
- repo-root pytest ownership and repo-Python re-exec support for local validation scripts
- lazy optional-dependency loading helpers for ONNX and optional provider stacks
- helper-module coverage for the refactored control-plane, router, trust, pre-context, execution, and server-security seams

### Changed
- standardized public product naming on `Byte` while preserving compatibility identifiers such as `import byte`, `byteai-cache`, `ByteCache`, and the current repository URL
- split the oversized runtime hotspots into thinner facades plus sibling private helper modules
- deduplicated sync and async pipeline execution around shared pipeline stages and stable internal error codes
- moved maintained benchmark logic out of large `scripts/deep_*` and `scripts/advanced_*` files so the package is the source of truth
- replaced deprecated FastAPI shutdown-event wiring with lifespan-managed runtime cleanup
- expanded curated mypy enforcement across config, router, pipeline, and server orchestration surfaces
- hardened optional test skipping and editable-install developer workflows

### Fixed
- streaming-response lease cleanup in server security middleware
- import-time optional dependency failures for ONNX, OpenRouter, and related feature modules
- lockfile generation leakage of local absolute filesystem paths
- inconsistent `ByteClient` response shapes across cache and proxy modes

## [1.0.1] - 2026-03-14

### Added
- library-native OpenTelemetry instrumentation for cache pipeline stages, report metrics, and per-request latency attribution
- explicit `ruff.toml` and `mypy.ini` configuration files wired into CI, tox, and pre-commit
- user-facing release notes with a documented semantic versioning policy

### Changed
- aligned package metadata on a stable `1.0.1` release
- pinned documentation dependencies for reproducible docs builds
- expanded repository ignore rules to cover generated temp directories, caches, editors, and local databases
- hardened pre-commit to catch large binary artifacts, malformed TOML, and type regressions before commit

### Removed
- tracked Hypothesis example database artifacts from the repository root
