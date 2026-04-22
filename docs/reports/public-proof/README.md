## Public Proof Bundles

This directory holds benchmark bundles that are intended to be auditable from
the repository itself.

Current bundle:

- `openai-tier1-local-20260419T154401Z/`

Each bundle must include:

- `benchmark-release-manifest.json`
- a raw-record artifact referenced by `raw_records_path`
- a summary artifact referenced by `summary_path`
- SHA-256 checksums for those artifacts

The current bundle is a provider-free local comparison lane. It does not claim
live hosted-provider reproducibility, but it does provide a public proof path
for the in-repo benchmark contract.
