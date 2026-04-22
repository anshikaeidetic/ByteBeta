# Reports Policy

This repository keeps two benchmark surfaces in git:

- `latest_release_summary.md`
- `public-proof/openai-tier1-local-20260419T154401Z/`

The retained summary is an engineering checkpoint, not a standalone public
release-signoff artifact.

The checked-in `public-proof/` bundle is a reproducible local proof lane with a
release manifest, a compact canonical `engineering_report.json` raw-record
artifact, summary artifacts, and SHA-256 checksums. It is the auditable proof
surface for the current provider-free benchmark contract.

Historical report directories, transient reasoning graphs, and ad hoc local
benchmark runs are still excluded from version control. They are generated
artifacts, not the source of truth for production claims.
