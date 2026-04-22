# Security Policy

## Reporting

Do not disclose security issues in public issues or public pull requests.

Report suspected vulnerabilities privately to the repository owner or through GitHub's private vulnerability reporting flow once it is enabled on `ByteNew`.

Include:

- affected area
- impact
- reproduction steps
- any proposed remediation

## Secure Development Expectations

- Every push must pass hygiene, lint, typecheck, package validation, compile validation, unit tests, and the repo-owned security gate.
- Coverage and integration smoke checks are required alongside the unit and security gates before a change is treated as releasable.
- Secrets, tokens, passwords, and provider keys must never be committed.
- This repository is private. Local pre-push hooks and GitHub Actions are the enforced review gates on the current GitHub plan.

## Scope

This policy covers the Python package, the FastAPI gateway, developer tooling in `scripts`, and repository configuration that affects packaging, CI, or runtime security.
