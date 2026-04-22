# Trust Calibration

Byte trust scoring is driven by a versioned calibration artifact rather than constants embedded in
the scoring modules. The active artifact is `byte/trust/calibration/byte-trust-v2.json`.

The artifact records:

- calibration buckets for raw confidence scores;
- named confidence scores, thresholds, and adjustments;
- named risk weights and deterministic-reference scores;
- source status, public-proof status, method, and a checksum over the canonical payload.

The current artifact is marked `internal_checkpoint`. That means its values preserve the retained
Byte v2 trust behavior and are suitable for regression testing, but they are not presented as an
independently calibrated public proof. Public claims about trust-score accuracy must either point to
a benchmark release manifest with raw records and checksums or remain labeled as checkpoint results.
The committed calibration artifact therefore does not ship private validation metrics. Those
metrics may only be published when the matching release manifest and raw checksum bundle are
available in the repository or linked release artifacts.

Run the calibration gate before release:

```bash
python scripts/check_trust_calibration.py
```

The gate validates the artifact checksum, rejects private validation metrics in internal checkpoint
artifacts, requires a manifest for any public-proof calibration status, and rejects unexplained float
literals in the trust scoring modules so score changes are reviewed as calibration changes, not
hidden implementation edits.
