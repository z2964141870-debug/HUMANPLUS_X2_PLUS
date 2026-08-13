# Phase72 attempt0 implementation failure

Decision: `FAIL_IMPLEMENTATION_STOP`. No Phase72 scientific result exists.

The first registered launch (`seed0_plus`) completed all 400 steps for 64 environments and wrote a finite raw evidence bundle. Screen serialization then failed because `technical["finite"]` was a zero-dimensional PyTorch tensor, which `json.dumps` cannot encode. The registered screen was never written.

The resource ledger reports child exit code 0 even though its immutable log contains the serialization traceback, so that exit code is not reliable for this attempt. The enclosing run script stopped on the missing screen before `seed0_minus`; one launch was consumed and the preregistration forbids a replacement or any remaining Phase72 launch.

The raw bundle contains no optimizer, model-state, or checkpoint payload. It is failure-diagnostic evidence only and cannot support a pair result, scientific conclusion, training unlock, export, or deployment claim.
