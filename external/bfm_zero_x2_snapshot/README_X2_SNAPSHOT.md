# BFM-Zero X2 scratch snapshot

This directory is the X2-specific overlay produced in the local
`BFM-Zero` checkout. It is part of the HUMANPLUS X2 project, while the
upstream BFM-Zero history remains separate.

- Upstream repository: `https://github.com/LeCAR-Lab/BFM-Zero.git`
- Upstream/local base commit: `918c3061633e0512bf2a586ef86a63508f4e6409`
- Local X2 overlay commit: `e355752c46c32271eaccad084876c61da068e8e5`
- Overlay contents: 153 changed or added source, test, script, and report files
- Model initialization: fresh BFM student; no Stage219 weights loaded into the student
- Verification: 31 X2 tests passed on 2026-08-14

The principal result is
`reports/x2_scratch_response_teacher_v7_result.json`. The response-domain
teacher dataset contains 204,800 frames. The resulting student survived
512/512 lanes for 400 control steps in both response and ideal evaluation
domains. This supports the diagnosis that earlier failures were primarily a
closed-loop/domain-coverage problem, not evidence that scratch training was
impossible.

Large model and optimizer files are intentionally excluded from Git. Their
per-file SHA256 inventory is stored in `ARTIFACT_MANIFEST.json`; the files are
backed up under the Baidu remote root recorded by that manifest. The completed
156-file remote byte audit is recorded in `BAIDU_SYNC_RESULT.json`.
