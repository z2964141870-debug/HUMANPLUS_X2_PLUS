#!/usr/bin/env python3
"""Execute the frozen Phase74 contract as a new mixed-device-safe Phase75 technical phase."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PHASE74 = ROOT / "scripts/run_x2_phase74_pairing_preflight.py"
EXPECTED_PHASE74_SHA256 = "3430a5130f8c04ea5083e932f387f69a91e781564333568c2de060e6187effcc"


source = FROZEN_PHASE74.read_text(encoding="utf-8")
if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_PHASE74_SHA256:
    raise RuntimeError("frozen Phase74 runner drifted before Phase75 transform")

source = source.replace("Phase74", "Phase75").replace("phase74", "phase75")
replacements = {
    "    tensor_hash,\n)": "    tensor_hash,\n    select_rows,\n)",
    "            value[:, recipients] = value[:, donors].clone()": (
        "            copy_pair_rows(value.transpose(0, 1), donors, recipients)"
    ),
    '                "donor_sha256": tensor_hash(value[donors]),\n'
    '                "recipient_sha256": tensor_hash(value[recipients]),': (
        '                "donor_sha256": tensor_hash(select_rows(value, donors)),\n'
        '                "recipient_sha256": tensor_hash(select_rows(value, recipients)),'
    ),
    '    commit = {\n': (
        '    tensor_sides = {}\n'
        '    for field_name, field_value in sorted(tensors.items()):\n'
        '        try:\n'
        '            tensor_sides[field_name] = {\n'
        '                "dtype": str(field_value.dtype), "device": str(field_value.device),\n'
        '                "shape": list(field_value.shape),\n'
        '                "donor_sha256": tensor_hash(select_rows(field_value, donors)),\n'
        '                "recipient_sha256": tensor_hash(select_rows(field_value, recipients)),\n'
        '            }\n'
        '        except BaseException as field_exc:\n'
        '            raise RuntimeError(f"Phase75 evidence selection failed for {field_name}") from field_exc\n'
        '    cpu_manifest_fields = sorted(\n'
        '        name for name, (value, _, _) in state_entries.items() if value.device.type == "cpu"\n'
        '    )\n'
        '    commit = {\n'
    ),
    '        "tensor_sides": {\n'
    '            name: {\n'
    '                "dtype": str(value.dtype), "shape": list(value.shape),\n'
    '                "donor_sha256": tensor_hash(select_rows(value, donors)),\n'
    '                "recipient_sha256": tensor_hash(select_rows(value, recipients)),\n'
    '            }\n'
    '            for name, value in sorted(tensors.items())\n'
    '        },': (
        '        "tensor_sides": tensor_sides,\n'
        '        "cpu_manifest_fields": cpu_manifest_fields,\n'
        '        "mixed_device_evidence_path_exercised": bool(cpu_manifest_fields),'
    ),
    '        diagnostics[name] = pair_diagnostics(value, donors, recipients, atol=threshold)': (
        '        try:\n'
        '            diagnostics[name] = pair_diagnostics(value, donors, recipients, atol=threshold)\n'
        '        except BaseException as field_exc:\n'
        '            raise RuntimeError(f"Phase75 diagnostics failed for {name}") from field_exc'
    ),
    '        and source_initial["nondegenerate"] is True\n': (
        '        and source_initial["nondegenerate"] is True\n'
        '        and bool(cpu_manifest_fields)\n'
    ),
    '        "unknown_mutable_tensor_fields_empty": not unknown_fields,\n': (
        '        "unknown_mutable_tensor_fields_empty": not unknown_fields,\n'
        '        "cpu_manifest_fields": cpu_manifest_fields,\n'
        '        "mixed_device_evidence_path_exercised": bool(cpu_manifest_fields),\n'
    ),
    '        "phase75_shadow_preregistration_unlocked": False,': (
        '        "phase76_shadow_preregistration_unlocked": False,'
    ),
    '        "phase75_scientific_preregistration_unlocked": False,': (
        '        "phase76_scientific_preregistration_unlocked": False,'
    ),
    '        "phase75_launch_unlocked": False,': '        "phase76_launch_unlocked": False,',
}
for old, new in replacements.items():
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Phase75 transform anchor count changed: {old!r} count={count}")
    source = source.replace(old, new)

if "value[donors]" in source or "value[recipients]" in source or "value[:, recipients]" in source:
    raise RuntimeError("Phase75 transformed runner retains a mixed-device unsafe evidence index")

exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())

