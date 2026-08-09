#!/usr/bin/env python3
"""Decode the bounded Phase32 LD_PRELOAD MuJoCo trace format."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path


MAX_Q = MAX_V = MAX_A = MAX_U = 128
MAX_CONTACTS = 64


class FileHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_char * 16),
        ("format_version", ctypes.c_uint32),
        ("header_size", ctypes.c_uint32),
        ("record_size", ctypes.c_uint32),
        ("max_records", ctypes.c_uint32),
        ("committed_records", ctypes.c_uint64),
        ("dropped_records", ctypes.c_uint64),
        ("process_id", ctypes.c_uint64),
        ("max_time_s", ctypes.c_double),
        ("mujoco_version", ctypes.c_char * 32),
        ("reserved", ctypes.c_char * 160),
    ]


class ContactRecord(ctypes.Structure):
    _fields_ = [
        ("geom1", ctypes.c_int32), ("geom2", ctypes.c_int32),
        ("dim", ctypes.c_int32), ("efc_address", ctypes.c_int32),
        ("distance", ctypes.c_double),
        ("position", ctypes.c_double * 3),
        ("frame", ctypes.c_double * 9),
        ("wrench", ctypes.c_double * 6),
    ]


class StateRecord(ctypes.Structure):
    _fields_ = [
        ("committed", ctypes.c_uint32), ("call_kind", ctypes.c_uint32),
        ("sequence", ctypes.c_uint64), ("thread_id", ctypes.c_uint64),
        ("model_pointer", ctypes.c_uint64), ("data_pointer", ctypes.c_uint64),
        ("nq", ctypes.c_int32), ("nv", ctypes.c_int32),
        ("na", ctypes.c_int32), ("nu", ctypes.c_int32),
        ("ncon", ctypes.c_int32), ("contacts_stored", ctypes.c_int32),
        ("solver_nisland", ctypes.c_int32), ("solver_nefc", ctypes.c_int32),
        ("time_s", ctypes.c_double),
        ("qpos", ctypes.c_double * MAX_Q), ("qvel", ctypes.c_double * MAX_V),
        ("act", ctypes.c_double * MAX_A), ("ctrl", ctypes.c_double * MAX_U),
        ("qacc", ctypes.c_double * MAX_V),
        ("qacc_warmstart", ctypes.c_double * MAX_V),
        ("qfrc_actuator", ctypes.c_double * MAX_V),
        ("qfrc_constraint", ctypes.c_double * MAX_V),
        ("contacts", ContactRecord * MAX_CONTACTS),
    ]


def _values(values: ctypes.Array, count: int) -> list[float]:
    return [float(values[index]) for index in range(max(0, count))]


def decode(path: Path, *, include_rows: bool = False) -> dict:
    payload = path.read_bytes()
    if len(payload) < ctypes.sizeof(FileHeader):
        raise ValueError("trace is smaller than header")
    header = FileHeader.from_buffer_copy(payload)
    magic = bytes(header.magic).split(b"\0", 1)[0]
    if magic != b"X2MJSHIMV1" or header.format_version != 1:
        raise ValueError(f"unsupported trace magic/version: {magic!r}/{header.format_version}")
    if header.header_size != ctypes.sizeof(FileHeader) or header.record_size != ctypes.sizeof(StateRecord):
        raise ValueError("C/Python trace ABI size mismatch")
    expected = header.header_size + header.max_records * header.record_size
    if len(payload) != expected:
        raise ValueError(f"trace byte count mismatch: {len(payload)} != {expected}")
    records = []
    for index in range(header.max_records):
        offset = header.header_size + index * header.record_size
        row = StateRecord.from_buffer_copy(payload, offset)
        if row.committed != 1:
            continue
        contacts = []
        for contact_index in range(row.contacts_stored):
            contact = row.contacts[contact_index]
            contacts.append({
                "geom1": int(contact.geom1), "geom2": int(contact.geom2),
                "dim": int(contact.dim), "efc_address": int(contact.efc_address),
                "distance": float(contact.distance),
                "position": _values(contact.position, 3),
                "frame": _values(contact.frame, 9),
                "wrench": _values(contact.wrench, 6),
            })
        records.append({
            "sequence": int(row.sequence), "call_kind": int(row.call_kind),
            "thread_id": int(row.thread_id), "model_pointer": int(row.model_pointer),
            "data_pointer": int(row.data_pointer), "time_s": float(row.time_s),
            "nq": int(row.nq), "nv": int(row.nv), "na": int(row.na), "nu": int(row.nu),
            "ncon": int(row.ncon), "contacts_stored": int(row.contacts_stored),
            "solver_nisland": int(row.solver_nisland), "solver_nefc": int(row.solver_nefc),
            "qpos": _values(row.qpos, row.nq), "qvel": _values(row.qvel, row.nv),
            "act": _values(row.act, row.na), "ctrl": _values(row.ctrl, row.nu),
            "qacc": _values(row.qacc, row.nv),
            "qacc_warmstart": _values(row.qacc_warmstart, row.nv),
            "qfrc_actuator": _values(row.qfrc_actuator, row.nv),
            "qfrc_constraint": _values(row.qfrc_constraint, row.nv),
            "contacts": contacts,
        })
    records.sort(key=lambda row: row["sequence"])
    by_kind = {str(kind): sum(row["call_kind"] == kind for row in records) for kind in (1, 2, 3)}
    by_data = {}
    for row in records:
        by_data[str(row["data_pointer"])] = by_data.get(str(row["data_pointer"]), 0) + 1
    result = {
        "header": {
            "magic": magic.decode(), "format_version": int(header.format_version),
            "header_size": int(header.header_size), "record_size": int(header.record_size),
            "max_records": int(header.max_records),
            "committed_records_header": int(header.committed_records),
            "committed_rows_scanned": len(records), "dropped_records": int(header.dropped_records),
            "process_id": int(header.process_id), "max_time_s": float(header.max_time_s),
            "mujoco_version": bytes(header.mujoco_version).split(b"\0", 1)[0].decode(),
        },
        "call_counts": by_kind,
        "data_pointer_counts": by_data,
        "sequence_contiguous": [row["sequence"] for row in records] == list(range(len(records))),
    }
    if include_rows:
        result["rows"] = records
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-rows", action="store_true")
    args = parser.parse_args()
    result = decode(args.trace, include_rows=args.include_rows)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
