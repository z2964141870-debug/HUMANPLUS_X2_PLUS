#!/usr/bin/env python3
"""Read-only SmartWear BLE scanner and V2 notification probe."""

from __future__ import annotations

import argparse
import asyncio
import struct
import time

from bleak import BleakClient, BleakScanner


NOTIFY_UUID = "adaf0101-c332-42a8-93bd-25e905756cb8"
FRAME_SIZE = 137
TARGET_MACS = {
    "jacket": "FB:08:4D:3B:D6:06",
    "pants": "CC:D1:FA:DD:6D:B8",
}


def mac_bytes(mac: str) -> tuple[bytes, bytes]:
    raw = bytes.fromhex(mac.replace(":", ""))
    return raw, raw[::-1]


def advertised_mac_matches(adv, mac: str) -> bool:
    needles = mac_bytes(mac)
    blobs = list((adv.manufacturer_data or {}).values())
    blobs.extend((adv.service_data or {}).values())
    return any(needle in bytes(blob) for blob in blobs for needle in needles)


def validate_v2(frame: bytes) -> tuple[bool, str]:
    if len(frame) != FRAME_SIZE:
        return False, f"length={len(frame)}"
    if frame[:2] != b"\xff\xfe":
        return False, f"header={frame[:2].hex()}"
    if struct.unpack_from("<H", frame, 2)[0] != FRAME_SIZE:
        return False, "embedded_length_invalid"
    if (sum(frame[:136]) & 0xFF) != frame[136]:
        return False, "checksum_invalid"
    magic, version, flags, utc_ns, _ = struct.unpack_from("<2sBBQB", frame, 124)
    if magic != b"UT" or version != 2 or flags & ~1:
        return False, f"ut={magic!r}/version={version}/flags={flags}"
    if bool(flags & 1) != bool(utc_ns):
        return False, f"utc_flag_value_mismatch flags={flags} utc_ns={utc_ns}"
    return True, f"V2 utc_synced={bool(flags & 1)} utc_ns={utc_ns}"


async def probe(address: str, label: str, seconds: float) -> None:
    total = valid = 0
    reasons: dict[str, int] = {}
    first_valid = None

    def notify(_sender, data: bytearray) -> None:
        nonlocal total, valid, first_valid
        total += 1
        ok, reason = validate_v2(bytes(data))
        if ok:
            valid += 1
            first_valid = first_valid or reason
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    print(f"PROBE {label}: connecting to CoreBluetooth id {address}", flush=True)
    async with BleakClient(address, timeout=15) as client:
        services = [str(s.uuid).lower() for s in client.services]
        print(f"PROBE {label}: connected services={services}", flush=True)
        await client.start_notify(NOTIFY_UUID, notify)
        await asyncio.sleep(seconds)
        await client.stop_notify(NOTIFY_UUID)
    print(
        f"RESULT {label}: notifications={total} valid_v2={valid} "
        f"first_valid={first_valid!r} invalid={reasons}",
        flush=True,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-seconds", type=float, default=20.0)
    parser.add_argument("--probe-seconds", type=float, default=5.0)
    parser.add_argument(
        "--connect-uuid",
        action="append",
        default=[],
        help="Explicit macOS CoreBluetooth UUID to probe (repeatable)",
    )
    args = parser.parse_args()

    print(f"Scanning for {args.scan_seconds:.1f}s...", flush=True)
    discovered = await BleakScanner.discover(
        timeout=args.scan_seconds, return_adv=True
    )
    candidates: list[tuple[str, str]] = []
    print(f"SCAN found={len(discovered)}", flush=True)
    for address, (device, adv) in sorted(
        discovered.items(), key=lambda item: item[1][1].rssi, reverse=True
    ):
        services = [str(x).lower() for x in (adv.service_uuids or [])]
        labels = [
            label for label, mac in TARGET_MACS.items()
            if advertised_mac_matches(adv, mac)
        ]
        has_notify_service = NOTIFY_UUID in services
        name = device.name or adv.local_name or ""
        print(
            f"DEVICE rssi={adv.rssi:4d} name={name!r} id={address} "
            f"notify_service={has_notify_service} mac_match={labels} "
            f"services={services}",
            flush=True,
        )
        if labels:
            candidates.append((address, "+".join(labels)))
        elif has_notify_service:
            candidates.append((address, name or "smartwear-service"))

    for index, address in enumerate(args.connect_uuid):
        candidates.append((address, f"explicit-{index + 1}"))

    unique = list(dict.fromkeys(candidates))
    if not unique:
        print(
            "NO_UNAMBIGUOUS_TARGET: macOS hides physical BLE MAC addresses. "
            "Use a DEVICE id above with --connect-uuid only after identifying it.",
            flush=True,
        )
        return 2
    if len(unique) > 2:
        print(f"AMBIGUOUS_TARGETS: refusing to connect automatically: {unique}")
        return 3

    for address, label in unique:
        try:
            await probe(address, label, args.probe_seconds)
        except Exception as exc:
            print(f"RESULT {label}: ERROR {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
