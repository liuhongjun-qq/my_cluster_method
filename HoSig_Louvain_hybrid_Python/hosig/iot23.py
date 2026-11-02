from collections import Counter
from datetime import datetime, timezone
import hashlib
import ipaddress
from pathlib import Path
import re

from ._01data import Event, parse_timestamp, write_events
from .reporting import write_json


def _unescape(value):
    return re.sub(r"\\x([0-9a-fA-F]{2})", lambda match: chr(int(match.group(1), 16)), value)


def read_zeek_rows(path):
    separator = "\t"
    fields = None
    unset_field, empty_field = "-", "(empty)"
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("#separator "):
                separator = _unescape(line.split(" ", 1)[1])
                if len(separator) != 1:
                    raise ValueError("Zeek separator must decode to a single character")
            elif line.startswith("#unset_field" + separator):
                unset_field = line.split(separator, 1)[1]
            elif line.startswith("#empty_field" + separator):
                empty_field = line.split(separator, 1)[1]
            elif line.startswith("#fields" + separator):
                fields = line.split(separator)[1:]
                if len(fields) != len(set(fields)):
                    raise ValueError(f"Duplicate Zeek fields at line {line_number}")
            elif line.startswith("#"):
                continue
            else:
                if fields is None:
                    raise ValueError(f"Missing #fields header before line {line_number}")
                values = line.split(separator)
                if len(values) != len(fields) and fields[-3:-1] == ["tunnel_parents", "label"] and fields[-1] in {"detailed-label", "det_label", "detailed_label"}:
                    prefix = values[:len(fields) - 3]
                    tail = separator.join(values[len(fields) - 3:]).split()
                    if len(tail) == 3:
                        values = prefix + tail
                if len(values) != len(fields):
                    raise ValueError(f"Zeek field count mismatch at line {line_number}: expected {len(fields)}, got {len(values)}")
                row = {
                    name: None if value == unset_field else "" if value == empty_field else _unescape(value)
                    for name, value in zip(fields, values)
                }
                yield line_number, row
        if fields is None:
            raise ValueError("Missing Zeek #fields header")


def read_iot23_events(path, label_mode="detailed", scenario="iot23"):
    if label_mode not in {"detailed", "binary"}:
        raise ValueError("label_mode must be detailed or binary")
    for line_number, row in read_zeek_rows(path):
        try:
            required = ("ts", "uid", "id.orig_h", "id.resp_h", "proto", "label")
            missing = [name for name in required if not row.get(name)]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
            ipaddress.ip_address(row["id.orig_h"])
            ipaddress.ip_address(row["id.resp_h"])
            context = {
                name: row[name] for name in ("proto", "service", "id.resp_p", "conn_state")
                if row.get(name) not in (None, "")
            }
            tokens = [f"{name}={value}" for name, value in context.items()]
            coarse = row["label"]
            detailed = row.get("det_label") or row.get("detailed-label") or row.get("detailed_label")
            label = coarse if label_mode == "binary" else detailed or coarse
            yield Event(
                timestamp=parse_timestamp(row["ts"]), src_ip=row["id.orig_h"],
                dst_ip=row["id.resp_h"], payload=" ".join(tokens), context=context,
                label=label, event_id=f"{scenario}:{row['uid']}",
            )
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"Invalid IoT-23 event at line {line_number}: {error}") from error


def convert_iot23(source, output, label_mode="detailed", scenario=None, order="timestamp", limit=None):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Input and output must be different files")
    if order not in {"timestamp", "source"}:
        raise ValueError("order must be timestamp or source")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    scenario = scenario or source.parent.name
    events = list(read_iot23_events(source, label_mode, scenario))
    if not events:
        raise ValueError("IoT-23 log contains no events")
    full_count = len(events)
    full_labels = Counter(event.label for event in events)
    inversions = sum(current.timestamp < previous.timestamp for previous, current in zip(events[:-1], events[1:]))
    if order == "timestamp":
        events.sort(key=lambda event: event.timestamp)
    if limit is not None:
        events = events[:limit]
    write_events(output, events)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    with output.open("rb") as handle:
        output_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    times = [event.timestamp for event in events]
    metadata = {
        "dataset": "IoT-23", "scenario": scenario, "source_path": str(source.resolve()),
        "source_sha256": source_hash, "output_sha256": output_hash,
        "source_records": full_count, "converted_records": len(events), "limit": limit,
        "source_label_counts": dict(sorted(full_labels.items())),
        "converted_label_counts": dict(sorted(Counter(event.label for event in events).items())),
        "label_mode": label_mode, "source_timestamp_inversions": inversions, "order": order,
        "timestamp_source": "Zeek ts: real connection first-packet timestamp, not alert arrival timestamp",
        "alert_arrival_times_available": False,
        "timestamps_real": True, "payload_real": False,
        "payload_mode": "non_temporal_zeek_connection_metadata_proxy",
        "proxy_fields": ["proto", "service", "id.resp_p", "conn_state"],
        "excluded_from_features": ["label", "det_label", "detailed-label", "duration", "ts", "uid", "bytes", "packet_counts"],
        "replay_protocol": "offline stable chronological replay" if order == "timestamp" else "original Zeek log order",
        "first_timestamp_utc": datetime.fromtimestamp(min(times), timezone.utc).isoformat(),
        "last_timestamp_utc": datetime.fromtimestamp(max(times), timezone.utc).isoformat(),
        "span_seconds": max(times) - min(times),
        "warning": "Real timestamps and IPs, but payload is a log-metadata proxy; behavior labels are not campaign ground truth. Not paper data.",
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata
