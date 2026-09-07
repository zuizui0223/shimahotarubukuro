"""Additional Chapter 3 coverage audit; never a causal or statistical admission gate.

Consumes the six izu-core field CSV channels after their native schema/QC audit.
Extra sampling_block_id on effort/SVD/treatment and effort_purpose on effort
prevent population pooling and assay waiting time from masquerading as a linked
within-plant, within-block natural-observation panel. Missing extensions remain
explicit gaps. Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

Row = Mapping[str, object]
CORE = ("open_pollinated", "bagged_autonomous", "supplemental_outcross")
CONTROL = {"bagged_unvisited_control", "exposed_no_visit_control"}
TERMINAL = {"mature_fruit", "aborted"}
CONTEXT = ("field_event_id", "island_id", "site_id")
CHANNELS = ("plants", "effort", "visits", "svd", "treatments", "fruits")


def text(row: Row, key: str) -> str:
    value = row.get(key, "")
    return "" if value is None else str(value).strip()


def index(rows: Sequence[Row], key: str) -> dict[str, Row]:
    result: dict[str, Row] = {}
    for row in rows:
        value = text(row, key)
        if not value or value in result:
            raise ValueError(f"blank or duplicate {key}: {value!r}")
        result[value] = row
    return result


def count(value: str, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid nonnegative integer: {label}") from exc
    if number < 0:
        raise ValueError(f"negative count: {label}")
    return number


def audit(plants: Sequence[Row], effort: Sequence[Row], visits: Sequence[Row],
          svd: Sequence[Row], treatments: Sequence[Row], fruits: Sequence[Row]) -> dict:
    registry = index(plants, "plant_id")
    efforts, bouts = index(effort, "effort_id"), index(visits, "visit_id")
    index(svd, "svd_id")
    index(treatments, "treatment_id")
    fruit_by_id = index(fruits, "fruit_id")
    buckets: dict[tuple[str, str], dict] = {}
    population_context: dict[str, tuple[str, ...]] = {}
    for plant in plants:
        for field in ("population_id", "taxon", *CONTEXT):
            if not text(plant, field):
                raise ValueError(f"plant registry lacks {field}")
        pop = text(plant, "population_id")
        signature = tuple(text(plant, k) for k in ("island_id", "site_id", "taxon"))
        if population_context.setdefault(pop, signature) != signature:
            raise ValueError("population mixes site/island/taxon")

    def linked(row: Row, extra: tuple[str, ...] = ()) -> str:
        pid = text(row, "plant_id")
        if pid not in registry:
            raise ValueError(f"unregistered plant: {pid!r}")
        for field in (*CONTEXT, *extra):
            if text(row, field) != text(registry[pid], field):
                raise ValueError(f"plant/context mismatch: {pid} {field}")
        return pid

    def bucket(pid: str, block: str) -> dict:
        if (pid, block) not in buckets:
            buckets[pid, block] = {
                "routine_effort_windows": 0, "usable_effort_windows": 0,
                "visit_bouts": 0, "scored_identity_contact_svd": 0,
                "single_visit_records": 0, "unscorable_contact_svd": 0,
                "no_visit_controls": 0, "assignments": Counter(),
                "terminal": Counter(), "seed_resolved": Counter(),
                "unresolved_fruit_links": 0, "missing_seed_counts": 0,
                "outcome_status_counts": Counter(),
            }
        return buckets[pid, block]

    usable: set[str] = set()
    for eid, row in efforts.items():
        pid = linked(row)
        b = bucket(pid, text(row, "sampling_block_id"))
        if text(row, "usable_observation") != "yes":
            continue
        start, end = (datetime.fromisoformat(text(row, k)) for k in ("start_time", "end_time"))
        flowers = float(text(row, "monitored_open_flower_count"))
        if start.tzinfo is None or end.tzinfo is None or end <= start or not math.isfinite(flowers) or flowers <= 0:
            raise ValueError(f"invalid usable exposure: {eid}")
        usable.add(eid)
        b["usable_effort_windows"] += 1
        b["routine_effort_windows"] += text(row, "effort_purpose") == "routine_observation"

    for row in visits:
        pid = linked(row)
        eid = text(row, "effort_id")
        if eid not in usable or pid != text(efforts[eid], "plant_id"):
            raise ValueError("visit lacks same-plant usable effort")
        if text(efforts[eid], "flower_id") and text(row, "flower_id") != text(efforts[eid], "flower_id"):
            raise ValueError("visit/effort flower mismatch")
        bucket(pid, text(efforts[eid], "sampling_block_id"))["visit_bouts"] += 1

    svd_flowers: set[tuple[str, str]] = set()
    svd_visits: set[str] = set()
    for row in svd:
        pid = linked(row, ("population_id", "taxon"))
        fid = text(row, "flower_id")
        key = pid, fid
        if not fid or key in svd_flowers:
            raise ValueError("blank/reused SVD flower; recounts need a separate record")
        svd_flowers.add(key)
        b = bucket(pid, text(row, "sampling_block_id"))
        count(text(row, "conspecific_pollen_grains"), "SVD")
        kind = text(row, "record_type")
        if kind in CONTROL:
            if text(row, "visit_id") or text(row, "visitor_group"):
                raise ValueError("no-visit control names a visitor")
            b["no_visit_controls"] += 1
            continue
        if kind != "single_visit" or text(row, "first_visit_confirmed") != "yes":
            raise ValueError("SVD requires a confirmed first visit or named no-visit control")
        vid = text(row, "visit_id")
        if vid not in bouts or vid in svd_visits:
            raise ValueError("unknown or reused SVD visit")
        svd_visits.add(vid)
        visit = bouts[vid]
        for field in ("plant_id", "flower_id", "effort_id", "visitor_group"):
            if not text(row, field) or text(row, field) != text(visit, field):
                raise ValueError(f"SVD/visit mismatch: {field}")
        if text(row, "sampling_block_id") != text(efforts[text(row, "effort_id")], "sampling_block_id"):
            raise ValueError("SVD/effort block mismatch")
        b["single_visit_records"] += 1
        # Recorded negative contacts and zero deposition are valid; positivity is NOT required.
        scored = all(text(visit, k) in {"confirmed", "not_seen"} for k in ("anther_contact", "stigma_contact"))
        identified = all(text(r, "identification_confidence") in {"confirmed", "group_level"} for r in (row, visit))
        identified = identified and text(row, "visitor_group") != "unknown_visitor"
        b["scored_identity_contact_svd"] += scored and identified
        b["unscorable_contact_svd"] += not scored

    treatment_flowers: set[tuple[str, str]] = set()
    used_fruits: set[str] = set()
    for row in treatments:
        pid = linked(row, ("population_id", "taxon"))
        key = pid, text(row, "flower_id")
        if not key[1] or key in treatment_flowers or key in svd_flowers:
            raise ValueError("reused treatment flower or destructive SVD/treatment flower collision")
        treatment_flowers.add(key)
        b = bucket(pid, text(row, "sampling_block_id"))
        treatment, state = text(row, "treatment_type"), text(row, "outcome_status")
        if treatment not in {*CORE, "hand_self", "emasculated_open"}:
            raise ValueError("unknown reproductive treatment")
        if state not in {*TERMINAL, "pending", "lost", "damaged"}:
            raise ValueError("unknown reproductive outcome state")
        if treatment == "supplemental_outcross" and text(row, "hand_pollen_source_plant_id") in {"", pid}:
            raise ValueError("outcross donor must be recorded and different from mother")
        b["assignments"][treatment] += 1
        b["outcome_status_counts"][state] += 1
        fruit_id = text(row, "fruit_id")
        if state != "mature_fruit" and fruit_id:
            raise ValueError("non-mature treatment names a fruit")
        if state not in TERMINAL:
            continue
        b["terminal"][treatment] += 1
        if state == "aborted":
            b["seed_resolved"][treatment] += 1  # Observed abortion is zero, not missing.
            continue
        if fruit_id and fruit_id in used_fruits:
            raise ValueError("fruit reused across treatment flowers")
        if fruit_id:
            used_fruits.add(fruit_id)
        fruit = fruit_by_id.get(fruit_id)
        if fruit is None:
            b["unresolved_fruit_links"] += 1
            continue
        if text(fruit, "maternal_id") != pid or text(fruit, "site_id") != text(row, "site_id"):
            raise ValueError("fruit maternal/site mismatch")
        value = text(fruit, "mature_seed_count")
        if not value:
            b["missing_seed_counts"] += 1
            continue
        count(value, f"fruit {fruit_id}")
        b["seed_resolved"][treatment] += 1

    for pid in registry:
        if not any(p == pid for p, _ in buckets):
            bucket(pid, "")
    rows, pooled = [], defaultdict(set)
    complete_plants: set[str] = set()
    required = {"routine_effort", "contact_svd", "svd_control", *CORE}
    for (pid, block), b in sorted(buckets.items()):
        flags = set()
        if b["routine_effort_windows"]:
            flags.add("routine_effort")
        if b["scored_identity_contact_svd"]:
            flags.add("contact_svd")
        if b["no_visit_controls"]:
            flags.add("svd_control")
        flags.update(t for t in CORE if b["terminal"][t])
        missing = sorted(required - flags)
        if not block:
            missing.append("sampling_block_id")
        terminal_complete = not missing
        # Do not silently ignore unresolved flowers within a nominally complete panel.
        seed_complete = terminal_complete and all(b["seed_resolved"][t] == b["terminal"][t] for t in CORE)
        plant = registry[pid]
        pooled[text(plant, "population_id"), text(plant, "field_event_id"), block].update(flags)
        if seed_complete:
            complete_plants.add(pid)
        rows.append({"plant_id": pid, "population_id": text(plant, "population_id"),
                     "field_event_id": text(plant, "field_event_id"), "sampling_block_id": block,
                     **b, "missing_channels": missing,
                     "same_plant_terminal_panel_present": terminal_complete,
                     "same_plant_seed_panel_resolved": seed_complete})
    pooling_only = []
    for (pop, event, block), flags in sorted(pooled.items()):
        if block and required <= flags and not any(r["population_id"] == pop and r["field_event_id"] == event and r["sampling_block_id"] == block and r["same_plant_terminal_panel_present"] for r in rows):
            pooling_only.append({"population_id": pop, "field_event_id": event, "sampling_block_id": block})
    return {"schema_version": "1.0", "status": "coverage_audited" if registry else "no_field_rows",
            "registered_plants": len(registry), "plant_blocks": rows,
            "plants_with_seed_resolved_panel": len(complete_plants),
            "population_pooled_only_panels": pooling_only,
            "unlinked_fruit_ids": sorted(set(fruit_by_id) - used_fruits),
            "historical_transition_identification": "not_assessed",
            "statistical_adequacy": "not_assessed", "provenance_and_native_qc": "required_separately",
            "community_functional_traits": "not_assessed",
            "boundary": "Coverage only; declared blocks are not proof of randomization or temporal comparability. No causal mediation, historical turnover, selection, realized selfing or Chapter 2 validation is established. Incomplete and zero-visit plants must remain in inference and missingness reporting."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for channel in CHANNELS:
        parser.add_argument("--" + channel, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs, identities = {}, {}
    for channel in CHANNELS:
        path = getattr(args, channel)
        payload = path.read_bytes()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"{channel} CSV needs a header, even when empty")
            inputs[channel] = list(reader)
        identities[channel] = {"sha256": hashlib.sha256(payload).hexdigest(), "rows": len(inputs[channel])}
    result = audit(**inputs)
    result["input_identities"] = identities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{result['status']}: {result['plants_with_seed_resolved_panel']}/{result['registered_plants']} plants with a seed-resolved panel")


if __name__ == "__main__":
    main()
