# -*- coding: utf-8 -*-
"""Apply confirmed rescue rows without altering the original automatic outputs.

Input is a completed copy of manual_rescue_template.csv. A selected automatic
candidate or two manually entered endpoints become a new organ instance in a
separate rescued CSV. The source automatic CSV remains immutable.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import measure_guides as base

MM_PX = float(base.MM_PX)


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_shimask")
    parser.add_argument("--rescue-dir", default="results_v3/rescue")
    parser.add_argument("--completed", required=True)
    parser.add_argument("--output", default="results_v3/organs_v3_rescued.csv")
    args = parser.parse_args()
    results, rescue = Path(args.results), Path(args.rescue_dir)
    organs = read(results / "organs_v3.csv")
    candidates = read(rescue / "rescue_candidates.csv") if (rescue / "rescue_candidates.csv").exists() else []
    completed = read(Path(args.completed))
    next_id = max([int(float(r.get("organ_instance_id", 0) or 0)) for r in organs] or [0]) + 1
    added = []
    for row in completed:
        action = str(row.get("action", "")).strip()
        if action in {"", "not_present", "unmeasurable"}:
            continue
        common = dict(island=row["island"], sheet=row["sheet"], nearest_corolla=int(float(row["corolla_id"])),
                      organ_instance_id=next_id, organ_sample_index=1, organ_sample_count=1,
                      organ_type_auto=row.get("organ_type", "unknown"), organ_identity_status="manual_rescue_confirmed",
                      organ_qc_required=0, organ_confidence=1.0, candidate_source="manual_rescue")
        if action == "select_candidate":
            rank = int(float(row["selected_candidate_rank"]))
            hit = next((c for c in candidates if c["sheet"] == row["sheet"] and int(float(c["corolla_id"])) == int(float(row["corolla_id"])) and int(float(c["candidate_rank"])) == rank), None)
            if hit is None:
                raise ValueError(f"candidate not found: {row['sheet']} C{row['corolla_id']} R{rank}")
            common.update(cx=hit["cx"], cy=hit["cy"], organ_len_mm=hit["organ_len_mm"], organ_width_mm=hit["organ_width_mm"],
                          organ_aspect=hit["aspect"], association_distance_mm=hit["distance_mm"])
        elif action == "enter_endpoints":
            x1,y1,x2,y2 = [float(row[k]) for k in ("x1","y1","x2","y2")]
            width = float(row["width_mm"]) if row.get("width_mm") not in (None, "") else ""
            length = math.hypot(x2-x1, y2-y1) * MM_PX
            common.update(cx=round((x1+x2)/2,2), cy=round((y1+y2)/2,2), organ_len_mm=round(length,2),
                          organ_width_mm=width, organ_aspect=round(length/width,2) if width not in ("",0) else "",
                          association_distance_mm="")
        else:
            raise ValueError(f"unsupported action: {action}")
        common["organ_id"] = f"rescue_{next_id}"
        common["measurement_unit"] = "confirmed_organ_instance"
        added.append(common); next_id += 1
    output = organs + added
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); write(out, output)
    print(f"automatic_rows={len(organs)} rescued_instances={len(added)} output_rows={len(output)}")


if __name__ == "__main__":
    main()
