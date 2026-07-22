#!/usr/bin/env python3
"""Join authoritative field metadata to the retained floral measurements.

``withlocation.csv`` supplies date, island, site, coordinates, plant identity,
continuous corolla number and sexual phase. Its row order and numbering are preserved;
measurement columns are added by the one-to-one ``collar``/``global_id`` join.

Writes ``results_shimask_all/corolla_master.csv`` and flags disagreements between
field sexual status and the presence of a measured reproductive-organ trace.
"""
from __future__ import annotations

import csv
from pathlib import Path

RESULTS = Path("results_shimask_all")

FINAL_COLS = [
    "fold_state",
    "corolla_length_mm",
    "corolla_width_obs_mm",
    "corolla_area_obs_mm2",
    "corolla_width_fulleq_mm",
    "corolla_area_fulleq_mm2",
    "roi_source",
    "match_iou",
    "has_nectar_guide",
    "guide_coverage_pct",
    "organ_length_mm",
    "qc_flag",
]
POLL_COLS = [
    "throat_width_mm",
    "mouth_width_mm",
    "corolla_aspect_L_W",
    "tube_flare_W_throat",
    "lobe_incision_mm",
    "lobe_incision_ratio",
    "organ_corolla_ratio",
]


def main() -> None:
    meta_path = Path("withlocation.csv")
    if not meta_path.exists():
        meta_path = RESULTS / "field_metadata.csv"
    metadata = list(csv.DictReader(meta_path.open(encoding="utf-8-sig")))
    global_index = {
        int(r["global_id"]): r
        for r in csv.DictReader((RESULTS / "global_index.csv").open(encoding="utf-8-sig"))
    }
    final = {
        (r["sheet"], r["corolla_id"]): r
        for r in csv.DictReader((RESULTS / "corolla_traits_final.csv").open(encoding="utf-8-sig"))
    }
    pollination = {
        (r["sheet"], r["corolla_id"]): r
        for r in csv.DictReader((RESULTS / "pollination_traits.csv").open(encoding="utf-8-sig"))
    }

    meta_cols = list(metadata[0].keys())
    out_cols = meta_cols + ["sheet", "sheet_corolla_id"] + FINAL_COLS + POLL_COLS + [
        "organ_status_check"
    ]
    rows = []
    n_disagree = 0
    for meta in metadata:
        index = global_index[int(meta["collar"])]
        sheet, corolla_id = index["sheet"], index["sheet_corolla_id"]
        final_row = final.get((sheet, corolla_id), {})
        poll_row = pollination.get((sheet, corolla_id), {})
        row = dict(meta)
        row["sheet"] = sheet
        row["sheet_corolla_id"] = corolla_id
        for column in FINAL_COLS:
            row[column] = final_row.get(column, "")
        for column in POLL_COLS:
            row[column] = poll_row.get(column, "")

        measured = bool(final_row.get("organ_length_mm"))
        status_has_organ = meta["status"] in ("s", "p")
        if status_has_organ and not measured:
            row["organ_status_check"] = "status_but_no_measurement"
            n_disagree += 1
        elif not status_has_organ and measured:
            row["organ_status_check"] = "measured_but_status_na"
            n_disagree += 1
        else:
            row["organ_status_check"] = ""
        rows.append(row)

    out = RESULTS / "corolla_master.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}  ({len(rows)} corollas; {n_disagree} organ status/measurement mismatches)")
    for row in rows:
        if row["organ_status_check"]:
            print(
                f"  collar {row['collar']} ({row['sheet']} C{row['sheet_corolla_id']}): "
                f"status={row['status']} organ_length='{row['organ_length_mm']}' "
                f"-> {row['organ_status_check']}"
            )


if __name__ == "__main__":
    main()
