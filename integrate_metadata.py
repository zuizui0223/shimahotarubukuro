#!/usr/bin/env python3
"""Integrate the field metadata table with the measured floral traits.

The reviewer's field table (results_shimask_all/field_metadata.csv) carries, per
corolla, the authoritative provenance and reproductive state:
  date, island, no (site), lat, lon, id (individual plant), collar (corolla number),
  organ (organ number), status (s = staminate/male phase, p = pistillate/female
  phase, na = no organ).
``collar`` is the continuous 1-218 corolla number that matches the numbered-index
sheets, so it joins one-to-one to our global_id. The table's row order and its
number assignments are authoritative and are preserved exactly - we only ADD the
measured trait columns, never reorder or renumber.

Writes results_shimask_all/corolla_master.csv: every metadata row, in its original
order, with the measured corolla/guide/organ/pollination traits joined on collar,
plus an ``organ_status_check`` flag where the field status and our measurement
disagree (status says an organ is present but none was measured, or vice versa).
"""
from __future__ import annotations

import csv
from pathlib import Path

RESULTS = Path("results_shimask_all")

# Measured-trait columns to carry across, and which file supplies them.
FINAL_COLS = ["fold_state", "corolla_length_mm", "corolla_width_obs_mm",
              "corolla_area_obs_mm2", "corolla_width_fulleq_mm", "corolla_area_fulleq_mm2",
              "roi_source", "match_iou", "has_nectar_guide", "guide_coverage_pct",
              "n_guide_spots", "guide_density_per_cm2", "organ_length_mm", "qc_flag"]
POLL_COLS = ["throat_width_mm", "mouth_width_mm", "corolla_aspect_L_W",
             "tube_flare_W_throat", "lobe_incision_mm", "lobe_incision_ratio",
             "style_length_mm", "style_corolla_ratio", "guide_reach_frac",
             "guide_centroid_frac", "guide_contrast_dE", "guide_saturation"]


def main() -> None:
    # Field table lives at the repo root as withlocation.csv (the reviewer's upload);
    # fall back to the in-results copy for older layouts.
    meta_path = Path("withlocation.csv")
    if not meta_path.exists():
        meta_path = RESULTS / "field_metadata.csv"
    meta = list(csv.DictReader(meta_path.open(encoding="utf-8-sig")))
    gindex = {int(r["global_id"]): r
              for r in csv.DictReader((RESULTS / "global_index.csv").open(encoding="utf-8-sig"))}
    final = {(r["sheet"], r["corolla_id"]): r
             for r in csv.DictReader((RESULTS / "corolla_traits_final.csv").open(encoding="utf-8-sig"))}
    poll = {(r["sheet"], r["corolla_id"]): r
            for r in csv.DictReader((RESULTS / "pollination_traits.csv").open(encoding="utf-8-sig"))}

    meta_cols = list(meta[0].keys())
    out_cols = meta_cols + ["sheet", "sheet_corolla_id"] + FINAL_COLS + POLL_COLS + ["organ_status_check"]
    rows = []
    n_disagree = 0
    for m in meta:
        collar = int(m["collar"])
        gi = gindex[collar]
        sheet, scid = gi["sheet"], gi["sheet_corolla_id"]
        fr = final.get((sheet, scid), {})
        pr = poll.get((sheet, scid), {})
        row = dict(m)
        row["sheet"] = sheet
        row["sheet_corolla_id"] = scid
        for c in FINAL_COLS:
            row[c] = fr.get(c, "")
        for c in POLL_COLS:
            row[c] = pr.get(c, "")
        # Cross-check field sexual state against the measured organ.
        measured = bool(fr.get("organ_length_mm"))
        has_status = m["status"] in ("s", "p")
        if has_status and not measured:
            row["organ_status_check"] = "status_but_no_measurement"
            n_disagree += 1
        elif not has_status and measured:
            row["organ_status_check"] = "measured_but_status_na"
            n_disagree += 1
        else:
            row["organ_status_check"] = ""
        rows.append(row)

    out = RESULTS / "corolla_master.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} corollas; {n_disagree} organ status/measurement mismatches)")
    for r in rows:
        if r["organ_status_check"]:
            print(f"  collar {r['collar']} ({r['sheet']} C{r['sheet_corolla_id']}): "
                  f"status={r['status']} organ_length='{r['organ_length_mm']}' -> {r['organ_status_check']}")


if __name__ == "__main__":
    main()
