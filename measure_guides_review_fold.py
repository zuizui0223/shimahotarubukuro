# -*- coding: utf-8 -*-
"""Manual fold-state review overrides for flattened corolla scans.

The scan preparation includes two distinct states:

- ``opened_full``: the five corolla lobes are opened flat;
- ``folded_half``: the corolla is folded longitudinally, so roughly 2.5 of the
  five lobes are visible and planar widths/areas are approximately halved.

For manually reviewed sheets, these labels override heuristic width:length rules.
Unknown sheets keep their automatic classification and remain subject to review.
"""
from __future__ import annotations

import math
from typing import MutableMapping


REVIEWED_FOLD_STATE = {
    ("shikinejima", "shikine1~4"): {
        1: {"state": "opened_full", "visible_lobes": 5.0},
        2: {"state": "folded_half", "visible_lobes": 2.5},
        3: {"state": "folded_half", "visible_lobes": 2.5},
        4: {"state": "opened_full", "visible_lobes": 5.0},
        5: {"state": "opened_full", "visible_lobes": 5.0},
        6: {"state": "opened_full", "visible_lobes": 5.0},
    }
}


def get_reviewed_fold_state(folder: str, sheet: str, corolla_id: int):
    return REVIEWED_FOLD_STATE.get((folder.lower(), sheet.lower()), {}).get(int(corolla_id))


def apply_reviewed_fold_state(
    summary: MutableMapping[str, object],
    *,
    folder: str,
    sheet: str,
    corolla_id: int,
) -> None:
    """Apply a reviewed fold state and recompute fold-sensitive planar proxies.

    The raw flattened measurements are preserved. Additional corrected-estimate
    fields make the fold assumption explicit instead of silently replacing the
    measured values.
    """
    reviewed = get_reviewed_fold_state(folder, sheet, corolla_id)
    if reviewed is None:
        summary["fold_state_reviewed"] = ""
        summary["visible_lobes_reviewed"] = ""
        summary["fold_state_source"] = "auto_unreviewed"
        return

    state = str(reviewed["state"])
    visible_lobes = float(reviewed["visible_lobes"])
    folded = state == "folded_half"
    factor = 2.0 if folded else 1.0

    flat_area = float(summary.get("corolla_area_ruler_mm2", 0.0) or 0.0)
    throat_span = float(summary.get("flat_throat_span_mm", 0.0) or 0.0)
    max_span = float(summary.get("corolla_max_span_ruler_mm", 0.0) or 0.0)
    mid_width = float(summary.get("flat_mid_tube_width_mm", 0.0) or 0.0)
    basal_width = float(summary.get("flat_basal_tube_width_mm", 0.0) or 0.0)

    unfolded_throat_span = throat_span * factor
    unfolded_max_span = max_span * factor
    unfolded_mid_width = mid_width * factor
    unfolded_basal_width = basal_width * factor
    unfolded_area = flat_area * factor

    mouth_diameter = unfolded_throat_span / math.pi
    entrance_area = math.pi * (mouth_diameter / 2.0) ** 2

    summary.update(
        {
            "fold_state_reviewed": state,
            "visible_lobes_reviewed": visible_lobes,
            "fold_state_source": "manual_sheet_review",
            "fold_width_area_correction_factor": factor,
            "corolla_area_unfolded_est_mm2": round(unfolded_area, 3),
            "max_span_unfolded_est_mm": round(unfolded_max_span, 3),
            "throat_span_unfolded_est_mm": round(unfolded_throat_span, 3),
            "mid_tube_width_unfolded_est_mm": round(unfolded_mid_width, 3),
            "basal_tube_width_unfolded_est_mm": round(unfolded_basal_width, 3),
            "prov_mouth_diameter_reviewed_mm": round(mouth_diameter, 3),
            "prov_entrance_area_reviewed_mm2": round(entrance_area, 3),
            "fold_state_auto_original": summary.get("fold_state_auto", ""),
            "fold_state_auto": state,
            "mouth_proxy_assumption": (
                "reviewed_folded_half_2x_flat_span_over_pi"
                if folded
                else "reviewed_open_full_flat_span_over_pi"
            ),
        }
    )

    qc = str(summary.get("visitor_trait_qc", "") or "")
    parts = [part for part in qc.split("|") if part and part != "folded_mouth_proxy"]
    summary["visitor_trait_qc"] = "|".join(parts)
