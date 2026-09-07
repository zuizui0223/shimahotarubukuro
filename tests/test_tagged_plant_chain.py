"""Synthetic software fixtures, not field observations or biological results."""
import copy
import csv
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_tagged_plant_chain.py"
audit = runpy.run_path(str(SCRIPT))["audit"]


def fixture():
    common = dict(population_id="P", field_event_id="E", island_id="oshima", site_id="S", taxon="synthetic_taxon")
    plant = dict(common, plant_id="p1")
    b = dict(plant, sampling_block_id="B")
    effort = [dict(b, effort_id="R", usable_observation="yes", effort_purpose="routine_observation", start_time="2026-07-01T10:00:00+09:00", end_time="2026-07-01T11:00:00+09:00", monitored_open_flower_count=1),
              dict(b, effort_id="A", usable_observation="yes", effort_purpose="svd_assay", start_time="2026-07-01T11:00:00+09:00", end_time="2026-07-01T12:00:00+09:00", monitored_open_flower_count=1)]
    visits = [dict(b, visit_id="V", effort_id="A", flower_id="sv", visitor_group="small_bee", identification_confidence="group_level", anther_contact="not_seen", stigma_contact="not_seen")]
    svd = [dict(b, svd_id="D", flower_id="sv", record_type="single_visit", effort_id="A", visit_id="V", visitor_group="small_bee", first_visit_confirmed="yes", identification_confidence="group_level", conspecific_pollen_grains=0),
           dict(b, svd_id="C", flower_id="ctrl", record_type="exposed_no_visit_control", conspecific_pollen_grains=0)]
    treatments = [dict(b, treatment_id=f"T{i}", flower_id=f"F{i}", treatment_type=t, outcome_status="mature_fruit", fruit_id=f"FR{i}", hand_pollen_source_plant_id="donor") for i, t in enumerate(("open_pollinated", "bagged_autonomous", "supplemental_outcross"))]
    fruits = [dict(fruit_id=f"FR{i}", site_id="S", maternal_id="p1", mature_seed_count=0) for i in range(3)]
    return dict(plants=[plant], effort=effort, visits=visits, svd=svd, treatments=treatments, fruits=fruits)


class TaggedPlantChainTests(unittest.TestCase):
    def test_zero_deposition_and_zero_seed_are_valid(self):
        result = audit(**fixture())
        self.assertEqual(result["plants_with_seed_resolved_panel"], 1)
        self.assertEqual(result["historical_transition_identification"], "not_assessed")
        self.assertEqual(result["statistical_adequacy"], "not_assessed")
        self.assertEqual(result["plant_blocks"][0]["routine_effort_windows"], 1)

    def test_disjoint_plants_fail_despite_population_coverage(self):
        data = fixture()
        data["plants"].append(dict(data["plants"][0], plant_id="p2"))
        for row in data["treatments"]:
            row["plant_id"] = "p2"
        for row in data["fruits"]:
            row["maternal_id"] = "p2"
        result = audit(**data)
        self.assertEqual(result["plants_with_seed_resolved_panel"], 0)
        self.assertEqual(len(result["population_pooled_only_panels"]), 1)

    def test_distinct_sampling_blocks_do_not_join(self):
        data = fixture()
        for row in data["treatments"]:
            row["sampling_block_id"] = "other"
        self.assertEqual(audit(**data)["plants_with_seed_resolved_panel"], 0)

    def test_missing_sampling_block_does_not_pass(self):
        data = fixture()
        for channel in ("effort", "svd", "treatments"):
            for row in data[channel]:
                row.pop("sampling_block_id")
        result = audit(**data)
        self.assertEqual(result["plants_with_seed_resolved_panel"], 0)
        self.assertIn("sampling_block_id", result["plant_blocks"][0]["missing_channels"])

    def test_removed_stigma_flower_cannot_be_reproductive_flower(self):
        data = fixture()
        data["treatments"][0]["flower_id"] = "sv"
        with self.assertRaisesRegex(ValueError, "destructive"):
            audit(**data)

    def test_cross_plant_fruit_is_rejected(self):
        data = fixture()
        data["fruits"][0]["maternal_id"] = "p2"
        with self.assertRaisesRegex(ValueError, "maternal"):
            audit(**data)

    def test_missing_controls_keep_panel_incomplete(self):
        data = fixture()
        data["svd"] = data["svd"][:1]
        self.assertIn("svd_control", audit(**data)["plant_blocks"][0]["missing_channels"])

    def test_missing_seed_is_not_zero(self):
        data = fixture()
        data["fruits"][0]["mature_seed_count"] = ""
        result = audit(**data)
        self.assertTrue(result["plant_blocks"][0]["same_plant_terminal_panel_present"])
        self.assertFalse(result["plant_blocks"][0]["same_plant_seed_panel_resolved"])
        self.assertEqual(result["plant_blocks"][0]["missing_seed_counts"], 1)

    def test_missing_fruit_link_remains_visible(self):
        data = fixture()
        data["fruits"] = []
        result = audit(**data)
        self.assertEqual(result["plants_with_seed_resolved_panel"], 0)
        self.assertEqual(result["plant_blocks"][0]["unresolved_fruit_links"], 3)

    def test_all_aborted_panel_is_not_a_missing_or_positive_only_sample(self):
        data = fixture()
        data["fruits"] = []
        for row in data["treatments"]:
            row.update(outcome_status="aborted", fruit_id="")
        result = audit(**data)
        self.assertEqual(result["plants_with_seed_resolved_panel"], 1)
        self.assertEqual(result["plant_blocks"][0]["outcome_status_counts"]["aborted"], 3)

    def test_lost_damaged_pending_not_terminal_failures(self):
        for state in ("lost", "damaged", "pending"):
            with self.subTest(state=state):
                data = fixture()
                data["treatments"][0].update(outcome_status=state, fruit_id="")
                result = audit(**data)
                self.assertEqual(result["plants_with_seed_resolved_panel"], 0)
                self.assertEqual(result["plant_blocks"][0]["outcome_status_counts"][state], 1)

    def test_repeat_flowers_do_not_add_independent_plants(self):
        data = fixture()
        data["treatments"].append(dict(data["treatments"][0], treatment_id="extra", flower_id="extra", outcome_status="aborted", fruit_id=""))
        result = audit(**data)
        self.assertEqual(result["registered_plants"], 1)
        self.assertEqual(result["plants_with_seed_resolved_panel"], 1)

    def test_zero_visit_plant_is_retained(self):
        data = fixture()
        data["plants"].append(dict(data["plants"][0], plant_id="zero"))
        data["effort"].append(dict(data["effort"][0], plant_id="zero", effort_id="Z"))
        result = audit(**data)
        self.assertEqual(result["registered_plants"], 2)
        row = next(r for r in result["plant_blocks"] if r["plant_id"] == "zero")
        self.assertEqual(row["routine_effort_windows"], 1)
        self.assertEqual(row["visit_bouts"], 0)
        self.assertFalse(row["same_plant_seed_panel_resolved"])

    def test_unscorable_contact_is_not_negative_contact(self):
        data = fixture()
        data["visits"][0]["stigma_contact"] = "not_confirmable"
        result = audit(**data)
        self.assertEqual(result["plant_blocks"][0]["unscorable_contact_svd"], 1)
        self.assertEqual(result["plants_with_seed_resolved_panel"], 0)

    def test_assay_effort_is_not_routine_effort(self):
        data = fixture()
        data["effort"][0]["effort_purpose"] = "svd_assay"
        result = audit(**data)
        self.assertIn("routine_effort", result["plant_blocks"][0]["missing_channels"])

    def test_reused_fruit_rejected(self):
        data = fixture()
        data["treatments"][1]["fruit_id"] = data["treatments"][0]["fruit_id"]
        with self.assertRaisesRegex(ValueError, "fruit reused"):
            audit(**data)

    def test_unknown_visitor_not_converted_into_identity(self):
        data = fixture()
        data["svd"][0]["visitor_group"] = "unknown_visitor"
        data["visits"][0]["visitor_group"] = "unknown_visitor"
        self.assertEqual(audit(**data)["plants_with_seed_resolved_panel"], 0)

    def test_negative_seed_rejected(self):
        data = fixture()
        data["fruits"][0]["mature_seed_count"] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            audit(**data)

    def test_svd_visit_identity_and_block_must_match(self):
        for key, value in (("flower_id", "other"), ("sampling_block_id", "other"), ("visit_id", "missing")):
            with self.subTest(key=key):
                data = fixture()
                data["svd"][0][key] = value
                with self.assertRaises(ValueError):
                    audit(**data)

    def test_no_field_rows_not_called_complete(self):
        result = audit(**{k: [] for k in fixture()})
        self.assertEqual(result["status"], "no_field_rows")
        self.assertEqual(result["plants_with_seed_resolved_panel"], 0)

    def test_cli_writes_input_hashes_without_claim_promotion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            command = [sys.executable, str(SCRIPT)]
            for name, rows in fixture().items():
                path = root / (name + ".csv")
                fields = list(dict.fromkeys(k for row in rows for k in row))
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                command += ["--" + name, str(path)]
            output = root / "audit.json"
            subprocess.run(command + ["--output", str(output)], check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
            self.assertEqual(len(result["input_identities"]), 6)
            self.assertEqual(result["historical_transition_identification"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
