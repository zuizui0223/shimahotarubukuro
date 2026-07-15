from __future__ import annotations

import unittest

from evaluate_v3_against_shimask_v2 import organ_instances


class OrganInstanceEvaluationTests(unittest.TestCase):
    def test_multiple_axis_samples_count_as_one_organ(self) -> None:
        rows = [
            {"organ_instance_id": "7", "cx": "10", "cy": "20"},
            {"organ_instance_id": "7", "cx": "15", "cy": "20"},
            {"organ_instance_id": "8", "cx": "50", "cy": "60"},
        ]

        instances = organ_instances(rows)

        self.assertEqual(len(instances), 2)
        self.assertIn([(10.0, 20.0), (15.0, 20.0)], instances)
        self.assertIn([(50.0, 60.0)], instances)

    def test_missing_instance_id_falls_back_to_organ_id(self) -> None:
        rows = [
            {"organ_id": "R1", "cx": "1", "cy": "2"},
            {"organ_id": "R1", "cx": "3", "cy": "4"},
        ]

        self.assertEqual(organ_instances(rows), [[(1.0, 2.0), (3.0, 4.0)]])

    def test_rows_without_coordinates_are_ignored(self) -> None:
        rows = [
            {"organ_instance_id": "1", "cx": "", "cy": "2"},
            {"organ_instance_id": "2", "cx": "3", "cy": "4"},
        ]

        self.assertEqual(organ_instances(rows), [[(3.0, 4.0)]])


if __name__ == "__main__":
    unittest.main()
