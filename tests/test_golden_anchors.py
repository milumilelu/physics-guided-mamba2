"""Golden scientific regression anchors (static, read-only).

Asserts that the frozen Phase 2.7r1 gate JSONs still carry the exact
registered values.  These anchors are the "science kept intact" tripwire for
the WP1 src/ migration: any accidental semantic change to the migrated
canonical implementations shows up here long before a rerun would.
Static layer only -- the regeneration layer lives in
scripts/40_refactor_golden_regression.py (scratch rerun, frozen tree
read-only per v2.1 F8).
"""

import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "outputs" / "phase2_7" / "summary"

# frozen anchor values (任务说明 v2.1 §4.3; registered 2026-09-04)
G27_1 = {
    ("A2_8_16", "src_gkf", "R2_full_median"): 0.5050539999437547,
    ("A2_8_16", "src_gkf", "median_delta_R2_h"): 0.6509672473528761,
    ("angular_entropy_8_16", "src_gkf", "median_delta_R2_h"): 0.6449781075749867,
    ("ilr_z1_z4", "src_gkf", "R2_full_median"): 0.31305506446560427,
    ("ilr_z1_z4", "src_gkf", "R2_minus_h_median"): 0.14779357389704195,
    ("ilr_z1_z4", "src_gkf", "median_delta_R2_h"): 0.18101258577517343,
    ("p_8_16", "src_gkf", "median_delta_R2_h"): 0.34961895113131647,
    ("A2_8_16", "proc_gkf", "median_delta_R2_h"): 0.7336429885484586,
    ("angular_entropy_8_16", "proc_gkf", "median_delta_R2_h"): 0.7385210797283288,
}


def _load(name: str) -> dict:
    with open(SUMMARY / name, encoding="utf-8") as fh:
        return json.load(fh)


class GoldenAnchors(unittest.TestCase):
    def test_g27_1_hatch_unique_contribution(self):
        doc = _load("gsl27_1_evaluation.json")
        self.assertEqual(doc["G_SL27_1"], "SUPPORTED")
        got = {(row["target"], row["variant"], key): row[key]
               for row in doc["delta_table"] for key in
               ("R2_full_median", "R2_minus_h_median", "median_delta_R2_h")}
        for key, value in G27_1.items():
            self.assertIn(key, got, f"missing anchor row {key}")
            self.assertEqual(got[key], value, f"anchor drift at {key}")

    def test_g27_2_m_decomposition(self):
        doc = _load("gsl27_2_evaluation.json")
        self.assertEqual(doc["G_SL27_2"], "DOMINANT_m=1")
        self.assertEqual(doc["C_family_all"], 0.9038461538461539)
        self.assertEqual(doc["tv_w"], 0.29721160530020074)
        self.assertEqual(doc["p_perm"], 9.999000099990002e-05)
        hd = doc["H_DEPENDENT"]
        self.assertEqual(hd["slope"], -0.33234665899340665)
        self.assertEqual(hd["p_value"], 0.4102948525737131)
        self.assertEqual(hd["flag"], "NO")

    def test_g27_3_envelope_array_model(self):
        # 2.7r2 values (weighted-LOHO recalibration; external review found
        # the r1 main statistic was a macro mean while the bootstrap was
        # weighted).  r1 values, superseded: tv_w_constant 0.4245949074074074,
        # tv_w_period2_loho 0.35469112596305574, delta_tv 0.06990378144435166.
        doc = _load("gsl27_3_evaluation.json")
        self.assertEqual(doc["revision"], "2.7r2")
        self.assertEqual(doc["G_SL27_3"], "MODEL_INADEQUATE")
        self.assertEqual(doc["tv_w_constant"], 0.6151350308641975)
        self.assertEqual(doc["tv_w_period2_loho"], 0.5290432098765432)
        self.assertEqual(doc["delta_tv"], 0.08609182098765433)
        self.assertEqual(doc["bootstrap_ci_low"], 0.018792438271604928)
        self.assertEqual(doc["p_boot"], 0.004997501249375313)


if __name__ == "__main__":
    unittest.main()
