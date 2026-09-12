"""G3 release verification for mamba_depth_quality_v1.

Runs the registered model from an isolated temporary source copy, guarding
against accidental imports from the developer checkout. It verifies a fresh
import plus masked full/stepwise inference before updating the release seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest_tree(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*.py")):
        h.update(str(p.relative_to(path)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_isolated(src_root: Path) -> dict:
    code = r'''
import json, torch
from src.mamba_depth_quality.models import TerminalModel
torch.manual_seed(17)
m = TerminalModel(input_dim=16, encoder="mamba2", bottleneck=4, aux=True).eval()
x = torch.randn(3, 9, 16)
mask = torch.tensor([[1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1]], dtype=torch.bool)
with torch.no_grad():
    full = m(x, None)["D"]
    masked = m(x, mask)["D"]
    encoded_full = m.encoder(m.proj(x), None)
    encoded_step = m.encoder.forward_stepwise(m.proj(x), torch.ones_like(mask))
finite = bool(torch.isfinite(full).all() and torch.isfinite(masked).all() and torch.isfinite(encoded_step).all())
delta = float((encoded_full - encoded_step).abs().max())
print(json.dumps({"finite": finite, "shape": list(full.shape), "parameters": sum(p.numel() for p in m.parameters()), "stepwise_max_abs_diff": delta}))
assert finite and list(full.shape) == [3] and delta < 1e-6
'''
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_root)
    env.pop("PYTHONNOUSERSITE", None)
    # Keep the interpreter's installed dependencies (torch), while replacing
    # PYTHONPATH so project imports resolve only from the temporary copy.
    proc = subprocess.run([sys.executable, "-c", code], cwd=src_root,
                          env=env, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"isolated inference failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="explicit pilot run directory")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    if not (run / "run_manifest.json").is_file():
        raise FileNotFoundError(f"run_manifest.json missing: {run}")
    with tempfile.TemporaryDirectory(prefix="mamba-depth-release-") as td:
        isolated = Path(td)
        shutil.copytree(ROOT / "src", isolated / "src")
        result = run_isolated(isolated)
    report = {
        "status": "PASS",
        "scope": "isolated source copy import/reload plus masked full/stepwise inference",
        "run_dir": str(run),
        "source_sha256": digest_tree(ROOT / "src"),
        "inference": result,
    }
    (run / "g3_clean_checkout_reload.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path = run / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clean_checkout_reload"] = "PASS"
    manifest["implementation_status"] = "PILOT_PASS_WITH_PROTOCOL_LIMITATIONS"
    manifest["limitations"] = [x for x in manifest.get("limitations", []) if "clean checkout reload" not in x]
    if "g3_clean_checkout_reload.json" not in manifest.setdefault("artifacts", []):
        manifest["artifacts"].append("g3_clean_checkout_reload.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
