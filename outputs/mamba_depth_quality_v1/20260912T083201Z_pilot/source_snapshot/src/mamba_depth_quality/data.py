from pathlib import Path
import numpy as np
import pandas as pd

MAIN_ROLES = ("formal", "pass_main")
DEFAULT_AUDIT = Path("outputs/task_state_v1/20260906T111435Z_E00_a1a1c9a8_533009c_9e0686")
DEFAULT_HEIGHT = Path("outputs/rectangle_registration/manual_internal_roi_v1/dataset/stable_roi_80um_dataset.npz")

def load_main180(audit_dir=DEFAULT_AUDIT, height_package=DEFAULT_HEIGHT):
    """Load frozen MAIN180 identities and recompute D/Sq from raw height.

    Height rows are aligned by the package identity columns, never by an
    unchecked CSV row number.  The returned frame contains only process and
    audit columns; arrays are returned separately for callers that need them.
    """
    audit_dir, height_package = Path(audit_dir), Path(height_package)
    targets = pd.read_csv(audit_dir / "frozen_targets.csv")
    split = pd.read_csv(audit_dir / "split_manifest.csv")
    frame = targets.merge(split, on=["dataset_index","session_id","sample_id",
                                    "shared_height_source_id","session_role"],
                          how="inner", validate="one_to_one")
    frame = frame[frame.session_role.isin(MAIN_ROLES)].copy()
    if len(frame) != 180:
        raise ValueError(f"MAIN180 coverage mismatch: {len(frame)}")
    z = np.load(height_package, allow_pickle=False)
    required = {"height_raw","valid_mask","session_id","measurement_id","sample_id"}
    if not required.issubset(z.files):
        raise ValueError("height package missing identity/raw fields")
    key = {(str(s), int(m), int(i)): j for j,(s,m,i) in enumerate(
        zip(z["session_id"], z["measurement_id"], z["sample_id"]))}
    raw, valid, ds = [], [], []
    for _, row in frame.sort_values("dataset_index").iterrows():
        k = (str(row.session_id), int(row.get("measurement_id", row.sample_id)), int(row.sample_id))
        j = key.get(k)
        if j is None:
            # Most frozen rows use sample_id as measurement_id in this package.
            j = key.get((str(row.session_id), int(row.sample_id), int(row.sample_id)))
        if j is None: raise ValueError(f"height identity not found: {k}")
        h, m = np.asarray(z["height_raw"][j], np.float64), np.asarray(z["valid_mask"][j], bool)
        if not m.all(): raise ValueError("invalid pixel mask is not supported in MAIN180")
        vals = h[m]; raw.append(h); valid.append(m); ds.append((-float(np.median(vals)),
            float(np.sqrt(np.mean((vals-vals.mean())**2)))))
    frame = frame.sort_values("dataset_index").reset_index(drop=True)
    frame["D_computed"] = [x[0] for x in ds]; frame["Sq"] = [x[1] for x in ds]
    if np.max(np.abs(frame.D_computed.to_numpy()-frame.D.to_numpy())) > 1e-8:
        raise ValueError("canonical D parity failure")
    return frame, np.stack(raw), np.stack(valid)

def component_weights(frame):
    n = frame.groupby("component_id")["dataset_index"].transform("count").to_numpy()
    c = frame.component_id.nunique()
    return 1.0 / (c*n)
