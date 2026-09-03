# phase2_manifest

Built by `experiments/phase2/_lib.build_manifest` from frozen inputs; see
Phase2_执行细则.md §2 for the column contract. `phase1_global_loco_*` columns
are backfilled by `01_instability_inventory.py`.

Power note: `measured_power_W` = 5.3333 W (post-objective, v2 §11) is
**provisional** — no independent measurement record is registered yet, so
`pulse_energy_proxy_uJ` / `areal_dose_proxy_J_per_mm2` carry the `_proxy`
suffix and must not be used in conclusive language until registration.
