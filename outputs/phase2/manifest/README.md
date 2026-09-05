# phase2_manifest

Built by `experiments/phase2/_lib.build_manifest` from frozen inputs; see
Phase2_执行细则.md §2 for the column contract. `phase1_global_loco_*` columns
are backfilled by `01_instability_inventory.py`.

Power note (updated 2026-09-04 per external review §1): `measured_power_W` =
5.3333 W is a **post-objective independently measured average power**, confirmed
physically trusted, and is now the **canonical physical input**. Canonical
derived columns (`pulse_energy_uJ`, `areal_dose_J_per_mm2`) are registered in
`src/provenance.py` (`POWER_REGISTRY`); instrument/date metadata is unavailable
and registered as such — incomplete metadata does not downgrade the measurement.

This frozen CSV keeps its legacy `pulse_energy_proxy_uJ` /
`areal_dose_proxy_J_per_mm2` columns unchanged for Phase 2–2.7 reproduction
(the two definitions are numerically identical); `power_measurement_version =
PENDING_REGISTRATION` records the pre-registration historical state. Phase 2.8+
uses canonical columns only.
