# instability montage (Phase 2A-2, two-round blinded)

- `round1/AUDIT-xx_blind.png`: blinded morphology-only pages. Do the FIRST
  review pass using ONLY these pages (fill the `blind_*` columns).
- `round2/sample_<ddd>_unblind.png`: unblind pages with neighbours + metadata
  (`blind_*` conclusions may then be confirmed or revised in the unblind
  columns). `*_groupscale.png` shares one colour scale across the selected
  pool for the residual/DCT-band panels; panel A (absolute height) stays
  own-scale in both variants because sample depths differ by ~60 um.
- `morphology_pattern` uses the checklist terms, `;`-separated:
  edge_contamination;large_area_dropout;repair_driven_feature;large_pit;ridge;periodic_stripe;anisotropic_texture;low_frequency_waviness;localized_collapse;multi_lobe_morphology
No automatic classification exists in this script (细则 §4).
