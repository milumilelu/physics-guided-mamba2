# Blinded four-edge validation protocol

Purpose: obtain evidence that is independent of v2–v7 objectives. This does not
authorize Phase A or substitute manual values into only the failed samples.

1. Use `manual_four_edge_validation.csv`. Hide process parameters, automatic
   centers, status, contrast quartile and method-version results from the annotator.
2. One human annotator (A, per user decision on 2026-08-31) manually annotates
   all 200 samples by dragging one rectangle in the fixed-angle canonical
   view. The tool records `left_u_um`, `right_u_um`, `top_v_um`, and
   `bottom_v_um`, plus the converted raw-image center. They may change contrast
   or switch height/depth display but may not consult an automatic overlay.
3. The list contains all 200 samples. Rows must not be removed for weak contrast.
4. Compute each annotation center as opposing-edge midpoint. Keep observed
   widths as QA; do not force an annotator's edges to exactly 200 um.
5. Because there is no second annotator, no inter-reader precision claim may be
   made. Compare v6 with the single-reader manual annotations only after all
   200 rows are complete. Report results by session and contrast quartile.
7. If acceptance fails preferentially in shallow samples, Phase A remains
   blocked. A per-sample fallback or selective deletion is forbidden.

## Visual boundary rule

- Mark the midpoint of the transition band between stable unprocessed exterior
  and the first spatially continuous modified interior.
- Follow the longest continuous part of each side. Ignore isolated splashes,
  particles and corner burrs; do not expand the box to contain them.
- A continuous raised/depressed rim belongs to the transition band: place the
  edge through its middle, rather than at its outermost pixel.
- Do not force width or height to 200 um. The displayed angle is already fixed;
  do not estimate or rotate the rectangle.
- If any one side cannot be located without inferring it from the opposite side,
  press `U` (Unusable) and explain which side is invisible in the comment.

Unfrozen fields requiring human decision before annotation:

- v6 center acceptance limits: **TBD**
- Required shallow-quartile performance: **TBD**
