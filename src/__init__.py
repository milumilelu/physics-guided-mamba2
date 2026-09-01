"""Scientific pipeline modules for rectangle registration and stable ROI.

Everything in here obeys the data contract from
``archive/rectangle_registration_history/CODEX_TASK_矩形加工区配准与稳定ROI_v2.md``:

    H_raw -> H_reg -> H_200 -> H_stable

Invalid samples are always ``NaN`` in ``z`` and ``False`` in ``valid_mask``.
Nothing in this package fills, interpolates or silently repairs them.
"""
