Pyvisgrid 0.2.0 (2025-09-25)
============================


API Changes
-----------


Bug Fixes
---------

- Downgraded the pinned version for ``numpy`` in ``pyproject.toml`` to avoid potential conflicts.
- Updated the docstring for the ``vis_data`` attribute in `GridData` to specify its shape for better clarity.
- Improved data selection and masking logic in ``GridData.from_ms`` to ensure correct row selection, flag masking, and shape alignment for measurement and channel data. This also fixes the handling of flagged data and ensures that the returned arrays are correctly filtered and shaped. [`#29 <https://github.com/radionets-project/pyvisgrid/pull/29>`__]


Data Model Changes
------------------


New Features
------------


Maintenance
-----------


Refactoring and Optimization
----------------------------
