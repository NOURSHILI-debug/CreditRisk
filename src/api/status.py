"""API prediction status constants.

The API currently returns the legacy ``"success"`` value. The canonical
dashboard status mapping is defined in the follow-up contract step; this
constant exists so the value has a single source in the meantime.
"""

from __future__ import annotations

SUCCESS_STATUS = "success"
