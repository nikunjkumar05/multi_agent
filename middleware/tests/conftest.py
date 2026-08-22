"""Test configuration for middleware.

Sets BAMAS_MIDDLEWARE_TEST_MODE=1 BEFORE any test module imports the app,
so routes/tasks.py registers MockAdapters instead of real CLI agents.
"""

import os

os.environ["BAMAS_MIDDLEWARE_TEST_MODE"] = "1"
