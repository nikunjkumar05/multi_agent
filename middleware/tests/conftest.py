"""Test configuration for middleware.

Sets env BEFORE any test module imports the app:
- BAMAS_MIDDLEWARE_TEST_MODE=1  -> registry builds MockAdapters only
- BAMAS_DB_PATH=<tmp>           -> SQLite persistence isolated per session
- BAMAS_API_KEY removed         -> auth middleware allows all in tests
"""

import os
import tempfile

os.environ["BAMAS_MIDDLEWARE_TEST_MODE"] = "1"
os.environ["BAMAS_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="bamas_test_"), "test_state.db"
)
os.environ.pop("BAMAS_API_KEY", None)
