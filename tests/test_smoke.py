"""
Smoke test to ensure basic modules import cleanly.
"""

import wellscope

def test_version():
    assert wellscope.__version__ == "1.0.0"
