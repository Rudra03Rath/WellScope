"""
Unit tests for multi-track log visualization.
"""

import pandas as pd
import numpy as np
from wellscope.visualization.log_tracks import create_multi_track_log_plot

def test_create_multi_track_log_plot():
    df = pd.DataFrame({
        "DEPTH": np.arange(2000.0, 2050.0, 1.0),
        "GR": np.linspace(30.0, 110.0, 50),
        "CALI": np.full(50, 8.5),
        "RT": np.linspace(2.0, 80.0, 50),
        "RXO": np.linspace(1.5, 20.0, 50),
        "RHOB": np.linspace(2.6, 2.2, 50),
        "NPHI": np.linspace(0.35, 0.15, 50),
        "DT": np.linspace(90.0, 75.0, 50),
        "PHID": np.linspace(0.05, 0.25, 50),
        "PHIE": np.linspace(0.04, 0.22, 50),
        "SW": np.linspace(0.9, 0.2, 50),
        "CANDIDATE_PAY": np.array([0]*35 + [1]*15),
    })

    fig = create_multi_track_log_plot(df, depth_col="DEPTH")
    assert fig is not None
    # Check that traces were added
    assert len(fig.data) > 5
    # Check that depth axis is reversed
    assert fig.layout.yaxis.autorange == "reversed"
