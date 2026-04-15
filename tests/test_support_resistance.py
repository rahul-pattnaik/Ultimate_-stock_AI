from __future__ import annotations

import pandas as pd

from technical.support_resistance import get_sr_zones, get_support_resistance


def test_get_support_resistance_returns_tuple(sideways_df: pd.DataFrame) -> None:
    support, resistance = get_support_resistance(sideways_df)
    assert support is not None
    assert resistance is not None
    assert support < resistance


def test_get_sr_zones_returns_nearest_levels(sideways_df: pd.DataFrame) -> None:
    result = get_sr_zones(sideways_df)
    assert "support_zones" in result
    assert "resistance_zones" in result
    assert "pivot_points" in result


def test_get_support_resistance_handles_empty_dataframe() -> None:
    support, resistance = get_support_resistance(pd.DataFrame())
    assert support is None
    assert resistance is None
