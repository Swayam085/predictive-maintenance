# tests/test_features.py
# Author      : Vrushabh (Data Engineer)
# Description : Unit tests for features.py

import pytest
import pandas as pd
import numpy as np
import sys

sys.path.append("src")
from data_loader import load_raw, clean
from features import (
    load_clean, add_power_feature, add_temp_diff_feature,
    add_wear_bins, add_interaction_terms,
    add_outlier_flags, add_rpm_per_torque,
    add_wear_rate, add_multi_flag
)


@pytest.fixture
def clean_df():
    df = load_raw()
    return clean(df)


class TestFeatures:

    def test_power_feature(self, clean_df):
        df = add_power_feature(clean_df)
        assert "Power" in df.columns
        expected = clean_df["Torque_Nm"] * clean_df["Rotational_speed_rpm"]
        pd.testing.assert_series_equal(
            df["Power"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_temp_diff_feature(self, clean_df):
        df = add_temp_diff_feature(clean_df)
        assert "Temp_diff" in df.columns
        assert df["Temp_diff"].min() >= 0, "Temp_diff negative nahi hona chahiye!"

    def test_wear_bins(self, clean_df):
        df = add_wear_bins(clean_df)
        assert "Tool_wear_bin" in df.columns
        assert set(df["Tool_wear_bin"].unique()).issubset({0, 1, 2})

    def test_interaction_terms(self, clean_df):
        df = add_power_feature(clean_df)
        df = add_temp_diff_feature(df)
        df = add_interaction_terms(df)
        assert "Torque_x_Wear" in df.columns
        assert "Power_x_Temp"  in df.columns

    def test_outlier_flags(self, clean_df):
        df = add_power_feature(clean_df)
        df = add_outlier_flags(df)
        flag_cols = [c for c in df.columns if c.endswith("_outlier")]
        assert len(flag_cols) == 4, "4 outlier flag columns hone chahiye!"
        for col in flag_cols:
            assert set(df[col].unique()).issubset({0, 1}), \
                f"{col} has values other than 0/1!"

    def test_multi_flag_range(self, clean_df):
        df = add_power_feature(clean_df)
        df = add_outlier_flags(df)
        df = add_multi_flag(df)
        assert df["Multi_flag"].min() >= 0
        assert df["Multi_flag"].max() <= 4

    def test_final_shape(self, clean_df):
        df = add_power_feature(clean_df)
        df = add_temp_diff_feature(df)
        df = add_wear_bins(df)
        df = add_interaction_terms(df)
        df = add_outlier_flags(df)
        df = add_rpm_per_torque(df)
        df = add_wear_rate(df)
        df = add_multi_flag(df)
        assert df.shape == (10000, 19), \
            f"Expected (10000, 19), got {df.shape}"