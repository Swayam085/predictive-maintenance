# tests/test_dashboard.py
# Author      : Vrushabh (Data Engineer)
# Description : Dashboard + data_feed unit tests

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append("src")
sys.path.append("dashboard")

from data_feed import (
    load_sensor_data,
    get_single_reading,
    simulate_stream,
    get_stream_summary
)
from inference import preprocess_input, load_feature_cols
from alert_system import get_alert_level, create_alert


class TestDataFeed:
    """data_feed.py tests."""

    def test_load_sensor_data_shape(self):
        """10000 rows hone chahiye."""
        df = load_sensor_data()
        assert len(df) == 10000

    def test_load_sensor_data_columns(self):
        """Required columns present hone chahiye."""
        df = load_sensor_data()
        required = [
            "Type", "Air_temperature_K",
            "Process_temperature_K",
            "Rotational_speed_rpm",
            "Torque_Nm", "Tool_wear_min",
            "Machine_failure"
        ]
        for col in required:
            assert col in df.columns, f"Missing: {col}"

    def test_get_single_reading_keys(self):
        """Reading dict mein required keys hone chahiye."""
        df  = load_sensor_data()
        row = get_single_reading(df, idx=0)
        required_keys = [
            "idx", "Type", "Air_temperature_K",
            "Process_temperature_K",
            "Rotational_speed_rpm",
            "Torque_Nm", "Tool_wear_min",
            "actual_failure"
        ]
        for key in required_keys:
            assert key in row, f"Missing key: {key}"

    def test_get_single_reading_idx(self):
        """idx=0 pe first row aani chahiye."""
        df  = load_sensor_data()
        row = get_single_reading(df, idx=0)
        assert row["idx"] == 0

    def test_get_single_reading_random(self):
        """Random reading valid range mein honi chahiye."""
        df  = load_sensor_data()
        row = get_single_reading(df)
        assert 0 <= row["idx"] < len(df)

    def test_simulate_stream_count(self):
        """n_readings ke barabar results aane chahiye."""
        df      = load_sensor_data()
        results = simulate_stream(df, n_readings=5, delay=0)
        assert len(results) == 5

    def test_simulate_stream_keys(self):
        """Result dict mein required keys hone chahiye."""
        df      = load_sensor_data()
        results = simulate_stream(df, n_readings=3, delay=0)
        required = [
            "reading_num", "idx", "sensor_input",
            "failure_prob", "alert_level",
            "message", "actual_failure", "timestamp"
        ]
        for key in required:
            assert key in results[0], f"Missing key: {key}"

    def test_simulate_stream_alert_levels(self):
        """Alert levels sirf GREEN/YELLOW/RED hone chahiye."""
        df      = load_sensor_data()
        results = simulate_stream(df, n_readings=10, delay=0)
        valid   = {"GREEN", "YELLOW", "RED"}
        for r in results:
            assert r["alert_level"] in valid

    def test_simulate_stream_prob_range(self):
        """Failure prob 0-1 range mein honi chahiye."""
        df      = load_sensor_data()
        results = simulate_stream(df, n_readings=5, delay=0)
        for r in results:
            assert 0.0 <= r["failure_prob"] <= 1.0

    def test_get_stream_summary_keys(self):
        """Summary dict mein required keys hone chahiye."""
        df      = load_sensor_data()
        results = simulate_stream(df, n_readings=5, delay=0)
        summary = get_stream_summary(results)
        required = [
            "total", "GREEN", "YELLOW",
            "RED", "actual_failures",
            "correct", "accuracy"
        ]
        for key in required:
            assert key in summary, f"Missing key: {key}"

    def test_get_stream_summary_total(self):
        """Total count sahi hona chahiye."""
        df      = load_sensor_data()
        results = simulate_stream(df, n_readings=5, delay=0)
        summary = get_stream_summary(results)
        assert summary["total"] == 5

    def test_get_stream_summary_counts(self):
        """GREEN + YELLOW + RED = total hona chahiye."""
        df      = load_sensor_data()
        results = simulate_stream(df, n_readings=10, delay=0)
        summary = get_stream_summary(results)
        assert (summary["GREEN"] +
                summary["YELLOW"] +
                summary["RED"]) == summary["total"]


class TestDashboardFiles:
    """Dashboard required files check."""

    def test_app_exists(self):
        """app.py exist karni chahiye."""
        assert os.path.exists("dashboard/app.py")

    def test_data_feed_exists(self):
        """data_feed.py exist karni chahiye."""
        assert os.path.exists("dashboard/data_feed.py")

    def test_featured_data_exists(self):
        """featured_data.csv exist karni chahiye."""
        assert os.path.exists("data/processed/featured_data.csv")

    def test_figures_folder_exists(self):
        """reports/figures/ exist karni chahiye."""
        assert os.path.exists("reports/figures")

    def test_pr_curve_exists(self):
        """pr_curve.png exist karni chahiye."""
        assert os.path.exists("reports/figures/pr_curve.png")