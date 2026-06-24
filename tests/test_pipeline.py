# tests/test_pipeline.py
# Author      : Vrushabh (Data Engineer)
# Description : Unit tests for pipeline.py + inference.py + alert_system.py

import pytest
import numpy as np
import sys

sys.path.append("src")
from inference import preprocess_input, load_feature_cols, get_alert_level
from alert_system import get_alert_level, get_alert_message, create_alert


class TestPreprocess:

    def test_output_shape(self):
        sample = {
            "Type": 1, "Air_temperature_K": 300.0,
            "Process_temperature_K": 310.5,
            "Rotational_speed_rpm": 1500,
            "Torque_Nm": 45.0, "Tool_wear_min": 150
        }
        feature_cols = load_feature_cols()
        X = preprocess_input(sample, feature_cols)
        assert X.shape == (1, 18), f"Expected (1, 18), got {X.shape}"

    def test_output_dtype(self):
        sample = {
            "Type": 1, "Air_temperature_K": 300.0,
            "Process_temperature_K": 310.5,
            "Rotational_speed_rpm": 1500,
            "Torque_Nm": 45.0, "Tool_wear_min": 150
        }
        feature_cols = load_feature_cols()
        X = preprocess_input(sample, feature_cols)
        assert X.dtype == np.float32, \
            f"Expected float32, got {X.dtype}"

    def test_power_calculated(self):
        sample = {
            "Type": 1, "Air_temperature_K": 300.0,
            "Process_temperature_K": 310.5,
            "Rotational_speed_rpm": 1500,
            "Torque_Nm": 40.0, "Tool_wear_min": 100
        }
        feature_cols = load_feature_cols()
        X = preprocess_input(sample, feature_cols)
        power_idx = feature_cols.index("Power")
        expected_power = 40.0 * 1500
        assert abs(X[0][power_idx] - expected_power) < 1.0


class TestAlertSystem:

    def test_green_alert(self):
        assert get_alert_level(0.10) == "GREEN"
        assert get_alert_level(0.29) == "GREEN"

    def test_yellow_alert(self):
        assert get_alert_level(0.30) == "YELLOW"
        assert get_alert_level(0.50) == "YELLOW"

    def test_red_alert(self):
        assert get_alert_level(0.60) == "RED"
        assert get_alert_level(0.90) == "RED"

    def test_alert_message_contains_prob(self):
        msg = get_alert_message("GREEN", 0.15)
        assert "15.00%" in msg

    def test_create_alert_keys(self):
        sensor = {"Type": 1, "Torque_Nm": 45.0}
        alert  = create_alert(sensor, 0.85, 1)
        required_keys = [
            "timestamp", "alert_level",
            "pred_label", "failure_prob",
            "message", "sensor_input"
        ]
        for key in required_keys:
            assert key in alert, f"Missing key: {key}"