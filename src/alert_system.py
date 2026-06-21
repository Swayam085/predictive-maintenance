# src/alert_system.py
# Author      : Vrushabh (Data Engineer)
# Branch      : feature/vrushabh-data
# Description : 3-tier alert system with logging
#               GREEN / YELLOW / RED alerts based on failure probability
# Week 3 Status:
#   Day 16 — alert_system, logger, alert history
# Used by     : inference.py + Dashboard (Week 4)

import os
import json
import logging
from datetime import datetime

# ── Paths ──────────────────────────────────────────────
LOGS_PATH    = os.path.join("reports", "alert_logs.json")
LOG_FILE     = os.path.join("reports", "inference.log")

# ── Alert Thresholds ───────────────────────────────────
# Calibrated from PR curve (Day 12) — optimal threshold = 0.2288
GREEN_THRESHOLD  = 0.30   # Below 0.30 = safe
YELLOW_THRESHOLD = 0.60   # 0.30-0.60 = warning
# Above 0.60 = critical (RED)


def setup_logger(log_file: str = LOG_FILE) -> logging.Logger:
  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("alert_system")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def get_alert_level(prob: float) -> str:

    if prob < GREEN_THRESHOLD:
        return "GREEN"
    elif prob < YELLOW_THRESHOLD:
        return "YELLOW"
    else:
        return "RED"


def get_alert_message(alert: str, prob: float) -> str:

    messages = {
        "GREEN" : f"[GREEN]  Machine operating normally. Failure prob: {prob:.2%}",
        "YELLOW": f"[YELLOW] Warning! Monitor machine closely. Failure prob: {prob:.2%}",
        "RED"   : f"[RED]    CRITICAL! Immediate maintenance required. Failure prob: {prob:.2%}"
    }
    return messages.get(alert, "Unknown alert level")


def create_alert(sensor_input: dict,
                 prob: float,
                 pred_label: int) -> dict:

    alert_level   = get_alert_level(prob)
    alert_message = get_alert_message(alert_level, prob)

    alert = {
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alert_level" : alert_level,
        "pred_label"  : pred_label,
        "failure_prob": round(prob, 4),
        "message"     : alert_message,
        "sensor_input": sensor_input
    }
    return alert


def log_alert(alert: dict,
              logger: logging.Logger = None) -> None:

    if logger is None:
        logger = setup_logger()

    level = alert["alert_level"]
    msg   = alert["message"]

    if level == "GREEN":
        logger.info(msg)
    elif level == "YELLOW":
        logger.warning(msg)
    elif level == "RED":
        logger.critical(msg)


def save_alert_history(alert: dict,
                       path: str = LOGS_PATH) -> None:

    os.makedirs(os.path.dirname(path), exist_ok=True)

    history = []
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    history.append(alert)


    with open(path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"[ALERT] History saved → {path} ({len(history)} alerts total)")


def process_alert(sensor_input: dict,
                  prob: float,
                  pred_label: int,
                  save_history: bool = True) -> dict:
  
    logger = setup_logger()
    alert  = create_alert(sensor_input, prob, pred_label)

    log_alert(alert, logger)

    if save_history:
        save_alert_history(alert)

    return alert


def get_alert_summary(path: str = LOGS_PATH) -> dict:

    if not os.path.exists(path):
        print("[SUMMARY] No alert history found!")
        return {}

    with open(path, "r") as f:
        history = json.load(f)

    total  = len(history)
    green  = sum(1 for a in history if a["alert_level"] == "GREEN")
    yellow = sum(1 for a in history if a["alert_level"] == "YELLOW")
    red    = sum(1 for a in history if a["alert_level"] == "RED")

    summary = {
        "total" : total,
        "GREEN" : green,
        "YELLOW": yellow,
        "RED"   : red
    }

    print(f"\n[SUMMARY] Alert History:")
    print(f"  Total  : {total}")
    print(f"  GREEN  : {green}")
    print(f"  YELLOW : {yellow}")
    print(f"  RED    : {red}")

    return summary


# ── Direct run ─────────────────────────────────────────
if __name__ == "__main__":

    print("="*50)
    print("ALERT SYSTEM TEST")
    print("="*50)

    # Sample sensor inputs with different risk levels
    test_cases = [
        {
            "input": {
                "Type": 1,
                "Air_temperature_K": 300.0,
                "Process_temperature_K": 310.5,
                "Rotational_speed_rpm": 1500,
                "Torque_Nm": 45.0,
                "Tool_wear_min": 50
            },
            "prob": 0.10,
            "label": 0,
            "expected": "GREEN"
        },
        {
            "input": {
                "Type": 1,
                "Air_temperature_K": 302.0,
                "Process_temperature_K": 312.0,
                "Rotational_speed_rpm": 1400,
                "Torque_Nm": 55.0,
                "Tool_wear_min": 150
            },
            "prob": 0.45,
            "label": 0,
            "expected": "YELLOW"
        },
        {
            "input": {
                "Type": 2,
                "Air_temperature_K": 304.0,
                "Process_temperature_K": 313.5,
                "Rotational_speed_rpm": 1200,
                "Torque_Nm": 65.0,
                "Tool_wear_min": 220
            },
            "prob": 0.85,
            "label": 1,
            "expected": "RED"
        }
    ]

    print("\n── Testing 3 Alert Levels ───────────────────")
    all_pass = True
    for i, tc in enumerate(test_cases):
        alert = process_alert(tc["input"], tc["prob"], tc["label"])
        match = alert["alert_level"] == tc["expected"]
        if not match:
            all_pass = False
        print(f"\nTest {i+1}: prob={tc['prob']} "
              f"→ {alert['alert_level']} "
              f"({'PASS' if match else 'FAIL'})")
        print(f"  {alert['message']}")

    print("\n── Alert Summary ────────────────────────────")
    get_alert_summary()

    print(f"\n[INFO] All tests passed: {all_pass}")
    print("[INFO] alert_system.py — Day 16 verified!")