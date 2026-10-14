"""
Project SPANDAN: Autonomous Civic & Hardware Alert Dispatcher
Dispatches synchronized hardware alerts (serial to ESP32) and cloud telemetry
(HTTP webhooks to PWD / Disaster Management dashboards) with offline fallbacks.
"""

import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ConnectedAlertDispatcher:
    def __init__(self, serial_port: str = None, baud_rate: int = 115200, webhook_url: str = None):
        self.serial_port_name = serial_port
        self.baud_rate = baud_rate
        self.webhook_url = webhook_url
        self.serial_conn = None

        if self.serial_port_name and SERIAL_AVAILABLE:
            try:
                self.serial_conn = serial.Serial(self.serial_port_name, self.baud_rate, timeout=0.1)
                logging.info(f"Connected to SPANDAN hardware node on {self.serial_port_name}")
            except Exception as e:
                logging.warning(f"Could not open serial device: {e}. Defaulting to network/simulation mode.")
        elif self.serial_port_name and not SERIAL_AVAILABLE:
            logging.warning("pyserial not installed. Running hardware alerts in simulation mode.")

    def trigger_physical_device(self, state: str = "ALARM_ON"):
        """Sends command to ESP32 node to trigger onboard buzzer/LED indicator."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                command = f"CMD:{state}\n"
                self.serial_conn.write(command.encode("utf-8"))
                logging.info(f"[SPANDAN Hardware Actuation] Dispatched: {command.strip()}")
                return
            except Exception as e:
                logging.error(f"[SPANDAN Hardware Actuation Error] {e}")
        logging.info(f"[Hardware Simulation] Node GPIO 2 Alert Output -> {state}")

    def dispatch_cloud_webhook(self, payload: dict):
        """Sends JSON alert payload to municipal PWD dashboard or webhook."""
        if self.webhook_url and REQUESTS_AVAILABLE:
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.webhook_url, json=payload, headers=headers, timeout=4)
                logging.info(f"[Cloud Dispatch] HTTP {response.status_code} - Alert broadcast successfully.")
                return
            except Exception as e:
                logging.error(f"[Cloud Dispatch Error] Webhook delivery failed: {e}")
        logging.info("[Local Cloud Log] Webhook payload generated:")
        print(json.dumps(payload, indent=2))

    def evaluate_and_alert(self, tier: str, structure_id: str, baseline_val: float,
                           current_val: float, degradation_pct: float):
        """
        Evaluates structural degradation and dispatches synchronized alerts using
        the standardized three-tier civic traffic-light protocol:
          - GREEN (Nominal): < 15% degradation
          - YELLOW (Maintenance Due): 15% - 25% degradation
          - RED (Critical Evacuation): >= 25% degradation
        """
        if degradation_pct >= 25.0:
            alert_status = "CRITICAL_RED"
            action_text = "Immediate Structural Evacuation & Traffic Closure; Emergency PWD Inspection"
            alarm_cmd = "ALARM_ON"
        elif degradation_pct >= 15.0:
            alert_status = "WARNING_YELLOW"
            action_text = "Preventive Maintenance Required; Structural Shoring & Heavy Traffic Diversion"
            alarm_cmd = "ALARM_OFF"
        else:
            alert_status = "NOMINAL_GREEN"
            action_text = "Structure Healthy; Routine Sentry Monitoring"
            alarm_cmd = "ALARM_OFF"

        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "project": "Project SPANDAN",
            "structure_id": structure_id,
            "diagnostic_tier": tier,
            "status": alert_status,
            "baseline_parameter": round(float(baseline_val), 3),
            "measured_parameter": round(float(current_val), 3),
            "degradation_pct": round(float(degradation_pct), 2),
            "action_required": action_text
        }

        if alert_status == "CRITICAL_RED":
            logging.warning(f"🚨 CRITICAL ALARM: {structure_id} ({degradation_pct:.1f}% stiffness loss)")
            self.trigger_physical_device(alarm_cmd)
            self.dispatch_cloud_webhook(payload)
        elif alert_status == "WARNING_YELLOW":
            logging.warning(f"⚠️ MAINTENANCE WARNING: {structure_id} ({degradation_pct:.1f}% deviation)")
            self.trigger_physical_device(alarm_cmd)
            self.dispatch_cloud_webhook(payload)
        else:
            logging.info(f"✅ HEALTHY: {structure_id} ({degradation_pct:.1f}% deviation)")
            self.trigger_physical_device(alarm_cmd)

        return payload
