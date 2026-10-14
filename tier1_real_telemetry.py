"""
Project SPANDAN Tier 1: Real Telemetry Ingestion Engine
Solves inverse damped oscillator ODE on real ESP32 accelerometer streams.
Zero double-integration drift; noise-regularized via PyTorch Autograd.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from alert_system import ConnectedAlertDispatcher

# Enforce deterministic cross-platform execution
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleSensorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

    def forward(self, t):
        return self.net(t)


def ingest_real_tier1_telemetry(file_path: str, max_duration: float = 2.5):
    """
    Ingests accelerometer CSV from hardware.
    Expected CSV columns: [timestamp, acceleration]
    Handles both millisecond and second timestamps; centers dynamic baseline.
    """
    df = pd.read_csv(file_path)
    t_raw = df.iloc[:, 0].to_numpy(dtype=np.float32)
    a_raw = df.iloc[:, 1].to_numpy(dtype=np.float32)

    # Convert timestamps to relative seconds
    if t_raw[0] > 1e4 or (len(t_raw) > 1 and (t_raw[1] - t_raw[0]) > 5.0):
        t_sec = (t_raw - t_raw[0]) / 1000.0  # Milliseconds to seconds
    else:
        t_sec = t_raw - t_raw[0]

    # Crop to standard Operational Modal Analysis (OMA) window
    mask = t_sec <= max_duration
    t_win = t_sec[mask]
    a_win = a_raw[mask]

    # Dynamic zero-centering: subtracting global mean removes static 1g gravity offset
    a_dynamic = a_win - np.mean(a_win)
    return t_win, a_dynamic


def solve_tier1_real(t_sec, a_meas, baseline_k: float = 64.0, epochs: int = 1500):
    """
    Inverts ODE: u''(t) + 2*beta*u'(t) + omega_n^2*u(t) = 0
    Direct unscaled seconds ensure gradient stability matching in-silico benchmarks.
    """
    model = SingleSensorField().to(DEVICE)
    t_tensor = torch.tensor(t_sec, dtype=torch.float32).view(-1, 1).to(DEVICE).requires_grad_(True)
    a_sensor = torch.tensor(a_meas, dtype=torch.float32).view(-1, 1).to(DEVICE)

    omega_sq = nn.Parameter(torch.tensor([25.0], device=DEVICE, requires_grad=True))
    beta = nn.Parameter(torch.tensor([0.2], device=DEVICE, requires_grad=True))

    opt_field = optim.Adam(model.parameters(), lr=0.008)
    opt_phys = optim.Adam([omega_sq, beta], lr=0.08)

    for epoch in range(epochs + 1):
        opt_field.zero_grad()
        opt_phys.zero_grad()

        u_pred = model(t_tensor)
        u_t = torch.autograd.grad(u_pred, t_tensor, torch.ones_like(u_pred), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_tensor, torch.ones_like(u_t), create_graph=True)[0]

        loss_data = torch.mean((u_tt - a_sensor) ** 2)
        residual = u_tt + 2.0 * torch.abs(beta) * u_t + torch.abs(omega_sq) * u_pred
        loss_ode = torch.mean(residual ** 2)

        total_loss = loss_data + 0.12 * loss_ode
        total_loss.backward()

        opt_field.step()
        if epoch > 150:
            opt_phys.step()

    discovered_k = omega_sq.item()
    degradation = max(0.0, (baseline_k - discovered_k) / baseline_k * 100.0)
    return discovered_k, degradation


if __name__ == "__main__":
    dispatcher = ConnectedAlertDispatcher()

    # Generate synthetic real-world test log if none exists
    sample_csv = "esp32_sensor_log.csv"
    if not os.path.exists(sample_csv):
        t_dummy = np.linspace(0.0, 2.5, 250, dtype=np.float32)
        # Simulate damaged beam (omega_n^2 = 38.4, beta = 0.6) with 1g offset and 5% noise
        a_clean = np.exp(-0.6 * t_dummy) * (-38.4 * np.cos(np.sqrt(38.4 - 0.36) * t_dummy))
        a_dummy = 9.806 + a_clean + 0.05 * np.std(a_clean) * np.random.randn(250)
        pd.DataFrame({"timestamp_ms": t_dummy * 1000.0, "accel_z": a_dummy}).to_csv(sample_csv, index=False)

    print(f"[Project SPANDAN Tier 1] Ingesting telemetry: {sample_csv}")
    t, a = ingest_real_tier1_telemetry(sample_csv)
    k_curr, drop_pct = solve_tier1_real(t, a, baseline_k=64.0)

    print(f"[Tier 1 Inversion Complete] Discovered omega_n^2: {k_curr:.2f} | Measured Drop: {drop_pct:.1f}%")
    dispatcher.evaluate_and_alert(
        tier="Tier 1 (Lumped Sentry ODE)",
        structure_id="Churu-Govt-Sec-School-BlockB",
        baseline_val=64.0,
        current_val=k_curr,
        degradation_pct=drop_pct
    )
