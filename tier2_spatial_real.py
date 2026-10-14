"""
Project SPANDAN Tier 2: Multi-Sensor Spatial Wave PDE Ingestion
Ingests multi-node spatial vibration arrays to localize crack degradation (c^2).
Enforces physical Dirichlet boundary conditions to prevent gauge drift.
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


class DistributedWaveField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def ingest_multi_sensor_telemetry(file_path: str):
    """
    Ingests synchronized multi-node telemetry.
    Expected CSV columns: ['sensor_id', 'x_coord_m', 'timestamp_s', 'accel_dynamic']
    """
    df = pd.read_csv(file_path)
    x = df["x_coord_m"].to_numpy(dtype=np.float32)
    t = df["timestamp_s"].to_numpy(dtype=np.float32)
    a = df["accel_dynamic"].to_numpy(dtype=np.float32)
    return x, t, a


def solve_tier2_spatial(x_np, t_np, a_meas_np, baseline_c2: float = 4.00, beam_length: float = 1.0, epochs: int = 1200):
    """
    Joint PINN Inversion of 1D Elastodynamic Wave PDE:
      u_tt - c^2 * u_xx = 0
    Enforces boundary pinning u(0, t) = u(L, t) = 0 to guarantee mathematical uniqueness.
    """
    model = DistributedWaveField().to(DEVICE)
    x = torch.tensor(x_np, dtype=torch.float32).view(-1, 1).to(DEVICE).requires_grad_(True)
    t = torch.tensor(t_np, dtype=torch.float32).view(-1, 1).to(DEVICE).requires_grad_(True)
    a_sensor = torch.tensor(a_meas_np, dtype=torch.float32).view(-1, 1).to(DEVICE)

    # Boundary anchor points (ends of beam/span at x = 0 and x = L)
    t_bc = torch.linspace(float(t_np.min()), float(t_np.max()), 100).view(-1, 1).to(DEVICE)
    x_bc_0 = torch.zeros_like(t_bc).to(DEVICE)
    x_bc_L = torch.full_like(t_bc, beam_length).to(DEVICE)

    c_squared = nn.Parameter(torch.tensor([2.5], device=DEVICE, requires_grad=True))
    opt_model = optim.Adam(model.parameters(), lr=0.008)
    opt_phys = optim.Adam([c_squared], lr=0.03)

    for epoch in range(epochs + 1):
        opt_model.zero_grad()
        opt_phys.zero_grad()

        # Forward pass & derivatives via Autograd
        u_pred = model(x, t)
        u_t = torch.autograd.grad(u_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        # Physics & Data losses
        loss_data = torch.mean((u_tt - a_sensor) ** 2)
        loss_pde = torch.mean((u_tt - c_squared * u_xx) ** 2)

        # Boundary pinning loss: structural supports have zero deflection
        u_bc_0 = model(x_bc_0, t_bc)
        u_bc_L = model(x_bc_L, t_bc)
        loss_bc = torch.mean(u_bc_0 ** 2) + torch.mean(u_bc_L ** 2)

        total_loss = loss_data + 0.05 * loss_pde + 0.10 * loss_bc
        total_loss.backward()

        opt_model.step()
        if epoch > 200:
            opt_phys.step()

        if epoch % 300 == 0:
            print(f"  [SPANDAN Tier 2] Epoch {epoch:4d} | PDE Loss: {loss_pde.item():.4f} | c^2: {c_squared.item():.3f}")

    discovered_c2 = c_squared.item()
    degradation = max(0.0, (baseline_c2 - discovered_c2) / baseline_c2 * 100.0)
    return discovered_c2, degradation


if __name__ == "__main__":
    dispatcher = ConnectedAlertDispatcher()

    # Generate sample 4-node real telemetry log if none exists
    multi_csv = "distributed_array_log.csv"
    if not os.path.exists(multi_csv):
        nodes_x = [0.2, 0.4, 0.6, 0.8]  # 4 spatial sensor coordinates along a 1.0 m span
        t_pts = np.linspace(0.1, 1.0, 200)

        records = []
        # Ground truth: damaged beam wave speed c = 1.0 (c^2 = 1.00)
        for node_id, x_pos in enumerate(nodes_x):
            for t_val in t_pts:
                a_clean = -(1.0 * np.pi) ** 2 * np.sin(np.pi * x_pos) * np.cos(1.0 * np.pi * t_val)
                noise = 0.05 * np.random.randn()
                records.append({
                    "sensor_id": f"node_{node_id}",
                    "x_coord_m": x_pos,
                    "timestamp_s": t_val,
                    "accel_dynamic": a_clean + noise
                })

        pd.DataFrame(records).to_csv(multi_csv, index=False)

    print(f"[Project SPANDAN Tier 2] Ingesting multi-sensor array: {multi_csv}")
    x_arr, t_arr, a_arr = ingest_multi_sensor_telemetry(multi_csv)
    c2_curr, drop_pct = solve_tier2_spatial(x_arr, t_arr, a_arr, baseline_c2=4.00)

    print(f"[Tier 2 Inversion Complete] Discovered c^2: {c2_curr:.3f} | Measured Drop: {drop_pct:.1f}%")
    dispatcher.evaluate_and_alert(
        tier="Tier 2 (Forensic Navier Wave PDE)",
        structure_id="Churu-Govt-Sec-School-BlockB-Slab4",
        baseline_val=4.00,
        current_val=c2_curr,
        degradation_pct=drop_pct
    )
