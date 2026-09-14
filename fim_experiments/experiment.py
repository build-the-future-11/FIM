from __future__ import annotations

import torch

from config import *
from systems import get_system, generate_batch
from models import get_model
from train import train_step
from evaluate import evaluate_model


def run_single(model_name, system_name, fim_model=None):
    device = torch.device(DEVICE)

    system = get_system(system_name, device)

    model = get_model(model_name, INPUT_DIM, HIDDEN_DIM, fim_model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\n=== Training {model_name} on {system_name} ===")

    for epoch in range(EPOCHS):
        x, y = generate_batch(system, BATCH_SIZE, STEPS, device)

        x = x.reshape(-1, x.shape[-1]).contiguous()
        y = y.reshape(-1, y.shape[-1]).contiguous()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loss = train_step(model, optimizer, x, y)

        if epoch % LOG_INTERVAL == 0:
            metrics = evaluate_model(
                model,
                system,
                device,
                ROLLOUT_STEPS,
            )

            print(
                f"[{model_name} | {system_name}] "
                f"Epoch {epoch} | "
                f"train={loss:.4f} | "
                f"1-step={metrics['one_step_mse']:.4f} | "
                f"rollout={metrics['rollout_mse']:.4f}"
            )

    return model


def run_all(fim_model=None):
    device = torch.device(DEVICE)

    results = {}

    for system_name in SYSTEMS:
        system = get_system(system_name, device)

        for model_name in MODELS:
            model = run_single(model_name, system_name, fim_model)

            metrics = evaluate_model(
                model,
                system,
                device,
                ROLLOUT_STEPS,
            )

            results[(model_name, system_name)] = metrics

            print(
                f"\nFinal [{model_name} | {system_name}] → "
                f"1-step={metrics['one_step_mse']:.4f}, "
                f"rollout={metrics['rollout_mse']:.4f}"
            )

    return results