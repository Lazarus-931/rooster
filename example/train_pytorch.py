"""
PyTorch training simulation — rooster example
==============================================

Simulates a ResNet-18 training run on CIFAR-10.
No GPU or real framework required — all curves are computed analytically
with realistic noise so the dashboard has interesting data to display.

Metrics logged
--------------
  loss          — cross-entropy, every step
  accuracy      — top-1 train accuracy, every step
  val_loss      — validation cross-entropy, every 20 steps
  val_accuracy  — validation top-1 accuracy, every 20 steps
  grad_norm     — gradient L2 norm, every step
  lr            — learning rate (linear warmup + cosine decay), every step

Run duration: ~3.5 minutes (2 500 steps × 0.085 s)

Usage
-----
  python example/train_pytorch.py
"""

import math
import random
import time

from rooster import Arrange, Collect, Define, MetricDef, Send, launch_dashboard

# ── 0. Open the dashboard ─────────────────────────────────────────────────────
# Opens a new Terminal window running `cargo run` (server + TUI).
# Remove this call if you start the server separately.
launch_dashboard()

# ── 1. Declare the run ────────────────────────────────────────────────────────

STEPS     = 2_500
WARMUP    = 150        # linear LR warmup steps
LR_MAX    = 0.1
LR_MIN    = 1e-4

definition = Define(
    project_name="cifar10_resnet",
    project_description="ResNet-18 on CIFAR-10 (simulated)",
    session_name=f"run_{random.randint(1000, 9999)}",
    framework="pytorch",
    metrics={
        "loss":         MetricDef(rate=1),
        "accuracy":     MetricDef(rate=1),
        "val_loss":     MetricDef(rate=20),
        "val_accuracy": MetricDef(rate=20),
        "grad_norm":    MetricDef(rate=1),
        "lr":           MetricDef(rate=1),
    },
)

# ── 2. Wire up components ─────────────────────────────────────────────────────

collect = Collect(framework="pytorch", metrics=definition.metrics)
arrange = Arrange(definition=definition, collector=collect)
send    = Send()

send.register(arrange)
collect.attach(send, arrange)

# ── 3. LR schedule: linear warmup → cosine decay ─────────────────────────────

def lr_schedule(step: int) -> float:
    if step < WARMUP:
        return LR_MAX * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


# ── 4. Training loop ──────────────────────────────────────────────────────────

print(f"\nrooster — session : {definition.session.name}")
print(f"          project : {definition.project.name}")
print(f"          metrics : {list(definition.metrics)}")
print(f"          steps   : {STEPS}  (~3.5 min)\n")

for step in range(STEPS):
    t = step / STEPS

    # Learning rate
    lr = lr_schedule(step)

    # Training loss: fast drop during warmup, then slow decay + noise
    base_loss   = 2.4 * math.exp(-4.5 * t) + 0.32
    noise_loss  = random.gauss(0.0, 0.04 * (1 + 0.5 * math.exp(-6 * t)))
    loss        = max(0.01, base_loss + noise_loss)

    # Training accuracy: sigmoid ramp, plateaus near 93%
    base_acc   = 0.93 * (1.0 - math.exp(-6.0 * t))
    noise_acc  = random.gauss(0.0, 0.008)
    accuracy   = max(0.0, min(1.0, base_acc + noise_acc))

    # Validation metrics (logged every 20 steps via rate gate)
    val_loss     = loss     + abs(random.gauss(0.06, 0.015))
    val_accuracy = accuracy - abs(random.gauss(0.025, 0.008))
    val_accuracy = max(0.0, min(1.0, val_accuracy))

    # Gradient norm: spikes during warmup, then decays
    grad_base = 3.5 * math.exp(-step / 300.0) + 0.15
    grad_norm = grad_base + abs(random.gauss(0.0, grad_base * 0.25))

    collect.log({"value": loss},         kind="loss")
    collect.log({"value": accuracy},     kind="accuracy")
    collect.log({"value": val_loss},     kind="val_loss")
    collect.log({"value": val_accuracy}, kind="val_accuracy")
    collect.log({"value": grad_norm},    kind="grad_norm")
    collect.log({"value": lr},           kind="lr")

    if step % 250 == 0:
        print(
            f"  step {step:5d}/{STEPS}  "
            f"loss={loss:.4f}  acc={accuracy:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}  "
            f"grad={grad_norm:.3f}  lr={lr:.5f}"
        )

    time.sleep(0.085)

# ── 5. End ────────────────────────────────────────────────────────────────────

send.end(arrange)
print(f"\nDone. Session '{definition.session.name}' finished {STEPS} steps.")
