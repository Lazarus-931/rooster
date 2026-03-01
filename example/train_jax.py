"""
JAX training simulation — rooster example
==========================================

Simulates a small Transformer training run on a text-classification task.
No GPU or real framework required — all curves are computed analytically
with realistic noise so the dashboard has interesting data to display.

Metrics logged
--------------
  loss          — cross-entropy, every step
  accuracy      — token-level train accuracy, every step
  val_loss      — validation cross-entropy, every 25 steps
  val_accuracy  — validation accuracy, every 25 steps
  perplexity    — exp(loss), every step
  grad_norm     — gradient global L2 norm, every step

Run duration: ~4 minutes (2 800 steps × 0.086 s)

Usage
-----
  python example/train_jax.py
"""

import math
import random
import time

from rooster import Arrange, Collect, Define, MetricDef, Send, launch_dashboard

# ── 0. Open the dashboard ─────────────────────────────────────────────────────
launch_dashboard()

# ── 1. Declare the run ────────────────────────────────────────────────────────

STEPS   = 2_800
WARMUP  = 200
LR_PEAK = 3e-4
LR_MIN  = 1e-5

definition = Define(
    project_name="transformer_textcls",
    project_description="Small Transformer on text classification (simulated)",
    session_name=f"run_{random.randint(1000, 9999)}",
    framework="jax",
    metrics={
        "loss":         MetricDef(rate=1),
        "accuracy":     MetricDef(rate=1),
        "val_loss":     MetricDef(rate=25),
        "val_accuracy": MetricDef(rate=25),
        "perplexity":   MetricDef(rate=1),
        "grad_norm":    MetricDef(rate=1),
    },
)

# ── 2. Wire up components ─────────────────────────────────────────────────────

collect = Collect(framework="jax", metrics=definition.metrics)
arrange = Arrange(definition=definition, collector=collect)
send    = Send()

send.register(arrange)
collect.attach(send, arrange)

# ── 3. LR schedule: inverse-sqrt warmup → linear decay ───────────────────────

def lr_schedule(step: int) -> float:
    if step < WARMUP:
        return LR_PEAK * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR_MIN + (LR_PEAK - LR_MIN) * (1.0 - progress) ** 0.5


# ── 4. Training loop ──────────────────────────────────────────────────────────

print(f"\nrooster — session : {definition.session.name}")
print(f"          project : {definition.project.name}")
print(f"          metrics : {list(definition.metrics)}")
print(f"          steps   : {STEPS}  (~4 min)\n")

for step in range(STEPS):
    t = step / STEPS

    # Training loss: slower descent typical of Transformers on text
    slow_start  = math.exp(-2.0 * max(0, step - WARMUP) / STEPS * 5)
    base_loss   = 3.8 * slow_start + 0.55 * (1 - slow_start) + 0.28
    noise_scale = 0.06 * (0.4 + 0.6 * slow_start)
    loss        = max(0.01, base_loss + random.gauss(0.0, noise_scale))

    # Perplexity = exp(loss) — clamp at a reasonable ceiling
    perplexity = min(math.exp(loss), 200.0)

    # Training accuracy: slower ramp than image models
    base_acc  = 0.88 * (1.0 - math.exp(-5.0 * t))
    accuracy  = max(0.0, min(1.0, base_acc + random.gauss(0.0, 0.012)))

    # Validation metrics (rate-gated to every 25 steps)
    val_loss     = loss     + abs(random.gauss(0.08, 0.02))
    val_accuracy = accuracy - abs(random.gauss(0.03, 0.01))
    val_accuracy = max(0.0, min(1.0, val_accuracy))

    # Gradient norm: sharp spike during warmup, settles with occasional bursts
    grad_base = 4.0 * math.exp(-step / 400.0) + 0.2
    spike     = 2.0 if random.random() < 0.008 else 0.0   # rare gradient spike
    grad_norm = grad_base + abs(random.gauss(0.0, grad_base * 0.2)) + spike

    collect.log({"value": loss},         kind="loss")
    collect.log({"value": accuracy},     kind="accuracy")
    collect.log({"value": val_loss},     kind="val_loss")
    collect.log({"value": val_accuracy}, kind="val_accuracy")
    collect.log({"value": perplexity},   kind="perplexity")
    collect.log({"value": grad_norm},    kind="grad_norm")

    if step % 280 == 0:
        print(
            f"  step {step:5d}/{STEPS}  "
            f"loss={loss:.4f}  ppl={perplexity:.2f}  "
            f"acc={accuracy:.4f}  val_acc={val_accuracy:.4f}  "
            f"grad={grad_norm:.3f}"
        )

    time.sleep(0.086)

# ── 5. End ────────────────────────────────────────────────────────────────────

send.end(arrange)
print(f"\nDone. Session '{definition.session.name}' finished {STEPS} steps.")
