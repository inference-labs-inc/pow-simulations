import torch
import os
import random
import json
from torch import nn
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 1024
MAX_SCORE = 1.0
SCALE = 14


class Reward(nn.Module):
    """
    This module is responsible for calculating the reward for a miner based on the provided score, verification_result,
    response_time, and proof_size in it's forward pass.
    """

    def __init__(self):
        super().__init__()
        self.success_weight = torch.tensor(1)
        self.difficulty_weight = torch.tensor(1)
        self.time_elapsed_weight = torch.tensor(0.3)
        self.failed_penalty_weight = torch.tensor(0.4)
        self.allocation_weight = torch.tensor(0.21)
        self.pow_min_difficulty = torch.tensor(8)
        self.pow_max_difficulty = torch.tensor(32)
        self.pow_timeout = torch.tensor(30.0)
        self.max_score_challenge = torch.tensor(
            100
            * (self.success_weight + self.difficulty_weight + self.time_elapsed_weight)
        )
        self.max_score = torch.tensor(
            100
            * (
                self.success_weight
                + self.difficulty_weight
                + self.time_elapsed_weight
                + self.allocation_weight
            )
        )
        self.failed_penalty_exp = torch.tensor(1.5)
        self.half_validators = torch.tensor(128.0)

    def percent(self, val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        return torch.where(max_val == 0, torch.zeros_like(val), 100.0 * (val / max_val))

    def percent_yield(self, val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        return torch.where(
            val == 0, torch.full_like(val, 100.0), 100.0 * ((max_val - val) / max_val)
        )

    def forward(
        self,
        challenge_attempts: torch.Tensor,
        challenge_successes: torch.Tensor,
        challenge_elapsed_time_avg: torch.Tensor,
        challenge_difficulty_avg: torch.Tensor,
        last_20_challenge_failed: torch.Tensor,
        has_docker: torch.Tensor,
        allocated_hotkey: torch.Tensor,
        penalized_hotkey_count: torch.Tensor,
        validator_uid: torch.Tensor,
    ):
        # Early return condition
        zero_score_mask = (
            (last_20_challenge_failed >= 19) | (challenge_successes == 0)
        ) & ~allocated_hotkey

        # Calculate difficulty score
        difficulty_val = torch.clamp(
            challenge_difficulty_avg, self.pow_min_difficulty, self.pow_max_difficulty
        )
        difficulty_modifier = self.percent(difficulty_val, self.pow_max_difficulty)
        difficulty = difficulty_modifier * self.difficulty_weight

        # Calculate success score
        successes_ratio = self.percent(
            challenge_successes.float(), challenge_attempts.float()
        )
        successes = successes_ratio * self.success_weight

        # Calculate time elapsed score
        time_elapsed_modifier = self.percent_yield(
            challenge_elapsed_time_avg, self.pow_timeout
        )
        time_elapsed = time_elapsed_modifier * self.time_elapsed_weight

        # Calculate penalty
        last_20_failed_modifier = self.percent(
            last_20_challenge_failed.float(), torch.tensor(20.0)
        )
        failed_penalty = (
            self.failed_penalty_weight
            * torch.pow(last_20_failed_modifier / 100.0, self.failed_penalty_exp)
            * 100.0
        )

        # Calculate allocation score
        allocation_score = difficulty_modifier * self.allocation_weight

        # Calculate final score
        intermediate_score = difficulty + successes + time_elapsed - failed_penalty
        docker_adjusted_score = torch.where(
            has_docker, intermediate_score, intermediate_score / 2.0
        )

        final_score = torch.where(
            allocated_hotkey,
            self.max_score_challenge * (1.0 - self.allocation_weight)
            + allocation_score,
            docker_adjusted_score,
        )

        # Apply penalties
        penalty_ratio = torch.clamp(
            1.0 - (penalized_hotkey_count.float() / self.half_validators), min=0.0
        )
        final_score = torch.where(
            penalized_hotkey_count > 0,
            torch.where(
                penalized_hotkey_count >= self.half_validators,
                torch.zeros_like(final_score),
                final_score * penalty_ratio,
            ),
            final_score,
        )

        # Apply zero score mask and final normalization
        final_score = torch.where(
            zero_score_mask, torch.zeros_like(final_score), final_score
        )
        final_score = torch.clamp(final_score, min=0.0)

        return final_score, validator_uid


torch.compile(Reward())

example_inputs = (
    torch.tensor(
        [random.randint(0, 100) for _ in range(BATCH_SIZE)], dtype=torch.int32
    ),  # challenge_attempts
    torch.tensor(
        [random.randint(0, 100) for _ in range(BATCH_SIZE)], dtype=torch.int32
    ),  # challenge_successes
    torch.tensor(
        [random.uniform(0, 30.0) for _ in range(BATCH_SIZE)], dtype=torch.float32
    ),  # challenge_elapsed_time_avg
    torch.tensor(
        [random.randint(8, 32) for _ in range(BATCH_SIZE)], dtype=torch.float32
    ),  # challenge_difficulty_avg
    torch.tensor(
        [random.randint(0, 20) for _ in range(BATCH_SIZE)], dtype=torch.int32
    ),  # last_20_challenge_failed
    torch.tensor(
        [random.choice([True, False]) for _ in range(BATCH_SIZE)], dtype=torch.bool
    ),  # has_docker
    torch.tensor(
        [random.choice([True, False]) for _ in range(BATCH_SIZE)], dtype=torch.bool
    ),  # allocated_hotkey
    torch.tensor(
        [
            random.choices([0, random.randint(1, 5)], weights=[0.95, 0.05])[0]
            for _ in range(BATCH_SIZE)
        ],
        dtype=torch.int32,
    ),  # penalized_hotkey_count
    torch.tensor(
        [random.randint(0, 256) for _ in range(BATCH_SIZE)], dtype=torch.int32
    ),  # validator_uid
)

input_data = {"input_data": [tensor.tolist() for tensor in example_inputs]}

with open("input.json", "w") as f:
    json.dump(input_data, f)
with open("calibration.json", "w") as f:
    json.dump(input_data, f)
torch.onnx.export(Reward(), example_inputs, "network.onnx")

os.system("ezkl gen-settings --input-visibility private --param-visibility fixed")
os.system(f"ezkl calibrate-settings --target accuracy")
os.system("ezkl get-srs")
os.system("ezkl compile-circuit")
os.system("ezkl setup")
os.system("ezkl gen-witness")
os.system("ezkl prove")
os.system("ezkl verify")

raw_outputs = Reward()(*example_inputs)
raw_outputs = torch.stack(raw_outputs)

with open("proof.json", "r") as f:
    proof = json.load(f)
    scaled_outputs = []
    for output_group in proof["pretty_public_inputs"]["rescaled_outputs"]:
        values = [float(x) for x in output_group[:BATCH_SIZE]]
        scaled_outputs.append(torch.tensor(values, dtype=torch.float32))
    scaled_outputs = torch.stack(scaled_outputs)

output_names = ["Score"]
comparison_data = []

print(len(raw_outputs))

for i in range(len(raw_outputs) - 1):
    raw = raw_outputs[i]
    scaled = scaled_outputs[i]

    diffs = []
    for j in range(len(raw)):
        if not torch.isinf(raw[j]) and not torch.isinf(scaled[j]):
            diffs.append(abs(raw[j] - scaled[j]))
    avg_diff = sum(diffs) / len(diffs) if diffs else 0

    comparison_data.append(
        {
            "Output": output_names[i],
            "Avg Raw Value": torch.mean(raw).item(),
            "Avg Proof Value": torch.mean(scaled).item(),
            "Avg Absolute Diff": avg_diff,
        }
    )

df = pd.DataFrame(comparison_data)
print("\nOutput Comparison:")
print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".8f"))

model = torch.compile(Reward())


def get_factor_balance(model_outputs, inputs):
    with torch.no_grad():
        # Calculate success rate and scale by average difficulty
        success_rate = inputs[1].float() / inputs[0].float()
        difficulty = torch.clamp(
            inputs[3],
            model.pow_min_difficulty,
            model.pow_max_difficulty,
        )
        score = success_rate * torch.mean(difficulty)

        # Apply penalty for failed hotkeys
        penalties = inputs[7] > 0

        # Return -1 for penalized hotkeys, otherwise the score
        return torch.where(penalties, torch.tensor(-1.0), score)


raw_scores = raw_outputs[0].numpy()
scaled_scores = scaled_outputs[0].numpy()
factor_balance = get_factor_balance(raw_outputs, example_inputs).numpy()


def get_colors(balance):
    colors = []
    for b in balance:
        if b < 0:
            colors.append((1, 0, 0))  # Red for penalized
        else:
            colors.append((0, 1, 0))  # Green for non-penalized
    return colors


point_colors = get_colors(factor_balance)

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])

sorted_indices = np.argsort(raw_scores)
scatter1 = ax1.scatter(
    range(len(raw_scores)),
    raw_scores[sorted_indices],
    c=[point_colors[i] for i in sorted_indices],
    alpha=0.8,
)
ax1.set_title("Raw Output Scores (Sorted)")
ax1.set_xlabel("Index")
ax1.set_ylabel("Score")

sorted_indices = np.argsort(scaled_scores)
scatter2 = ax2.scatter(
    range(len(scaled_scores)),
    scaled_scores[sorted_indices],
    c=[point_colors[i] for i in sorted_indices],
    alpha=0.8,
)
ax2.set_title("Proof Output Scores (Sorted)")
ax2.set_xlabel("Index")
ax2.set_ylabel("Score")

import matplotlib as mpl

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
sm.set_array([])
plt.colorbar(sm, cax=cax, label="Red: Penalized, Green: Non-penalized")

plt.tight_layout()
plt.show()
