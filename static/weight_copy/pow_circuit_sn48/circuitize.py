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
SCALE = 16


class Reward(nn.Module):
    """
    Calculates batch rewards for price and date predictions, supporting 1024 parallel computations.
    """

    def __init__(self):
        super().__init__()
        self.price_weight = torch.tensor(0.86)
        self.date_weight = torch.tensor(0.14)
        self.max_date_points = torch.tensor(14.0)

    def forward(
        self,
        actual_prices: torch.Tensor,  # Shape: [BATCH_SIZE]
        predicted_prices: torch.Tensor,  # Shape: [BATCH_SIZE]
        date_differences: torch.Tensor,  # Shape: [BATCH_SIZE] (pre-calculated days between dates)
        validator_uid: torch.Tensor,  # Shape: [BATCH_SIZE]
    ):
        # Calculate date scores (14 points max, 1 point deducted per day off)
        date_scores = (
            torch.clamp(self.max_date_points - date_differences, min=0)
            / self.max_date_points
        ) * 100

        # Calculate price accuracy
        price_differences = torch.abs(actual_prices - predicted_prices) / actual_prices
        price_scores = torch.clamp(100 - (price_differences * 100), min=0)

        # Combine scores with weights
        final_scores = (price_scores * self.price_weight) + (
            date_scores * self.date_weight
        )

        return final_scores, validator_uid[0]


torch.compile(Reward())

example_inputs = (
    torch.tensor(
        [random.uniform(100, 1000) for _ in range(BATCH_SIZE)], dtype=torch.float32
    ),  # actual_prices
    torch.tensor(
        [random.uniform(100, 1000) for _ in range(BATCH_SIZE)], dtype=torch.float32
    ),  # predicted_prices
    torch.tensor(
        [random.randint(0, 14) for _ in range(BATCH_SIZE)], dtype=torch.float32
    ),  # date_differences
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
os.system(f"ezkl calibrate-settings --target accuracy --scales {SCALE}")
os.system("ezkl get-srs")
os.system("ezkl compile-circuit")
os.system("ezkl setup")
os.system("ezkl gen-witness")
os.system("ezkl prove")
os.system("ezkl verify")

raw_outputs = Reward()(*example_inputs)

with open("proof.json", "r") as f:
    proof = json.load(f)
    scaled_outputs = []
    for output_group in proof["pretty_public_inputs"]["rescaled_outputs"]:
        values = [float(x) for x in output_group[:BATCH_SIZE]]
        scaled_outputs.append(torch.tensor(values, dtype=torch.float32))

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

# Simplified visualization without factor balance
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

sorted_indices = np.argsort(raw_scores)
ax1.scatter(range(len(raw_scores)), raw_scores[sorted_indices], alpha=0.8, c="blue")
ax1.set_title("Raw Output Scores (Sorted)")
ax1.set_xlabel("Index")
ax1.set_ylabel("Score")

sorted_indices = np.argsort(scaled_scores)
ax2.scatter(
    range(len(scaled_scores)), scaled_scores[sorted_indices], alpha=0.8, c="blue"
)
ax2.set_title("Proof Output Scores (Sorted)")
ax2.set_xlabel("Index")
ax2.set_ylabel("Score")

plt.tight_layout()
plt.show()
