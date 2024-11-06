import torch
import os
import random
import json
from torch import nn
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

RATE_OF_DECAY = 0.4
RATE_OF_RECOVERY = 0.1
FLATTENING_COEFFICIENT = 0.9
PROOF_SIZE_THRESHOLD = 3648
PROOF_SIZE_WEIGHT = 0
RESPONSE_TIME_WEIGHT = 1
MAXIMUM_RESPONSE_TIME_DECIMAL = 0.99
BATCH_SIZE = 1024
MAX_RESPONSE_TIME = 30
MIN_RESPONSE_TIME = 0
NUM_KEYS_TO_SIMULATE = 256
PLOT_INTERVALS = 25
MAX_SCORE = 1 / 256
ENABLE_LOGS = False
FIX_TIMES_AFTER_INTERVAL = False
SCALE = 18


class Reward(nn.Module):
    """
    This module is responsible for calculating the reward for a miner based on the provided score, verification_result,
    response_time, and proof_size in it's forward pass.
    """

    def __init__(self):
        super().__init__()
        # Convert constants to 1D tensors with shape [1]
        self.RATE_OF_DECAY = torch.tensor([RATE_OF_DECAY])
        self.RATE_OF_RECOVERY = torch.tensor([RATE_OF_RECOVERY])
        self.FLATTENING_COEFFICIENT = torch.tensor([FLATTENING_COEFFICIENT])
        self.PROOF_SIZE_THRESHOLD = torch.tensor([PROOF_SIZE_THRESHOLD])
        self.PROOF_SIZE_WEIGHT = torch.tensor([PROOF_SIZE_WEIGHT])
        self.RESPONSE_TIME_WEIGHT = torch.tensor([RESPONSE_TIME_WEIGHT])
        self.MAXIMUM_RESPONSE_TIME_DECIMAL = torch.tensor(
            [MAXIMUM_RESPONSE_TIME_DECIMAL]
        )

    def shifted_tan(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Shifted tangent curve
        """
        return torch.tan(
            torch.mul(
                torch.mul(torch.sub(x, torch.tensor(0.5)), torch.pi),
                self.FLATTENING_COEFFICIENT,
            )
        )

    def tan_shift_difference(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Difference
        """
        return torch.sub(self.shifted_tan(x), self.shifted_tan(torch.tensor(0.0)))

    def normalized_tangent_curve(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.div(
            self.tan_shift_difference(x), self.tan_shift_difference(torch.tensor(1.0))
        )

    def forward(
        self,
        maximum_score: torch.FloatTensor,
        previous_score: torch.FloatTensor,
        verified: torch.BoolTensor,
        proof_size: torch.IntTensor,
        response_time: torch.FloatTensor,
        maximum_response_time: torch.FloatTensor,
        minimum_response_time: torch.FloatTensor,
        validator_uid: torch.IntTensor,
    ):
        """
        This method calculates the reward for a miner based on the provided score, verification_result,
        response_time, and proof_size using a neural network module.
        Positional Arguments:
            max_score (FloatTensor): The maximum score for the miner.
            score (FloatTensor): The current score for the miner.
            verified (BoolTensor): Whether the response that the miner submitted was valid.
            proof_size (FloatTensor): The size of the proof.
            response_time (FloatTensor): The time taken to respond to the query.
            maximum_response_time (FloatTensor): The maximum response time received from validator queries
            minimum_response_time (FloatTensor): The minimum response time received from validator queries
            validator_uid (IntTensor): The validator's uid
        Returns:
            [new_score, validator_uid]
        """

        # Determine rate of scoring change based on whether the response was verified
        rate_of_change = torch.where(
            verified,
            self.RATE_OF_RECOVERY.expand(BATCH_SIZE),
            self.RATE_OF_DECAY.expand(BATCH_SIZE),
        )

        # Normalize the response time into a decimal between zero and the maximum response time decimal
        # Maximum is capped at maximum response time decimal here to limit degree of score reduction
        # in cases of very poor performance
        response_time_normalized = torch.clamp(
            torch.div(
                torch.sub(response_time, minimum_response_time),
                torch.sub(maximum_response_time, minimum_response_time),
            ),
            min=torch.tensor(0.0),
            max=self.MAXIMUM_RESPONSE_TIME_DECIMAL,
        )

        # Calculate reward metrics from both response time and proof size
        response_time_reward_metric = torch.mul(
            self.RESPONSE_TIME_WEIGHT,
            torch.sub(
                torch.tensor(1), self.normalized_tangent_curve(response_time_normalized)
            ),
        )
        proof_size_reward_metric = torch.mul(
            self.PROOF_SIZE_WEIGHT,
            torch.clamp(
                proof_size / self.PROOF_SIZE_THRESHOLD, torch.tensor(0), torch.tensor(1)
            ),
        )

        # Combine reward metrics to provide a final score based on provided inputs
        calculated_score_fraction = torch.clamp(
            torch.sub(response_time_reward_metric, proof_size_reward_metric),
            torch.tensor(0),
            torch.tensor(1),
        )

        # Adjust the maximum score for the miner based on calculated metrics
        maximum_score = torch.mul(maximum_score, calculated_score_fraction)

        # Get the distance of the previous score from the new maximum or zero, depending on verification status
        distance_from_score = torch.where(
            verified, torch.sub(maximum_score, previous_score), previous_score
        )

        # Calculate the difference in scoring that will be applied based on the rate and distance from target score
        change_in_score = torch.mul(rate_of_change, distance_from_score)

        # Provide a new score based on their previous score and change in score. In cases where verified is false,
        # scores are always decreased.
        new_score = torch.where(
            verified,
            previous_score + change_in_score,
            previous_score - change_in_score,
        )

        return [new_score, validator_uid[0]]


model = torch.compile(Reward())
model.eval()
example_inputs = (
    torch.full((BATCH_SIZE,), MAX_SCORE, dtype=torch.float32),
    torch.full((BATCH_SIZE,), 0, dtype=torch.float32),
    torch.tensor([i != BATCH_SIZE - 1 for i in range(BATCH_SIZE)], dtype=torch.bool),
    torch.tensor(
        [random.randint(0, 10000) for _ in range(BATCH_SIZE)], dtype=torch.int32
    ),
    torch.tensor(
        [
            random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME + 2)
            for _ in range(BATCH_SIZE)
        ],
        dtype=torch.float32,
    ),
    torch.full((BATCH_SIZE,), MAX_RESPONSE_TIME, dtype=torch.float32),
    torch.full((BATCH_SIZE,), MIN_RESPONSE_TIME, dtype=torch.float32),
    torch.full((BATCH_SIZE,), random.randint(0, 256), dtype=torch.int32),
)

input_data = {
    "input_data": [
        tensor.tolist() if tensor.dim() > 0 else [tensor.item()]
        for tensor in example_inputs
    ]
}

input_data["input_shapes"] = [tensor.shape for tensor in example_inputs]

with open("input.json", "w") as f:
    json.dump(input_data, f)
with open("calibration.json", "w") as f:
    json.dump(input_data, f)

torch.onnx.export(
    Reward(),
    example_inputs,
    "network.onnx",
    input_names=[
        "maximum_score",
        "previous_score",
        "verified",
        "proof_size",
        "response_time",
        "maximum_response_time",
        "minimum_response_time",
        "validator_uid",
    ],
    output_names=["new_score", "validator_uid"],
    export_params=False,
    opset_version=17,
)

os.system("ezkl gen-settings --input-visibility private --param-visibility fixed")
os.system(
    f"ezkl calibrate-settings --target accuracy --max-logrows 17 --scales {SCALE}"
)
os.system("ezkl get-srs")
os.system("ezkl compile-circuit")
os.system("ezkl setup")
os.system("ezkl gen-witness")
os.system("ezkl prove")
os.system("ezkl verify")

# Get raw outputs from model
raw_outputs = Reward()(*example_inputs)

# Get outputs from proof
with open("proof.json", "r") as f:
    proof = json.load(f)
    scaled_outputs = []

    # Handle float outputs (1024 values)
    float_outputs = [
        float(x) for x in proof["pretty_public_inputs"]["rescaled_outputs"][0][:1024]
    ]
    scaled_outputs.append(torch.tensor(float_outputs, dtype=torch.float32))

    # Handle int output (1 value)
    int_output = int(proof["pretty_public_inputs"]["rescaled_outputs"][1][0])
    scaled_outputs.append(torch.tensor([int_output], dtype=torch.int32))

# Create comparison table
output_names = ["New Score", "Validator UID"]
comparison_data = []
for i in range(len(raw_outputs)):
    raw = raw_outputs[i]
    scaled = scaled_outputs[i]

    # Compare each value and take mean of absolute differences
    diffs = []
    if raw.dim() > 0:  # Check if tensor is not 0-dimensional
        for j in range(len(raw)):
            if not torch.isinf(raw[j]) and not torch.isinf(scaled[j]):
                diffs.append(abs(raw[j] - scaled[j]))
    else:
        if not torch.isinf(raw) and not torch.isinf(scaled):
            diffs.append(abs(raw - scaled))

    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    comparison_data.append(
        {
            "Output": output_names[i],
            "Avg Raw Value": raw.item() if raw.dim() == 0 else torch.mean(raw).item(),
            "Avg Proof Value": (
                scaled.item() if scaled.dim() == 0 else scaled.float().mean().item()
            ),
            "Avg Absolute Diff": avg_diff,
        }
    )

df = pd.DataFrame(comparison_data)
print("\nOutput Comparison:")
print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".8f"))

# Sort scores and get corresponding response times
raw_scores = raw_outputs[0].numpy()
scaled_scores = scaled_outputs[0].numpy()
response_times = example_inputs[4].numpy()

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Raw outputs
sorted_indices = np.argsort(raw_scores)
scatter1 = ax1.scatter(
    range(len(raw_scores)),
    raw_scores[sorted_indices],
    c=response_times[sorted_indices],
    cmap="viridis",
)
ax1.set_title("Raw Output Scores (Sorted)")
ax1.set_xlabel("Index")
ax1.set_ylabel("Score")
fig.colorbar(scatter1, ax=ax1, label="Response Time")

# Plot 2: Proof outputs
sorted_indices = np.argsort(scaled_scores)
scatter2 = ax2.scatter(
    range(len(scaled_scores)),
    scaled_scores[sorted_indices],
    c=response_times[sorted_indices],
    cmap="viridis",
)
ax2.set_title("Proof Output Scores (Sorted)")
ax2.set_xlabel("Index")
ax2.set_ylabel("Score")
fig.colorbar(scatter2, ax=ax2, label="Response Time")

plt.tight_layout()
plt.show()
