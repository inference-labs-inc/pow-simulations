import json
import random

# Generate 1024 sets of inputs
n = 1024
inputs = {
    "actual_price": [random.randint(7000000000, 9000000000) for _ in range(n)],
    "predicted_price": [random.randint(7000000000, 9000000000) for _ in range(n)],
    "date_difference": [random.randint(0, 14) for _ in range(n)],
    "price_weight": [86] * n,  # Fixed weight of 86 for all entries
    "date_weight": [14] * n,  # Fixed weight of 14 for all entries
}

# Write to input.json
with open("input.json", "w") as f:
    json.dump(inputs, f)
