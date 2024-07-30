import sys
import os
import concurrent.futures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import torch
import matplotlib.pyplot as plt
import numpy as np

NUM_RUNS = 20
WC_PENALTY = 0.4
COPY_START_TEMPO = 5

def simulate_yuma_run(W, S, invalid_proof_tempo):
    result = utils.Yuma2(W[:-1], S[:-1])

    run_dividends = {WC_PENALTY: {'copier': [], 'validators': []}}
    run_vtrust = {WC_PENALTY: {'copier': [], 'validators': []}}
    for tempo in range(1, NUM_RUNS + 1):
        W_varied = utils.randomize_weight(W, scale=1)

        # Pre-calculate consensus weights for the last validator
        pre_result = utils.Yuma2(W_varied[:-1], S[:-1])
        pre_calculated_weights = pre_result['server_consensus_weight'] / pre_result['server_consensus_weight'].sum()

        # Simulate the last validator submitting pre-calculated weights
        if tempo > COPY_START_TEMPO:
            W_varied[-1] = pre_calculated_weights

        # Run the actual Yuma2 simulation
        result = utils.Yuma2(W_varied, S, wc_penalty=WC_PENALTY if tempo > invalid_proof_tempo else 0)

        copier_reward = result['validator_reward_normalized'][-1].item()
        validator_rewards = result['validator_reward_normalized'][:-1].tolist()
        run_dividends[WC_PENALTY]['copier'].append(copier_reward)
        run_dividends[WC_PENALTY]['validators'].append(validator_rewards)

        copier_vtrust = result['validator_trust'][-1].item()
        validator_vtrusts = result['validator_trust'][:-1].tolist()
        run_vtrust[WC_PENALTY]['copier'].append(copier_vtrust)
        run_vtrust[WC_PENALTY]['validators'].append(validator_vtrusts)

    return run_dividends, run_vtrust

def simulate_yuma_runs(W, S, invalid_proof_tempo):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_yuma_run, W, S, invalid_proof_tempo) for _ in range(1)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results[0]

def plot_simulation(ax1, ax2, ax3, dividends, vtrust, num_runs, invalid_proof_tempo):
    x = range(1, num_runs + 1)

    # Plot dividends
    ax1.plot(x, dividends[WC_PENALTY]['copier'], label='Target Validator', color='red', linewidth=2)
    validator_dividends = np.array(dividends[WC_PENALTY]['validators'])
    median_dividends = np.median(validator_dividends, axis=1)
    std_dividends = np.std(validator_dividends, axis=1)
    ax1.plot(x, median_dividends, label='Median Validator', color='blue', linewidth=2)
    ax1.fill_between(x, median_dividends - std_dividends, median_dividends + std_dividends, color='blue', alpha=0.1)

    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Average Dividend')
    ax1.set_title('Dividend Simulation')
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)

    # Plot VTrust
    ax2.plot(x, vtrust[WC_PENALTY]['copier'], label='Target Validator', color='red', linewidth=2)
    validator_vtrust = np.array(vtrust[WC_PENALTY]['validators'])
    median_vtrust = np.median(validator_vtrust, axis=1)
    std_vtrust = np.std(validator_vtrust, axis=1)
    ax2.plot(x, median_vtrust, label='Median Validator', color='blue', linewidth=2)
    ax2.fill_between(x, median_vtrust - std_vtrust, median_vtrust + std_vtrust, color='blue', alpha=0.1)

    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('VTrust')
    ax2.set_title('VTrust Simulation')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)

    # Plot relative dividends
    median_validator_dividends = np.median(dividends[WC_PENALTY]['validators'], axis=1)
    relative_copier_dividends = np.array(dividends[WC_PENALTY]['copier']) / median_validator_dividends

    ax3.plot(x, relative_copier_dividends, label='Target Validator', color='red', linewidth=2)
    ax3.axhline(y=1, color='blue', linestyle='--', linewidth=1, label='Median Validator')
    ax3.axhline(y=0.82, color='orange', linestyle='--', linewidth=2, label='Minimum Profitability Threshold')

    ax3.set_xlabel('Tempo')
    ax3.set_ylabel('Relative Dividend')
    ax3.set_title('Relative Dividend Simulation')
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)

    for ax in (ax1, ax2, ax3):
        ax.axvline(x=invalid_proof_tempo, color='red', linestyle='--', linewidth=2)
        ax.text(invalid_proof_tempo + 0.2, ax.get_ylim()[1] * 0.95, 'Penalty Applied', ha='left', va='top', rotation=0, fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax.axvline(x=COPY_START_TEMPO, color='black', linestyle='--', linewidth=2, label='Consensus Copying Starts')
        ax.text(COPY_START_TEMPO + 0.2, ax.get_ylim()[1] * 0.95, 'Target Validator\nStarts Copying', ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

if __name__ == '__main__':
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 30))
    invalid_proof_tempo = 10

    W = utils.init_weight(20, 236)
    S = torch.full((20,), 500000)

    S = S / S.sum()

    dividends, vtrust = simulate_yuma_runs(W, S, invalid_proof_tempo)
    plot_simulation(ax1, ax2, ax3, dividends, vtrust, NUM_RUNS, invalid_proof_tempo)

    plt.tight_layout()
    plt.savefig('yuma_simulation_wc_penalty_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
