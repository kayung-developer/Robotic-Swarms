import numpy as np
import matplotlib.pyplot as plt


def rule_30(neighbors):
    # Implements the Rule 30 logic based on a 3-bit input
    pattern = tuple(neighbors)
    rules = {
        (1, 1, 1): 0, (1, 1, 0): 0, (1, 0, 1): 0, (1, 0, 0): 1,
        (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): 0
    }
    return rules[pattern]


def run_encryption_sim(width=101, generations=50):
    # Initial data: a single '1' on a background of '0's (plaintext)
    initial_data = np.zeros(width, dtype=int)
    initial_data[width // 2] = 1

    # History of the data transformations (the encryption process)
    history = [initial_data]

    current_gen = initial_data
    for _ in range(generations - 1):
        next_gen = np.zeros_like(current_gen)
        for i in range(1, width - 1):
            neighbors = (current_gen[i - 1], current_gen[i], current_gen[i + 1])
            next_gen[i] = rule_30(neighbors)
        history.append(next_gen)
        current_gen = next_gen

    encrypted_data = np.array(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Show initial data
    axes[0].imshow(initial_data.reshape(1, -1), cmap='binary', aspect='auto')
    axes[0].set_title('Plaintext Data (Initial State)')
    axes[0].set_ylabel('Time')
    axes[0].set_xlabel('Data Position')

    # Show encrypted data over time
    axes[1].imshow(encrypted_data, cmap='binary', aspect='auto')
    axes[1].set_title('Physically Encrypted Data (Rule 30)')
    axes[1].set_xlabel('Data Position')

    plt.suptitle('Figure 6. Simulation of Physical Data Encryption via Cellular Automaton', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


run_encryption_sim()