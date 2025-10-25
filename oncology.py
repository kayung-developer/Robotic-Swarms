import numpy as np
import matplotlib.pyplot as plt


class OncologyEnv:
    def __init__(self, size=50, num_bots=40):
        self.size = size
        self.grid = np.ones((size, size))  # 1: Healthy

        # Create a tumor (value 2)
        y, x = np.ogrid[-size // 2:size // 2, -size // 2:size // 2]
        mask = x * x + y * y <= 8 ** 2
        self.grid[mask] = 2

        self.bots = [{'pos': (np.random.randint(size), np.random.randint(size))} for _ in range(num_bots)]

    def step(self):
        for bot in self.bots:
            y, x = bot['pos']
            # Differentiate and act
            if self.grid[y, x] == 2:  # Found tumor cell
                self.grid[y, x] = 0  # Induce apoptosis (benign)
            else:  # On healthy or empty cell, search
                dy, dx = np.random.randint(-1, 2), np.random.randint(-1, 2)
                bot['pos'] = (np.clip(y + dy, 0, self.size - 1), np.clip(x + dx, 0, self.size - 1))


def run_oncology_sim(steps=400):
    env = OncologyEnv()
    initial_state = env.grid.copy()

    for _ in range(steps): env.step()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.get_cmap('viridis', 3)
    axes[0].imshow(initial_state, cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title('Before: Healthy Tissue (Purple) with Tumor (Yellow)')
    axes[1].imshow(env.grid, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title('After: Tumor Eradicated')
    for ax in axes: ax.set_axis_off()
    plt.suptitle('Figure 3. Simulation of Targeted Oncology', fontsize=16)
    plt.show()


run_oncology_sim()