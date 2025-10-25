import numpy as np
import matplotlib.pyplot as plt


class SelfHealingEnv:
    def __init__(self, size=60, num_bots=1000, crack_width=2):
        self.size = size
        self.material = np.ones((size, size))
        self.bots = np.zeros((size, size))
        self.signal = np.zeros((size, size))

        # Create crack and initial bot/signal distribution
        crack_start, crack_end = size // 4, 3 * size // 4
        self.material[crack_start:crack_end, size // 2 - crack_width:size // 2 + crack_width] = 0
        self.signal[self.material == 0] = 1.0

        bot_indices = np.random.choice(size * size, num_bots, replace=True)
        bot_coords = np.unravel_index(bot_indices, (size, size))
        np.add.at(self.bots, bot_coords, 1)

    def diffuse(self, field, D):
        # Simple 5-point stencil finite difference
        laplacian = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                     np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 4 * field)
        return field + D * laplacian

    def step(self, D_signal=0.1, D_bots=0.02, chi=0.3):
        # 1. Signal diffuses
        self.signal = self.diffuse(self.signal, D_signal)
        self.signal[self.material == 0] = 1.0  # Crack is a constant source
        self.signal = np.clip(self.signal, 0, 1)

        # 2. Bots move via chemotaxis and diffusion
        grad_y, grad_x = np.gradient(self.signal)
        advection_y = -chi * self.bots * grad_y
        advection_x = -chi * self.bots * grad_x

        # Shift bots based on advection (simplified)
        self.bots -= (np.roll(advection_y, -1, axis=0) - np.roll(advection_y, 1, axis=0)) / 2
        self.bots -= (np.roll(advection_x, -1, axis=1) - np.roll(advection_x, 1, axis=1)) / 2
        self.bots = self.diffuse(self.bots, D_bots)
        self.bots = np.clip(self.bots, 0, None)

        # 3. Repair material
        repair_mask = (self.material == 0) & (self.bots > 0.5)
        self.material[repair_mask] = 1


def run_healing_sim(total_steps=150):
    env = SelfHealingEnv()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for step in range(total_steps):
        if step == 0:
            axes[0].imshow(env.material, cmap='gray_r')
            axes[0].set_title('Step 0: Initial Crack')
        if step == 50:
            axes[1].imshow(env.bots, cmap='inferno')
            axes[1].set_title(f'Step {step}: Bots Converging on Damage')
        env.step()

    axes[2].imshow(env.material, cmap='gray_r')
    axes[2].set_title(f'Step {total_steps}: Material Repaired')
    for ax in axes: ax.set_axis_off()
    plt.suptitle('Figure 4. Simulation of a Self-Healing Material', fontsize=16)
    plt.show()


run_healing_sim()