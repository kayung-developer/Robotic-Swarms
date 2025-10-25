import numpy as np
import matplotlib.pyplot as plt


class RemediationEnv:
    def __init__(self, size=100, F=0.035, k=0.065, Du=0.16, Dv=0.08):
        self.size = size
        self.F, self.k, self.Du, self.Dv = F, k, Du, Dv

        self.U = np.ones((size, size))
        self.V = np.zeros((size, size))

        # Initial spill
        r = size // 8
        center = size // 2
        y, x = np.ogrid[-center:size - center, -center:size - center]
        mask = x * x + y * y <= r * r
        self.U[mask] = 0.5
        self.V[mask] = 0.25
        self.U += 0.05 * np.random.random((size, size))
        self.V += 0.05 * np.random.random((size, size))

    def laplacian(self, grid):
        return (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid)

    def step(self, dt=1.0):
        lap_U = self.laplacian(self.U)
        lap_V = self.laplacian(self.V)

        reaction = self.U * self.V ** 2

        delta_U = self.Du * lap_U - reaction + self.F * (1 - self.U)
        delta_V = self.Dv * lap_V + reaction - (self.F + self.k) * self.V

        self.U += delta_U * dt
        self.V += delta_V * dt


def run_remediation_sim(total_steps=5000):
    env = RemediationEnv()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for step in range(total_steps):
        if step == 0:
            axes[0].imshow(env.V, cmap='magma')
            axes[0].set_title('Step 0: Initial Spill')
        if step == 1000:
            axes[1].imshow(env.V, cmap='magma')
            axes[1].set_title(f'Step {step}: Neutralization in Progress')
        env.step()

    axes[2].imshow(env.V, cmap='magma')
    axes[2].set_title(f'Step {total_steps}: Contaminant Neutralized')
    for ax in axes: ax.set_axis_off()
    plt.suptitle('Figure 5. Simulation of Environmental Remediation', fontsize=16)
    plt.show()


run_remediation_sim()