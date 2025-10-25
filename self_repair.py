import numpy as np
import matplotlib.pyplot as plt


class SelfRepairEnv:
    def __init__(self, width, height, num_bots=30):
        self.width = width
        self.height = height
        self.grid = np.ones((height, width))  # Intact material

        # Create a crack (value 0)
        self.grid[height // 2 - 1:height // 2 + 1, 10:width - 10] = 0

        self.bots = [{'x': np.random.randint(0, width), 'y': np.random.randint(0, height)} for _ in range(num_bots)]
        self.crack_coords = np.argwhere(self.grid == 0)

    def step(self):
        for bot in self.bots:
            # Find nearest point on the crack
            if len(self.crack_coords) > 0:
                distances = np.sum((self.crack_coords - [bot['y'], bot['x']]) ** 2, axis=1)
                nearest_crack_point = self.crack_coords[np.argmin(distances)]

                # Move towards it
                if nearest_crack_point[1] > bot['x']:
                    bot['x'] += 1
                elif nearest_crack_point[1] < bot['x']:
                    bot['x'] -= 1
                if nearest_crack_point[0] > bot['y']:
                    bot['y'] += 1
                elif nearest_crack_point[0] < bot['y']:
                    bot['y'] -= 1

                # If adjacent to crack, repair it
                if abs(bot['x'] - nearest_crack_point[1]) <= 1 and abs(bot['y'] - nearest_crack_point[0]) <= 1:
                    self.grid[nearest_crack_point[0], nearest_crack_point[1]] = 1
                    # Update crack coords
                    self.crack_coords = np.argwhere(self.grid == 0)

            # Boundary checks
            bot['x'] = np.clip(bot['x'], 0, self.width - 1)
            bot['y'] = np.clip(bot['y'], 0, self.height - 1)


def run_repair_simulation(total_steps=100):
    env = SelfRepairEnv(50, 50, num_bots=50)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Initial State
    axes[0].imshow(env.grid, cmap='bone', vmin=0, vmax=1)
    axes[0].set_title('Step 0: Material with Crack')

    # Mid-repair State
    for step in range(total_steps):
        env.step()
        if step == total_steps // 2:
            mid_state = env.grid.copy()
            axes[1].imshow(mid_state, cmap='bone', vmin=0, vmax=1)
            axes[1].set_title(f'Step {step}: Swarm Migrating & Repairing')

    # Final State
    final_state = env.grid
    axes[2].imshow(final_state, cmap='bone', vmin=0, vmax=1)
    axes[2].set_title(f'Step {total_steps}: Material Repaired')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Figure 7. Simulation of a Self-Healing Material', fontsize=16)
    plt.show()


run_repair_simulation()