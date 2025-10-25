import numpy as np
import matplotlib.pyplot as plt

class ManufacturingEnv:
    def __init__(self, size=50, num_bots=25, num_particles=150):
        self.size = size
        self.grid = np.zeros((size, size))
        self.target_path = [(i, size // 2) for i in range(5, size - 5)]
        
        # Place raw material (value 2)
        particle_indices = np.random.choice(size * size, num_particles, replace=False)
        p_coords = np.unravel_index(particle_indices, (size, size))
        self.grid[p_coords] = 2

        self.bots = [{'pos': (np.random.randint(size), np.random.randint(size)), 'payload': False} for _ in range(num_bots)]

    def get_reward(self, bot_pos):
        return 1 if bot_pos in self.target_path else -0.1

    def step(self):
        for bot in self.bots:
            y, x = bot['pos']
            if not bot['payload']:
                material_coords = np.argwhere(self.grid == 2)
                if len(material_coords) > 0:
                    distances = np.sum((material_coords - [y, x])**2, axis=1)
                    target_pos = material_coords[np.argmin(distances)]
                    if np.array_equal(target_pos, [y, x]):
                        bot['payload'] = True
                        self.grid[y, x] = 0
                else:
                    target_pos = (y,x) # No material left
            else:
                empty_path_points = [p for p in self.target_path if self.grid[p] == 0]
                if len(empty_path_points) > 0:
                    distances = np.sum((np.array(empty_path_points) - [y, x])**2, axis=1)
                    target_pos = empty_path_points[np.argmin(distances)]
                    if np.array_equal(target_pos, [y,x]):
                        bot['payload'] = False
                        self.grid[y, x] = 1 # Deposit material
                else:
                    target_pos = (y,x) # Path is complete
            
            # Move towards target
            dy = np.sign(target_pos[0] - y)
            dx = np.sign(target_pos[1] - x)
            bot['pos'] = (np.clip(y + dy, 0, self.size-1), np.clip(x + dx, 0, self.size-1))

def run_manufacturing_sim(steps=250):
    env = ManufacturingEnv()
    initial_state = env.grid.copy()

    for _ in range(steps): env.step()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.get_cmap('magma', 3)
    axes[0].imshow(initial_state, cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title('Initial State: Raw Material (Yellow)')
    axes[1].imshow(env.grid, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title('Final State: Assembled Wire (Pink)')
    for ax in axes: ax.set_axis_off()
    plt.suptitle('Figure 2. Simulation of On-Demand Manufacturing', fontsize=16)
    plt.show()

run_manufacturing_sim()