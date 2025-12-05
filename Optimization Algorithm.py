# PSO Optimization Class
class PSOOptimizer:
    def __init__(self, n_particles=10, n_iterations=5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf

    def initialize_swarm(self):
        search_space = {
            'num_units': [64, 128, 256],
            'num_layers': [1, 2, 3]
        }

        for _ in range(self.n_particles):
            particle = {
                'position': {
                    'num_units': np.random.choice(search_space['num_units']),
                    'num_layers': np.random.choice(search_space['num_layers'])
                },
                'velocity': {
                    'num_units': np.random.uniform(-10, 10),
                    'num_layers': np.random.uniform(-1, 1)
                },
                'best_position': None,
                'best_fitness': -np.inf
            }
            self.swarm.append(particle)

    def update_velocity_position(self, w=0.5, c1=1.5, c2=1.5):
        for particle in self.swarm:
            # Update velocity
            r1, r2 = np.random.rand(), np.random.rand()

            # For num_units
            particle['velocity']['num_units'] = (
                w * particle['velocity']['num_units'] +
                c1 * r1 * (particle['best_position']['num_units'] - particle['position']['num_units']) +
                c2 * r2 * (self.global_best_position['num_units'] - particle['position']['num_units'])
            )
