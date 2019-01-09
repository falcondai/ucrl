import numpy as np


class MDP:
    def reset(self, init_state=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class SimpleMDP(MDP):
    def __init__(self, n_states, n_actions, p, r, initial_state_distribution=None):
        self.p = np.asarray(p)
        self.r = np.asarray(r)
        assert self.p.shape == (n_states, n_actions, n_states)
        assert self.r.shape == (n_states, n_actions)

        self.n_states = n_states
        self.n_actions = n_actions
        # Default initial state distribution is uniform
        self.initial_state_distribution = initial_state_distribution or np.ones(self.n_states) / self.n_states

    def reset(self, initial_state=None):
        if initial_state is None:
            self.state = np.random.choice(self.n_states, p=self.initial_state_distribution)
        else:
            self.state = initial_state
        return self.state

    def step(self, action):
        next_state = np.random.choice(self.n_states, p=self.p[self.state, action])
        reward = self.r[self.state, action]
        self.state = next_state
        return next_state, reward


if __name__ == '__main__':
    eps = 0.2
    alpha = 0.1
    p = [
            [[1, 0], [1 - eps, eps]],
            [[0, 1], [eps, 1 - eps]],
         ]
    r = [
            [1 - alpha, 1 - alpha],
            [1, 1],
        ]
    mdp = SimpleMDP(2, 2, p, r)
    print(mdp.reset(0))
    for i in range(10):
        ac = np.random.randint(2)
        print(ac, mdp.step(ac))
