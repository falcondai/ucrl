import itertools
import numpy as np


class MDP:
    '''Markov decision process'''

    def reset(self, init_state=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class SimpleMDP(MDP):
    '''Markov decision process as defined by transition and reward matrices'''

    def __init__(self, n_states, n_actions, p, r, initial_state_distribution=None, random_seed=None):
        self.p = np.asarray(p)
        self.r = np.asarray(r)
        assert self.p.shape == (n_states, n_actions, n_states)
        assert self.r.shape == (n_states, n_actions)

        self.n_states = n_states
        self.n_actions = n_actions
        # Default initial state distribution is uniform
        self.initial_state_distribution = initial_state_distribution or np.ones(self.n_states) / self.n_states
        self.random = np.random.RandomState(seed=random_seed)

    def reset(self, initial_state=None):
        if initial_state is None:
            self.state = self.random.choice(self.n_states, p=self.initial_state_distribution)
        else:
            self.state = initial_state
        return self.state

    def step(self, action):
        next_state = self.random.choice(self.n_states, p=self.p[self.state, action])
        reward = self.r[self.state, action]
        self.state = next_state
        return next_state, reward

    def expected_hitting_time(self, policy, origin, destination):
        '''
        Computes the expected hitting time from origin to destination given a Markov deterministic policy.
        Args:
            policy : [action] of size self.n_states where action in [0, self.n_actions).
                Stationary deterministic policy.
            origin : int.
                A state to start at.
            destination : int.
                A state to end at.
        '''

        # Markov chain transition probability
        mc_p = np.zeros((self.n_states, self.n_states))
        for st, ac in enumerate(policy):
            mc_p[st] = self.p[st, ac]
        # We keep the expected hitting time to destination y from (s, a)
        eht = np.zeros((self.n_states, self.n_actions))


    def compute_diameter(self):
        # Compute by definition. diameter D(M) := max_{x, y \in S} min_{pi : S -> A} E[time to hit y|x, pi]
        max_travel_time = 0
        # For each pair of distinct states (x -> x has travel time 0)
        for origin, destination in itertools.product(range(self.n_states), repeat=2):
            if origin == destination:
                continue
            # Iterate over all stationary deterministic policies. O(|S|^2 |A|^|S|)
            for destination in range(self.n_states):
                for pi in itertools.product(range(self.n_actions), repeat=self.n_states):
                    pi = np.asarray(pi, dtype='int')






# class MRP:
#     '''Markov reward process'''
#
#     def __init__(self, n_states, n_actions):
#         raise NotImplementedError


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
