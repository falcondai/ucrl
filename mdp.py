import itertools
import numpy as np
import scipy.optimize


class MDP:
    '''Markov decision process'''

    def reset(self, init_state=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


def find_expected_hitting_times_inv(p, target):
    '''
    Solve for expected hitting times to a target state with matrix inversion
    '''
    n_states = p.shape[0]
    b = np.ones((n_states - 1, 1))
    # Remove the target state from p
    ind = list(range(n_states))
    ind.remove(target)
    pp = p[ind][:, ind]
    tau_non_targets = np.linalg.inv(np.eye(n_states - 1) - pp) @ b
    # The hitting time from target state to itself is 0
    tau = np.vstack((tau_non_targets[:target + 1], [[0]], tau_non_targets[target + 1:]))
    return tau.flatten()


def find_expected_hitting_times_lp(p, target):
    '''
    Solve for expected hitting times to a target state with linear programming (interior point method)
    '''
    n_states = p.shape[0]
    # Swap the target state to 0
    ind = list(range(n_states))
    ind.remove(target)
    ind = [target] + ind
    pp = p[ind][:, ind]
    # Set up a linear program to solve with target state being 0
    A_eq = np.zeros_like(pp)
    b_eq = np.zeros(n_states)
    # Suppose tau_i := expected hitting time to targets starting in state i
    # We want the minimum so c = [1, ..., 1]
    c = np.ones(n_states)
    # Constraint 1: the expected hitting time to 0 from 0 is 0
    A_eq[0, 0] = 1
    b_eq[0] = 0
    # Constraint 2: the expected hitting times to 0 from non-targets is tau_i = 1 + sum_j p_ij tau_j
    A_eq[1:, 1:] = np.eye(n_states - 1) - pp[1:, 1:]
    b_eq[1:] = 1
    res = scipy.optimize.linprog(c=c, A_eq=A_eq, b_eq=b_eq, method='interior-point')
    if not res.success:
        raise Exception(res.message)
    return res.x


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

    def mrp_under_markov_policy(self, policy):
        '''
        Return the Markov reward process (p, r) given a Markov deterministic policy.
        Args:
            policy : [action] of size self.n_states where action in [0, self.n_actions).
                Stationary deterministic policy.
        '''
        pp = np.zeros((self.n_states, self.n_actions))
        rr = np.zeros(self.n_states)
        for st, ac in enumerate(policy):
            pp[st] = self.p[st, ac]
            rr[st] = self.r[st, ac]
        return pp, rr

    # def expected_hitting_time(self, policy, destination):
    #     '''
    #     Computes the expected hitting times starting in all states to the given destination under a Markov deterministic policy.
    #     Args:
    #         policy : [action] of size self.n_states where action in [0, self.n_actions).
    #             Stationary deterministic policy.
    #         destination : int.
    #             A state to end at.
    #     '''
    #     # Markov chain transition probability
    #     mc_p = np.zeros((self.n_states, self.n_states))
    #     for st, ac in enumerate(policy):
    #         mc_p[st] = self.p[st, ac]
    #     # We keep the expected hitting time to destination y from (s, a)
    #     eht = np.zeros((self.n_states, self.n_actions))
    #
    #
    # def compute_diameter(self):
    #     # Compute by definition. diameter D(M) := max_{x, y \in S} min_{pi : S -> A} E[time to hit y|x, pi]
    #     max_travel_time = 0
    #     # For each pair of distinct states (x -> x has travel time 0)
    #     for origin, destination in itertools.product(range(self.n_states), repeat=2):
    #         if origin == destination:
    #             continue
    #         # Iterate over all stationary deterministic policies. O(|S|^2 |A|^|S|)
    #         for destination in range(self.n_states):
    #             for pi in itertools.product(range(self.n_actions), repeat=self.n_states):
    #                 pi = np.asarray(pi, dtype='int')


class RingMDP(SimpleMDP):
    '''
    Special class of MDPs to illustrate the effect of potential-based reward shaping
    '''

    def __init__(self, n_states, next_state_prob, initial_state_distribution=0, random_seed=None):
        # There are two actions: [`stay here`, `go to the next state`]
        n_actions = 2
        # Going to the next state has reward 0
        # Staying at all states except for state n has reward 0
        r = np.zeros((n_states, n_actions))
        # Staying at state n has reward 1
        r[-1, 0] = 1
        self.next_state_prob = next_state_prob
        p = np.zeros((n_states, n_actions, n_states))
        # `Stay here` action stays in the current state
        p[range(n_states), 0, range(n_states)] = 1
        # `Go to the next state` action
        # In state s, it stays with probability 1 - next_state_prob
        p[range(n_states), 1, range(n_states)] = 1 - self.next_state_prob
        # Goes to state s + 1 mod n_states with probability next_state_prob
        next_states = [(s + 1) % n_states for s in range(n_states)]
        p[range(n_states), 1, next_states] = self.next_state_prob

        super().__init__(n_states, 2, p, r, initial_state_distribution, random_seed)


def stochastic_matrix(n):
    '''Creates a stochastic matrix of size (n, n) from a uniform prior.'''
    p = np.zeros((n, n))
    for i in range(n):
        p[i] = np.random.dirichlet(np.ones(n))
    return p

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
    # print(mdp.reset(0))
    # for i in range(10):
    #     ac = np.random.randint(2)
    #     print(ac, mdp.step(ac))

    mdp = RingMDP(10, 0.2)
    st = mdp.reset(0)
    print(st)
    pi = [1] * 9 + [0]
    for i in range(100):
        # ac = np.random.randint(2)
        ac = pi[st]
        st, r = mdp.step(ac)
        print(ac, st, r)
