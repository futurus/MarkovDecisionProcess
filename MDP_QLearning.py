__author__ = 'vunguyen'

from collections import *
from utils import *
import random
import time

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

Wall = None

class GridWorld():
    def __init__(self, grid, start=(2, 3), gamma=0.9):
        self.start = start
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.reward = {}
        self.states = set()
        self.gamma = gamma

        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[(x, y)] = grid[(self.rows - 1) - y][x]
                if grid[(self.rows - 1) - y][x] is not Wall:
                    self.states.add((x, y))

    def go(self, state, action):
        # roll the dice
        rand = random.random()
        T = self.transition(action)
        if rand < T[0][0]:
            action = T[0][1]
        elif rand < T[0][0] + T[1][0]:
            action = T[1][1]
        elif rand < T[0][0] + T[1][0] + T[2][0]:
            action = T[2][1]

        state1 = (state[0] + action[0], state[1] + action[1])
        if state1 in self.states:
            return state1
        else:
            return state

    def transition(self, action):
        return [(0.8, action),
                (0.1, left(action)),
                (0.1, right(action))]

    def f(self, u, n, n_arg=100):
        if n < n_arg:
            return 500
        else:
            return u

    def alpha(self, t):
        return 60.0/(59.0 + t)

    def actions(self):
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def to_grid(self, mapping):
        return list(reversed([[mapping.get((x,y), 'Wall') for x in range(self.cols)] for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))


gworld = GridWorld([[ +1.00,  Wall, +1.00, -0.04, -0.04, +1.00],
                    [ -0.04, -1.00, -0.04, +1.00,  Wall, -1.00],
                    [ -0.04, -0.04, -1.00, -0.04, +1.00, -0.04],
                    [ -0.04, -0.04, -0.04, -1.00, -0.04, +1.00],
                    [ -0.04,  Wall,  Wall,  Wall, -1.00, -0.04],
                    [ -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]], gamma=0.99)


def qlearning(mdp, epoch=1000, epsilon=0.01):
    Q = dict([((s, a), 0) for s in mdp.states for a in mdp.actions()])
    N = dict([((s, a), 0) for s in mdp.states for a in mdp.actions()])
    R, f, alpha, gamma = mdp.reward, mdp.f, mdp.alpha, mdp.gamma

    t = 0
    for episode in range(epoch):
        s = random.sample(mdp.states, 1)[0]

        while True:
            t += 1
            domain = [(s, a) for a in mdp.actions()]
            a = argmax(domain, lambda el:f(Q[el], N[el]))[1]

            N[(s, a)] += 1
            sp = mdp.go(s, a)

            Q[(s, a)] = Q[(s, a)] + alpha(t) * (R[s] + gamma * max([Q[sp, ap] for ap in mdp.actions()]) - Q[(s, a)])
            s = sp

            if alpha(t) < epsilon * (1 - gamma)/gamma:
                break

    return Q

def estimated_utility(mdp, Q):
    U = {}

    for s in mdp.states:
        U[s] = max([Q[(sp, a)] for a in mdp.actions() for sp in mdp.states if sp == s])
    return U  # return max_a Q(s, a) for each state s

def best_policy(mdp, Q):
    pi = {}
    for s in mdp.states:
        domain = [(s, a) for a in mdp.actions()]
        max = -1000000000
        for state_action in domain:
            if Q[state_action] > max:
                max = Q[state_action]
                best_a = state_action[1]

        pi[s] = best_a
    return pi


tic()
Q = qlearning(gworld, epoch=10000)
print estimated_utility(gworld, Q)
print_table(gworld.to_arrows(best_policy(gworld, Q)))
toc()