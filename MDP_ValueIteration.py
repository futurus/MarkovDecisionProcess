__author__ = 'vunguyen'

from collections import *
from utils import *
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
        state1 = (state[0] + action[0], state[1] + action[1])
        if state1 in self.states:
            return state1
        else:
            return state

    def transition(self, state, action):
        return [(0.8, self.go(state, action)),
                (0.1, self.go(state, left(action))),
                (0.1, self.go(state, right(action)))]

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

def value_iteration(mdp, epsilon=0.001):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.reward, mdp.transition, mdp.gamma
    count = 0
    while True:
        count += 1
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R[s] + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)]) for a in mdp.actions()])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon*(1 - gamma)/gamma:
            print count
            return U


def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.transition(s, a)])


def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(), lambda a:expected_utility(a, s, U, mdp))
    return pi

tic()
U = value_iteration(gworld)
print U
print_table(gworld.to_arrows(best_policy(gworld, U)))
toc()

# print gworld.to_arrows(best_policy(gworld, value_iteration(gworld)))

# print gworld.reward
# print gworld.states
# print gworld.transition((0, 0), (1, 0))
# print gworld.transition((0, 0), (0, 1))
# print gworld.transition((1, 0), (0, 1))