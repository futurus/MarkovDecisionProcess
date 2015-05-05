__author__ = 'vunguyen'

from utils import *
from collections import *
import random
import time
import math

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
    def __init__(self, grid, start=(2, 2), gamma=0.9):
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

        prob = 0
        for i in range(len(T)):
            prob += T[i][0]
            if rand < prob:
                action = T[i][1]
                break

        state1 = (state[0] + action[0], state[1] + action[1])
        if state1 in self.states:
            return state1
        else:
            return state

    def transition(self, action):
        return [(0.8, action),
                (0.1, left(action)),
                (0.1, right(action))]

    def f(self, u, n, n_arg=1000):
        if n < n_arg:
            return 100
        else:
            return u

    def alpha(self, t):
        return 11000.0/(10999.0 + t)

    def actions(self):
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def to_grid(self, mapping):
        return list(reversed([[mapping.get((x,y), 0.0) for x in range(self.cols)] for y in range(self.rows)]))

    def print_grid(self, mapping):
        mapping = self.to_grid(mapping)
        for i in range(self.rows):
            for j in range(self.cols):
                print repr(round(mapping[i][j], 2)).rjust(5),
            print

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))


gworld = GridWorld([[ +1.00,  Wall, +1.00, -0.04, -0.04, +1.00],
                    [ -0.04, -1.00, -0.04, +1.00,  Wall, -1.00],
                    [ -0.04, -0.04, -1.00, -0.04, +1.00, -0.04],
                    [ -0.04, -0.04, -0.04, -1.00, -0.04, +1.00],
                    [ -0.04,  Wall,  Wall,  Wall, -1.00, -0.04],
                    [ -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]], gamma=0.99)


def RMSE(mdp, U, Uv):
    error = 0

    for s in mdp.states:
        error += (U[s] - Uv[s])**2

    return math.sqrt(error/len(mdp.states))


def qlearning(mdp, epoch=1000, epsilon=0.01):
    Q = dict([((s, a), 0) for s in mdp.states for a in mdp.actions()])
    N = dict([((s, a), 0) for s in mdp.states for a in mdp.actions()])
    Start = dict([(s, 0) for s in mdp.states])
    R, f, alpha, gamma = mdp.reward, mdp.f, mdp.alpha, mdp.gamma

    Uv = {(0, 0): 92.94, (0, 1): 94.31, (0, 2): 95.55, (0, 3): 96.95, (0, 4): 98.39, (0, 5): 100.0,
          (1, 0): 91.73, (1, 1): 00.00, (1, 2): 94.45, (1, 3): 95.59, (1, 4): 95.88, (1, 5): 00.00,
          (2, 0): 90.53, (2, 1): 00.00, (2, 2): 93.23, (2, 3): 93.29, (2, 4): 94.54, (2, 5): 95.04,
          (3, 0): 89.36, (3, 1): 00.00, (3, 2): 91.11, (3, 3): 93.18, (3, 4): 94.40, (3, 5): 93.87,
          (4, 0): 88.57, (4, 1): 89.55, (4, 2): 91.81, (4, 3): 93.10, (4, 4): 00.00, (4, 5): 92.65,
          (5, 0): 89.30, (5, 1): 90.57, (5, 2): 91.89, (5, 3): 91.79, (5, 4): 90.92, (5, 5): 93.33}

    # mystates = set([(0, 5), (0, 3), (2, 5), (5, 4), (4, 1)])
    # for s in mystates:
    #     print s, ",",
    # print "RMSE"

    t = 0
    for episode in range(epoch):
        s = random.choice(tuple(mdp.states))
        # s = mdp.start
        Start[s] += 1

        # if episode % 250 == 0:
        #     U = estimated_utility(mdp, Q)
        #     for s in mystates:
        #         print U[s], ",",
        #     print RMSE(mdp, U, Uv)

        while True:
            # time.sleep(1)
            t += 1
            domain = [(s, a) for a in mdp.actions()]
            random.shuffle(domain)  # this is simply for tie-breaking

            a = argmax(domain, lambda el:f(Q[el], N[el]))[1]

            N[(s, a)] += 1
            sp = mdp.go(s, a)

            Q[(s, a)] = (1 - alpha(t)) * Q[(s, a)] + alpha(t) * (R[s] + gamma * max([Q[sp, ap] for ap in mdp.actions()]))

            s = sp

            # if alpha(t) < epsilon * (1 - gamma)/gamma:
            if t % 100 == 0:
                break

    return Q, N, Start, R

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
Q, N, S, R = qlearning(gworld, epoch=10000)
gworld.print_grid(estimated_utility(gworld, Q))
print_table(gworld.to_arrows(best_policy(gworld, Q)))

print N
toc()