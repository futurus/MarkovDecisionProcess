__author__ = 'vunguyen'

from utils import *
from collections import *
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
    def __init__(self, grid, start=(2, 2), gamma=0.9):
        self.start = start
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.reward = {}
        self.states = set()
        self.gamma = gamma
        self.stores = set()
        self.bank = set()
        self.friend = set()

        for x in range(self.cols):
            for y in range(self.rows):
                # self.reward[(x, y)] = grid[(self.rows - 1) - y][x]
                self.reward[(x, y)] = -0.1  # unnecessary

                if grid[(self.rows - 1) - y][x] is not Wall:
                    self.states.add((x, y))

                if grid[(self.rows - 1) - y][x] is 'S':
                    self.stores.add((x, y))
                elif grid[(self.rows - 1) - y][x] is 'B':
                    self.bank.add((x, y))
                elif grid[(self.rows - 1) - y][x] is 'F':
                    self.friend.add((x, y))
                    self.reward[(x, y)] = 5.0

    def go(self, state, action, status):
        # roll the dice
        rand = random.random()
        T = self.transition(state, action, status)
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

    def transition(self, state, action, status):  # status 0: no money no gift, 1: money no gift, 2 no money gift, 3 money gift
        if status > 1:
            if state in self.stores:
                return [(0.30, (0, 0)),
                        (0.60, action),
                        (0.05, left(action)),
                        (0.05, right(action))]
            else:
                return [(0.10, (0, 0)),
                        (0.80, action),
                        (0.05, left(action)),
                        (0.05, right(action))]
        else:
            return [(0.90, action),
                    (0.05, left(action)),
                    (0.05, right(action))]

    def f(self, u, n, n_arg=25):
        if n < n_arg:
            return 1000
        else:
            return u

    def alpha(self, t):
        return 5000.0/(4999.0 + t)

    def actions(self):
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def getMapping(self, Q, status):
        out = dict([((s, a), 0) for s in self.states for a in self.actions()])

        for s, a, st in Q.keys():
            if st == status:
                out[(s, a)] = Q[(s, a, st)]

        return out

    def to_grid(self, mapping):
        return list(reversed([[mapping.get((x,y), 'W') for x in range(self.cols)] for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))


gworld = GridWorld([[Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall],
                    [Wall,  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ', Wall],
                    [Wall,  ' ',  'S',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  'S',  ' ', Wall],
                    [Wall,  ' ',  ' ',  ' ', Wall,  ' ',  ' ',  ' ',  ' ',  ' ', Wall,  ' ',  ' ',  ' ', Wall],
                    [Wall, Wall, Wall, Wall, Wall,  ' ',  ' ',  ' ',  ' ',  ' ', Wall, Wall, Wall, Wall, Wall],
                    [Wall,  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ', Wall,  ' ',  ' ',  ' ', Wall],
                    [Wall,  ' ',  'B',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ',  'F',  ' ', Wall],
                    [Wall,  ' ',  ' ',  ' ',  ' ',  'S',  'S',  'S',  ' ',  ' ',  ' ',  ' ',  ' ',  ' ', Wall],
                    [Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall, Wall]],
                    gamma=0.99)


def qlearning(mdp, epoch=1000, epsilon=0.01):
    Q = dict([((s, a, status), 0) for s in mdp.states for a in mdp.actions() for status in range(4)])
    N = dict([((s, a, status), 0) for s in mdp.states for a in mdp.actions() for status in range(4)])
    Start = dict([(s, 0) for s in mdp.states])
    R, f, alpha, gamma = mdp.reward, mdp.f, mdp.alpha, mdp.gamma

    t = 0
    for episode in range(epoch):
        s = random.choice(tuple(mdp.states))
        status = random.choice(range(4))
        newstatus = status
        # s = mdp.start
        Start[s] += 1

        while True:
            # time.sleep(0.25)
            t += 1
            domain = [(s, a, status) for a in mdp.actions()]
            random.shuffle(domain)  # this is simply for tie-breaking
            a = argmax(domain, lambda el:f(Q[el], N[el]))[1] # check this

            N[(s, a, status)] += 1
            sp = mdp.go(s, a, status)

            reward = -0.1

            # change status here
            # print "i'm currently at", s, "my stat:", status, "my action is", a, "new state:", sp,
            if (status == 2 or status == 3) and sp in mdp.friend:
                # print "dropping gift off",
                newstatus = status - 2  # drop gift off at friend
                reward = 5.

            elif (status == 0 or status == 2) and sp in mdp.bank:  # we prioritize giving gift to friend than going to the bank
                # print "withdrawing money",
                newstatus = status + 1  # withdraw money from the bank
                reward = 1.

            elif status == 1 and sp in mdp.stores:
                # print "buying gift",
                newstatus = status + 1  # buy gift at one of the stores
                reward = 1.

            # print "getting reward", reward

            Q[(s, a, status)] = (1 - alpha(t)) * Q[(s, a, status)] + alpha(t) * (reward + gamma * max([Q[sp, ap, newstatus] for ap in mdp.actions()]))

            s = sp
            status = newstatus

            if t % 1000 == 0:
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
Q, N, S, R = qlearning(gworld, epoch=1000)
Q0 = gworld.getMapping(Q, 0)
Q1 = gworld.getMapping(Q, 1)
Q2 = gworld.getMapping(Q, 2)
Q3 = gworld.getMapping(Q, 3)
# print estimated_utility(gworld, Q)

# print S

print
print_table(gworld.to_arrows(best_policy(gworld, Q0)))
print
print_table(gworld.to_arrows(best_policy(gworld, Q1)))
print
print_table(gworld.to_arrows(best_policy(gworld, Q2)))
print
print_table(gworld.to_arrows(best_policy(gworld, Q3)))
toc()
