import math
import copy
import random
import time

class Node():
    def __init__(self, s):
        self.parent = None
        self.child = [None, None, None, None]
        self.N = 0
        self.Q = 0
        self.s = s
        self.c = 100

    def UCT(self, action):
        if self.child[action] == None:
            return math.inf
        return self.child[action].Q + self.c * math.sqrt(math.log(self.N) / self.child[action].N)

class MCTS():
    def __init__(self, s, env):
        self.root = Node(s)
        self.env = copy.deepcopy(env)
        self.depth = 10
    
    def select(self, node):
        ret = 0
        _max = -math.inf
        for action in range(4):
            value = node.UCT(action)
            if value > _max:
                _max = value
                ret = action
        return ret
    
    def expand(self):
        test_env = copy.deepcopy(self.env)
        node = self.root
        while True:
            action = self.select(node)
            test_env.step(action)
            if node.child[action] != None:
                node = node.child[action]
            else:
                child_node = Node(test_env.state)
                child_node.parent = node
                node.child[action] = child_node
                return child_node, test_env

    def rollout(self, env):
        reward = env.score
        for i in range(self.depth):
            obs, rew, done, info = env.step(random.randint(0, 3))
            if done:
                break
            reward = env.score
        return reward
    
    def back(self, node, value):
        while node != None:
            node.N += 1
            node.Q += (value - node.Q) / node.N
            node = node.parent

class Agent():
    def __init__(self, time):
        self.limit = time
    
    def act(self, state, env):
        agent = MCTS(state, env)
        agent.root = Node(state)
        agent.env = copy.deepcopy(env)
        start_time = time.time()
        while time.time() - start_time < self.limit:
            node, new_env = agent.expand()
            value = agent.rollout(new_env)
            agent.back(node, value)
        action = 0
        _max = -math.inf
        for a in range(4):
            if agent.root.child[a] == None:
                continue
            value = agent.root.child[a].Q
            if value > _max:
                _max = value
                action = a
        agent.env.step(action)
        agent.root = agent.root.child[action]
        agent.root.parent = None
        return action
