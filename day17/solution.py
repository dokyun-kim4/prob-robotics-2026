import numpy as np
import matplotlib.pyplot as plt
import copy
import time

class Reward(object):
    '''Creates a reward computing object for a gridworld.'''
    def __init__(self, goal, payoff, cost):
        self.goal = goal  # the location of the payoff
        self.payoff = payoff  # the payoff amount
        self.cost = cost  # action costs

    def compute_reward(self, state, action, next_state):
        '''Compute the immediate reward that is accumulated based on the current state, action taken, and resulting state.'''
        if (state == self.goal and action == "stay"):
            return self.payoff
        elif next_state == self.goal and action == "right" and state == (self.goal[0]-1, self.goal[1]):
            return self.payoff
        elif next_state == self.goal and action == "left" and state == (self.goal[0]+1, self.goal[1]):
            return self.payoff
        elif next_state == self.goal and action == "up" and state == (self.goal[0], self.goal[1]-1):
            return self.payoff
        elif next_state == self.goal and action == "down" and state == (self.goal[0], self.goal[1]+1):
            return self.payoff
        else:
            return self.cost

class Transition(object):
    '''Create a transition function object for a gridworld.'''
    def __init__(self):
        pass

    def compute_transition(self, state, action, next_state):
        '''For a given state and action, compute the probability of the next state.'''
        match action:
            case "stay":
                if next_state == state:
                    return 0.8
                elif next_state[0] == state[0] + 1 and next_state[1] == state[1] or \
                    next_state[0] == state[0] -1 and next_state[1] == state[1] or \
                    next_state[0] == state[0] and next_state[1] == state[1] + 1 or \
                    next_state[0] == state[0] and next_state[1] == state[1] - 1:
                    return 0.05
                else:
                    return 0.0

            case "right":
                if next_state[0] == state[0] + 1 and next_state[1] == state[1]:
                    return 0.9
                elif next_state[0] == state[0] -1 and next_state[1] == state[1]:
                    return 0.1
                else:
                    return 0.0
            
            case "left":
                if next_state[0] == state[0] - 1 and next_state[1] == state[1]:
                    return 0.6
                elif next_state[0] == state[0] + 1 and next_state[1] == state[1]:
                    return 0.4
                else:
                    return 0.0

            case "up":
                if next_state[0] == state[0] and next_state[1] == state[1] + 1:
                    return 0.5
                elif next_state[0] == state[0] and next_state[1] == state[1] - 1:
                    return 0.5
                else:
                    return 0.0

            case "down":
                if next_state[0] == state[0] and next_state[1] == state[1] - 1:
                    return 0.8
                elif next_state[0] == state[0] and next_state[1] == state[1] + 1:
                    return 0.2
                else:
                    return 0.0

            case _:
                return 0.0

class GridWorld(object):
    def __init__(self, xdim, ydim, reward, transition):
        self.xdim = xdim  # number of columns in the world
        self.ydim = ydim  # number of rows in the world
        self.reward = reward  # reward function
        self.transition = transition  # transition function

        self._make_grid()  # populate the world
    
    def _make_grid(self):
        """Create the states for the grid world"""
        self.grid = dict()
        grid = []
        for i in range(self.xdim):
            for j in range(self.ydim):
                grid.append((i, j))
        for i, coord in enumerate(grid):
            self.grid[i] = coord

    def compute_action_probability(self, state, action, next_state):
        """Convenience function for computing transition probability."""
        return self.transition.compute_transition(state, action, next_state)
    
    def compute_reward(self, state, action, next_state):
        """Convenience function for computing immediate reward."""
        return self.reward.compute_reward(state, action, next_state)

def initialize_value(grid_world):
    """Given a grid world, initialize the value function."""
    Vhat = np.zeros(len(grid_world.grid))
    return Vhat

def value_iteration(grid_world, actions, discount, epsilon):
    """Perform value iteration.
    Inputs:
        grid_world: a GridWorld class
        actions: a list of possible actions a robot can take
        discount: the discount factor to apply to rewards
        episilon: the convergence threshold
    Outputs:
        Vhat: the converged value function over all world states
    """
    Vhat = initialize_value(grid_world)
    while True:
        delta = 0
        for state in grid_world.grid.keys():
            v = Vhat[state]  # store the current value for checking convergence
            Vhat[state] = max(sum(grid_world.compute_action_probability(grid_world.grid[state], action, grid_world.grid[next_state]) * 
                                (grid_world.compute_reward(grid_world.grid[state], action, grid_world.grid[next_state]) + discount * Vhat[next_state]) 
                                for next_state in grid_world.grid.keys()) for action in actions)
            delta = max(delta, abs(v - Vhat[state]))
        if delta < epsilon:
            break
    return Vhat  

def policy_selection(grid_world, Vhat, discount):
    """Compute the optimal policy.
    Inputs:
        grid_world: A GridWorld class
        Vhat: a converged value distribution from value iteration
        discount: the discount factor applied to rewards
    Outputs:
        policy: a lookup table (dictionary) of best actions for every state in grid_world
    """
    policy = {}
    for state in grid_world.grid.keys():
        policy[state] = max(actions, key=lambda a: sum(grid_world.compute_action_probability(grid_world.grid[state], a, grid_world.grid[next_state]) * 
                                                       (grid_world.compute_reward(grid_world.grid[state], a, grid_world.grid[next_state]) + discount * Vhat[next_state])
                                                       for next_state in grid_world.grid.keys()))
    return policy

def initialize_policy(grid_world, actions):
    """Given a grid world and set of actions, initialize the policy."""
    pihat = dict()
    for key in grid_world.grid.keys():
        pihat[key] = np.random.choice(actions)
    return pihat

def policy_iteration(grid_world, actions, discount):
    """Performs policy iteration for a given Gridworld."""
    Vhat = initialize_value(grid_world)
    pihat = initialize_policy(grid_world, actions)

    while True:
        p = copy.deepcopy(pihat)  # current policy; will compare after refinement
        for state in grid_world.grid.keys():  # update the value function
            Vhat[state] = sum(grid_world.compute_action_probability(grid_world.grid[state], pihat[state], grid_world.grid[next_state]) * 
                              (grid_world.compute_reward(grid_world.grid[state], pihat[state], grid_world.grid[next_state]) + discount * Vhat[next_state]) 
                              for next_state in grid_world.grid.keys())
        for state in grid_world.grid.keys():  # refine the policy
            pihat[state] = max(actions, key=lambda a: sum(grid_world.compute_action_probability(grid_world.grid[state], a, grid_world.grid[next_state]) * 
                                                       (grid_world.compute_reward(grid_world.grid[state], a, grid_world.grid[next_state]) + discount * Vhat[next_state])
                                                       for next_state in grid_world.grid.keys()))
            
        if p == pihat:  # policy has converged
            break

    return Vhat, pihat

if __name__ == "__main__":
    # Problem Set-Up
    actions = ["stay","right","left","up","down"]
    xdim = 15
    ydim = 15
    discount = 0.91
    convergence_threshold = 0.001
    payoff = 100
    payoff_loc = (3,2)
    cost = -1
    reward = Reward(payoff_loc, payoff=payoff, cost=cost)
    transition = Transition()
    world = GridWorld(xdim, ydim, reward, transition)

    start = time.perf_counter()
    # Value Iteration and Optimal Policy Computation
    # Vhat = value_iteration(world, actions, discount, convergence_threshold)
    # policy = policy_selection(world, Vhat, discount)

    Vhat, policy = policy_iteration(world, actions, discount)

    end = time.perf_counter()
    print(f"Value function max: {np.max(Vhat):.3f}, Value function min: {np.min(Vhat):.3f}")
    print(f"Time taken: {end - start:.4f} seconds")

    # Plotting
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
    plot_actions = {"stay":(0,0), "right":(0.4,0), "left":(-0.4,0), "up":(0,0.4), "down":(0,-0.4)}
    ax[0].imshow(Vhat.reshape(xdim, ydim).T, origin="lower")
    ax[0].set_title(f"Value Iteration -- discount:{discount}, payoff:{payoff}, goal:{payoff_loc}")

    ax[1].imshow(Vhat.reshape(xdim, ydim).T, origin="lower")
    for key, value in policy.items():
        xval = world.grid[key][0]+plot_actions[value][0]
        yval = world.grid[key][1]+plot_actions[value][1]
        ax[1].arrow(world.grid[key][0], world.grid[key][1], plot_actions[value][0], plot_actions[value][1], width=0.1)
    ax[1].set_title("Optimal Policy")
    plt.show()
