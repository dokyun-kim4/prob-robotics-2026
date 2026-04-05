import numpy as np
import matplotlib.pyplot as plt
from reward import Reward
from transition import Transition
from copy import deepcopy
import time

class World(object):
    def __init__(self, xdim, ydim, reward, transition):
        self.reward = reward  # reward function
        self.transition = transition  # transition function
        self.world = np.zeros((xdim, ydim))
        self.all_states = [(r, c) for r in range(self.world.shape[0]) for c in range(self.world.shape[1])]

    def compute_action_probability(self, state, action, next_state):
        """Convenience function for computing transition probability."""
        return self.transition.compute_transition(state, action, next_state)
    
    def compute_reward(self, state, action, next_state):
        """Convenience function for computing immediate reward."""
        return self.reward.compute_reward(state, action, next_state)


def initialize_value(world: World):
    """Given a grid world, initialize the value function."""
    Vhat = np.zeros_like(world.world)
    return Vhat

def value_iteration(world: World, actions: list[str], discount: float, epsilon: float):

    V_hat = initialize_value(world)

    while True:
        delta = 0
        for state in world.all_states:
            v = V_hat[state]
            V_hat[state] = max(
                            sum(
                                world.compute_action_probability(state, action, next_state) * 
                                (world.compute_reward(state, action, next_state) + discount * V_hat[next_state]) 
                                for next_state in world.all_states
                                )
                            for action in actions
                            )
        delta = max(delta, abs(v - V_hat[state]))
        if delta < epsilon:
            break
    return V_hat

def compute_policy(world: World, V_hat: np.ndarray, discount: float):

    policy = {}
    for state in world.all_states:
        policy[state] = max(actions, key=lambda a: sum(world.compute_action_probability(state, a, next_state) * 
                                                       (world.compute_reward(state, a, next_state) + discount * V_hat[next_state])
                                                       for next_state in world.all_states))
    return policy


def initialize_policy(world: World, actions):
    """Given a grid world and set of actions, initialize the policy."""
    pihat = dict()
    for key in world.all_states:
        pihat[key] = np.random.choice(actions)
    return pihat


def policy_iteration(world: World, actions: list[str], discount: float):
    """Performs policy iteration for a given Gridworld."""
    Vhat = np.zeros_like(world.world)
    pihat = initialize_policy(world, actions)

    while True:
        p = deepcopy(pihat)  # current policy; will compare after refinement
        for state in world.all_states:  # update the value function
            Vhat[state] = sum(world.compute_action_probability(state, pihat[state], next_state) * 
                              (world.compute_reward(state, pihat[state], next_state) + discount * Vhat[next_state]) 
                              for next_state in world.all_states)
        for state in world.all_states:  # refine the policy
            pihat[state] = max(actions, key=lambda a: sum(world.compute_action_probability(state, a, next_state) * 
                                                       (world.compute_reward(state, a, next_state) + discount * Vhat[next_state])
                                                       for next_state in world.all_states))
            
        if p == pihat:  # policy has converged
            break
    
    for state in world.all_states:  # update the value function
            Vhat[state] = sum(world.compute_action_probability(state, pihat[state], next_state) * 
                              (world.compute_reward(state, pihat[state], next_state) + discount * Vhat[next_state]) 
                              for next_state in world.all_states)
    return Vhat, pihat

if __name__ == "__main__":
    # Problem Set-Up
    actions = ["stay","right","left","up","down"]
    xdim = 20
    ydim = 20
    discount = 0.7
    convergence_threshold = 0.0001
    payoff = 100
    payoff_loc = (3,2)
    cost = -1
    reward = Reward(payoff_loc, payoff=payoff, cost=cost)
    transition = Transition()
    world = World(xdim, ydim, reward, transition)

    start = time.perf_counter()

    # Value Iteration and Optimal Policy Computation
    Vhat = value_iteration(world, actions, discount, convergence_threshold)
    policy = compute_policy(world, Vhat, discount)

    # Vhat, policy = policy_iteration(world, actions, discount)
    print(f"Value function max: {np.max(Vhat):.3f}, Value function min: {np.min(Vhat):.3f}")

    end = time.perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")
    # Plotting
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
    plot_actions = {"stay":(0,0), "right":(0.4,0), "left":(-0.4,0), "up":(0,0.4), "down":(0,-0.4)}
    ax[0].imshow(Vhat.reshape(xdim, ydim).T, origin="lower")
    ax[0].set_title(f"Value iteration -- discount:{discount}, payoff:{payoff}, goal:{payoff_loc}")

    ax[1].imshow(Vhat.reshape(xdim, ydim).T, origin="lower")
    for key, value in policy.items():
        xval = key[0]+plot_actions[value][0]
        yval = key[1]+plot_actions[value][1]
        ax[1].arrow(key[0], key[1], plot_actions[value][0], plot_actions[value][1], width=0.1)
    ax[1].set_title("Optimal Policy")
    plt.show()
