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
