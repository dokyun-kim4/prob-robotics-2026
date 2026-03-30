class Transition(object):
    def __init__(self):
        pass

    def compute_transition(self, state, action, next_state):
        """
        Compute P(next_state | state, action)
        """
    
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
        # match action:
        #     case "stay":
        #         if next_state == state:
        #             return 1.0
        #         elif next_state[0] == state[0] + 1 and next_state[1] == state[1] or \
        #             next_state[0] == state[0] -1 and next_state[1] == state[1] or \
        #             next_state[0] == state[0] and next_state[1] == state[1] + 1 or \
        #             next_state[0] == state[0] and next_state[1] == state[1] - 1:
        #             return 0.0
        #         else:
        #             return 0.0

        #     case "right":
        #         if next_state[0] == state[0] + 1 and next_state[1] == state[1]:
        #             return 1.0
        #         elif next_state[0] == state[0] -1 and next_state[1] == state[1]:
        #             return 0.0
        #         else:
        #             return 0.0
            
        #     case "left":
        #         if next_state[0] == state[0] - 1 and next_state[1] == state[1]:
        #             return 1.0
        #         elif next_state[0] == state[0] + 1 and next_state[1] == state[1]:
        #             return 0.0
        #         else:
        #             return 0.0

        #     case "up":
        #         if next_state[0] == state[0] and next_state[1] == state[1] + 1:
        #             return 0.1
        #         elif next_state[0] == state[0] and next_state[1] == state[1] - 1:
        #             return 0.9
        #         else:
        #             return 0.0

        #     case "down":
        #         if next_state[0] == state[0] and next_state[1] == state[1] - 1:
        #             return 1.0
        #         elif next_state[0] == state[0] and next_state[1] == state[1] + 1:
        #             return 0.0
        #         else:
        #             return 0.0

        #     case _:
        #         return 0.0
