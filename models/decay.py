from math import log, exp


class Decay:
    def __init__(
        self,
        start: float,
        end: float,
        episodes: int,
        proportion_to_decay_over: float = 0.9,
    ):
        """A decay object decays a parameter from start to end over the given 
        proportion of episodes exponentially with base e

        Parameters
        ----------
        start : float
            The starting value for the parameter (episode 0)
        end : float
            The lowest value the parameter will take
        episodes : int
            The total number of episodes that training takes place over
        proportion_to_decay_over : float, optional
            The proportion of episodes that the parameter should be decayed over
            eg 0.9 means decay to end value over first 90% of training, by default 0.9
        """
        self.b = log(end / start) / (proportion_to_decay_over * episodes)
        self.episode = 0
        self.end = end

    def decay(self):
        """Gets called at the end of an episode to decay the value by one step
        """
        self.episode += 1

    def get_current_value(self) -> float:
        """Returns the current decayed value of the parameter
        """
        return max(exp(self.b * self.episode), self.end)

