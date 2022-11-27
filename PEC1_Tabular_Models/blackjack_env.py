import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    """
    Compares two numbers, returns 0 if both equal, 1 if a>b, -1 if a<b
    Args:
        a: number a
        b: sumber b
    """
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    """
    Picks up a random card from deck
    Args:
        np_random: Seed to use
    """
    return np_random.choice(deck)


def draw_hand(np_random):
    """
    Draws a hand, consisting of two random cards from deck
    Args:
        np_random: Seed to use
    """
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):
    """
    Performs an action over the current state
    Args:
        hand: Action to perform
    """
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    """
    Calculates the sum of all cards in the hand
    Args:
        hand: Cards in hand
    """
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    """
    Calculates if hand sums up more than 21
    Args:
        hand: Cards in hand
    """
    return sum_hand(hand) > 21


def score(hand): 
    """
    Performs an action over the current state
    Args:
        hand: Cards in hand
    """
    return 0 if is_bust(hand) else sum_hand(hand)


class BlackjackEnv(gym.Env):
    """
    ### Description:
    This environment is a blackjack card game problem.
    
    ### Action Space
    There are two discrete actions available to the player: request a new card (hit) or to stick (stick).
    So the action space is: A = {hit,stick}
    
    ### Observation Space
    The observation space consists of:
        - The player's points
        - The card the dealer has
        - Whether the player has a usable ace
    
    ### Rewards
    Rewards are: +1 for winning, -1 for losing and 0 for draw.
    """
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()
        self._reset()

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _seed(self, seed=None):
        """
        Seeds the random generator with given seed
        Args:
            seed: seed to use
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """
        Performs an action over the current state
        Args:
            action: Action to perform
        """
        assert self.action_space.contains(action), "Fallo, Action = {}".format(action)
        if action:  # Descripción
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # Descripción
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Returns the observation of the environment
        """
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def _reset(self):
        """
        Resets the environment
        """
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return self._get_obs()