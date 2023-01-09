

class randAgent():
    '''Class for a random agent.'''
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space.n
        self.observation_space = env.observation_space.shape[0]
    
    
    def get_action(self, state):
        return self.env.action_space.sample()