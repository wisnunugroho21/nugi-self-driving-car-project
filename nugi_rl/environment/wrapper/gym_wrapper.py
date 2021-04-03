import gym

class GymWrapper():
    def __init__(self, env):
        self.env = env        

    def is_discrete(self):
        return type(self.env.action_space) is not gym.spaces.Box

    def get_obs_dim(self):
        if type(self.env.observation_space) is gym.spaces.Box:
            state_dim = 1

            if len(self.env.observation_space.shape) > 1:                
                for i in range(len(self.env.observation_space.shape)):
                    state_dim *= self.env.observation_space.shape[i]            
            else:
                state_dim = self.env.observation_space.shape[0]

            return state_dim
        else:
            return self.env.observation_space.n
            
    def get_action_dim(self):
        if self.is_discrete():
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()