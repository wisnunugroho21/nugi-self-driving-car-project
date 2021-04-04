import gym

class VectorEnv():
    def __init__(self, envs):
        self.envs = envs

    def is_discrete(self):
        return type(self.envs[0].action_space) is not gym.spaces.Box

    def get_obs_dim(self):
        if type(self.envs[0].observation_space) is gym.spaces.Box:
            state_dim = 1

            if len(self.envs[0].observation_space.shape) > 1:                
                for i in range(len(self.envs[0].observation_space.shape)):
                    state_dim *= self.envs[0].observation_space.shape[i]            
            else:
                state_dim = self.envs[0].observation_space.shape[0]

            return state_dim
                    
        else:
            return self.envs[0].observation_space.n
            
    def get_action_dim(self):
        if self.is_discrete():
            return self.envs[0].action_space.n
        else:
            return self.envs[0].action_space.shape[0]

    # Call this only once at the beginning of training (optional):
    def seed(self, seeds):
        assert len(self.envs) == len(seeds)
        return tuple(env.seed(s) for env, s in zip(self.envs, seeds))

    # Call this only once at the beginning of training:
    def reset(self):
        return tuple(env.reset() for env in self.envs)

    # Call this on every timestep:
    def step(self, actions):
        assert len(self.envs) == len(actions)

        return_values = []
        for env, a in zip(self.envs, actions):
            observation, reward, done, info = env.step(a)
            if done:
                observation = env.reset()
            return_values.append((observation, reward, done, info))
            
        return tuple(return_values)

    def render(self):
        for env in self.envs:
            env.render()

    # Call this at the end of training:
    def close(self):
        for env in self.envs:
            env.close()