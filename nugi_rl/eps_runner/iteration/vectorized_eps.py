import numpy as np
from eps_runner.iteration.iter_runner import IterRunner

class VectorizedRunner(IterRunner): 
    def __init__(self, agent, envs, memory, training_mode, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100):
        self.envs               = envs
        self.agent              = agent
        self.render             = render
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = [env.reset() for env in self.envs]
        self.memories           = memory

        self.states             = [env.reset() for env in envs]
        self.total_rewards      = [0 for _ in range(len(envs))]
        self.eps_times          = [0 for _ in range(len(envs))]
        self.i_episodes         = [0 for _ in range(len(envs))]

    def run(self):
        for memory in self.memories:
            memory.clear_memory()

        for _ in range(self.n_update):
            actions = self.agent.act(self.states)

            for index, (env, memory, action) in enumerate(zip(self.envs, self.memories, actions)):
                if self.is_discrete:
                    action = int(action)

                if self.max_action is not None and not self.is_discrete:
                    action_gym  =  np.clip(action, -1.0, 1.0) * self.max_action
                    next_state, reward, done, _ = env.step(action_gym)
                else:
                    next_state, reward, done, _ = env.step(action)

                if self.training_mode:
                    memory.save_eps(self.states[index].tolist(), action, reward, float(done), next_state.tolist())

                self.states[index]           = next_state
                self.total_rewards[index]    += reward
                self.eps_times[index]        += 1

                if self.render:
                    env.render()

                if done:
                    self.i_episodes[index]  += 1
                    print('Agent {} Episode {} \t t_reward: {} \t time: {} '.format(index, self.i_episodes[index], self.total_rewards[index], self.eps_times[index]))

                    if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                        self.writer.add_scalar('Rewards', self.total_rewards[index], self.i_episodes[index])
                        self.writer.add_scalar('Times', self.eps_times[index], self.i_episodes[index])

                    self.states[index]           = env.reset()
                    self.total_rewards[index]    = 0
                    self.eps_times[index]        = 0
        
        return self.memories