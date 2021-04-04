import numpy as np
from utils.math_function import prepro_half_one_dim
from eps_runner.iteration.iter_runner import IterRunner

class PongRunner(IterRunner):
    def __init__(self, agent, env, memory, training_mode, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100):
        super().__init__(agent, env, memory, training_mode, render, n_update, is_discrete, max_action, writer, n_plot_batch)

        obs         = self.env.reset()  
        self.obs    = prepro_half_one_dim(obs)
        self.states = self.obs

    def run(self, agent):
        self.memories.clear_memory()

        for _ in range(self.n_update):
            action      = agent.act(self.states)
            action_gym  = action + 1 if action != 0 else 0

            next_obs, reward, done, _ = self.env.step(action_gym)
            next_obs    = prepro_half_one_dim(next_obs)
            next_state  = next_obs - self.obs
            
            if self.training_mode:
                self.memories.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist())
                
            self.states         = next_state
            self.obs            = next_obs
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                obs         = self.env.reset()  
                self.obs    = prepro_half_one_dim(obs)
                self.states = self.obs

                self.total_reward   = 0
                self.eps_time       = 0             
        
        return self.memories