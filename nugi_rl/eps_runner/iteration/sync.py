import numpy as np
import torch

from eps_runner.iteration.iter_runner import IterRunner
import ray

@ray.remote(num_gpus = 0.25)
class SyncRunner(IterRunner):
    def __init__(self, agent, env, memory, training_mode, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100, 
        folder = '', tag = 0):

        self.env                = env
        self.agent              = agent
        self.render             = render
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch
        self.folder             = folder
        self.tag                = tag

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = self.env.reset()
        self.memories           = memory
        self.is_discrete        = is_discrete

    def run(self):
        self.memories.clear_memory()
        self.agent.load_weights(self.folder)

        for _ in range(self.n_update):
            action = self.agent.act(self.states)

            if self.is_discrete:
                action = int(action)

            if self.max_action is not None and not self.is_discrete:
                action_gym  =  np.clip(action, -1.0, 1.0) * self.max_action
                next_state, reward, done, _ = self.env.step(action_gym)
            else:
                next_state, reward, done, _ = self.env.step(action)
            
            if self.training_mode:
                self.memories.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist())
                
            self.states         = next_state
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

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0             
        
        return self.memories, self.tag