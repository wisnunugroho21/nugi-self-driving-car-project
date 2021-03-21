import numpy as np

class EpisodicRunner():
    def __init__(self, agent, env, memory, training_mode, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100):
        self.agent              = agent
        self.env                = env
        self.memories           = memory

        self.render             = render
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch
        self.is_discrete        = is_discrete

        self.i_episode          = 0

    def run(self):
        self.memories.clear_memory()
        state = self.env.reset() 

        done            = False
        total_reward    = 0
        eps_time        = 0
        ############################################ 
        while not done:
            action = self.agent.act(state) 

            if self.is_discrete:
                action = int(action)

            if self.max_action is not None and not self.is_discrete:
                action_gym  =  np.clip(action, -1.0, 1.0) * self.max_action
                next_state, reward, done, _ = self.env.step(action_gym)
            else:
                next_state, reward, done, _ = self.env.step(action)
            
            if self.training_mode:
                self.memories.save_eps(state.tolist(), action, reward, float(done), next_state.tolist())
                
            state = next_state
            eps_time       += 1 
            total_reward   += reward
                    
            if self.render:
                self.env.render()
                    
        self.i_episode  += 1
        print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, total_reward, eps_time))

        if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
            self.writer.add_scalar('Rewards', total_reward, self.i_episode)
            self.writer.add_scalar('Times', eps_time, self.i_episode)
        
        return self.memories