import time
import datetime

import ray

class SyncExecutor():
    def __init__(self, agent, env, n_iteration, runner, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 1024, n_aux_update = 10, 
        n_saved = 10, max_action = 1.0, load_weights = False):

        self.agent              = agent
        self.env                = env
        self.runner             = runner
        
        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.reward_threshold   = reward_threshold
        self.n_plot_batch       = n_plot_batch
        self.max_action         = max_action
        self.n_aux_update       = n_aux_update
        self.load_weights       = load_weights

        self.t_updates          = 0
        self.t_aux_updates      = 0
        
    def execute(self):
        if self.load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

        start = time.time()
        print('Running the training!!')
        
        try:            
            for i_iteration in range(self.n_iteration):
                self.agent.save_temp_weights()
                futures  = [runner.run.remote() for runner in self.runner]
                memories = ray.get(futures)

                for memory in memories:
                    self.agent.save_memory(memory)

                self.agent.update()   

                if self.save_weights:
                    if i_iteration % self.n_saved == 0:
                        self.agent.save_weights() 
                        print('weights saved')

        finally:
            ray.shutdown()
            
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))