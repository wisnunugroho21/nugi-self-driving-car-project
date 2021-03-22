import datetime
import time

class Executor():
    def __init__(self, agent, n_iteration, runner, save_weights = False, n_saved = 10, load_weights = False, is_training_mode = True):
        self.agent              = agent
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.is_training_mode   = is_training_mode

        if load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

    def execute(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                memories  = self.runner.run()
                self.agent.save_memory(memories)

                if self.is_training_mode:
                    self.agent.update()

                    if self.save_weights:
                        if i_iteration % self.n_saved == 0:
                            self.agent.save_weights()
                            print('weights saved')

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))