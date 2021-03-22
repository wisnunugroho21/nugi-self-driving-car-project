import datetime
import time
import ray

class CentralLearnerExecutor():
    def __init__(self, agent, n_iteration, redis, memory, save_weights = False, n_saved = 10):
        self.agent  = agent
        self.redis  = redis
        self.memory = memory

        self.n_iteration    = n_iteration
        self.save_weights   = save_weights
        self.n_saved        = n_saved

    def execute(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                if self.memory.is_exists_redis(self.redis):
                    self.memory.load_redis()
                    self.memory.delete_redis()

                self.agent.save_memory(self.memory)
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