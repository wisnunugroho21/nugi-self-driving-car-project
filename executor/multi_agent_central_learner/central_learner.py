import datetime
import time
import ray

class CentralLearnerExecutor():
    def __init__(self, agent, n_iteration, child_executors, save_weights = False, n_saved = 10):
        self.agent              = agent
        self.child_executors    = child_executors

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved

    def execute(self):
        start = time.time()
        print('Running the training!!')

        episode_ids = []
        for i, executor in enumerate(self.child_executors):
            episode_ids.append(executor.execute.remote())
            time.sleep(4)

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                ready, not_ready    = ray.wait(episode_ids)
                memories, tag       = ray.get(ready[0])

                episode_ids = not_ready
                episode_ids.append(self.child_executors[tag].execute.remote())

                self.agent.save_memory(memories)
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
            print('\nTimelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))