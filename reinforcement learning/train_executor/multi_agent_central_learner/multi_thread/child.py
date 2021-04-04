import ray

@ray.remote(num_gpus=0.25)
class ChildExecutor():
    def __init__(self, agent, runner, tag, load_weights = False, save_weights = False):
        self.agent  = agent
        self.runner = runner
        self.tag    = tag

        self.save_weights   = save_weights

        if load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

    def execute(self):
        memories  = self.runner.run()
        self.agent.save_memory(memories)

        self.agent.update()

        if self.save_weights:
            self.agent.save_weights()
            print('weights saved')

        return memories, self.tag