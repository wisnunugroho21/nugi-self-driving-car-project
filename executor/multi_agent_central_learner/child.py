import ray

@ray.remote()
class ChildExecutor():
    def __init__(self, agent, runner, tag, load_weights = False):
        self.agent  = agent
        self.runner = runner
        self.tag    = tag

        if load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

    def execute(self):
        memories  = self.runner.run()
        self.agent.save_memory(memories)

        self.agent.update()
        self.agent.save_weights()

        return memories, self.tag