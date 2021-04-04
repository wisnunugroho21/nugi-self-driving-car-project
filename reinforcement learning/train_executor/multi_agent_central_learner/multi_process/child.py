class ChildExecutor():
    def __init__(self, agent, runner, tag, redis, load_weights = False, save_weights = False):
        self.agent  = agent
        self.runner = runner
        self.tag    = tag
        self.redis  = redis

        self.save_weights   = save_weights

        if load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

    def execute(self):
        try:
            memories  = self.runner.run()
            memories.save_redis(self.redis)

            self.agent.save_memory(memories)
            self.agent.update()

            if self.save_weights:
                self.agent.save_weights()
                print('weights saved')
        except KeyboardInterrupt:
            print('Stopped by User')