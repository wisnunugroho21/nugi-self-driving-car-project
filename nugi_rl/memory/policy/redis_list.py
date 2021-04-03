from memory.policy.standard import PolicyMemory

class PolicyRedisListMemory(PolicyMemory):
    def __init__(self, redis, capacity = 100000, datas = None):
        super().__init__(capacity, datas)
        self.redis = redis

    def save_redis(self):
        self.redis.append('states', self.states)
        self.redis.append('actions', self.actions)
        self.redis.append('rewards', self.rewards)
        self.redis.append('dones', self.dones)
        self.redis.append('next_states', self.next_states)

    def load_redis(self):
        self.states         = self.redis.lrange('states', 0, -1)
        self.actions        = self.redis.lrange('actions', 0, -1)
        self.rewards        = self.redis.lrange('rewards', 0, -1)
        self.dones          = self.redis.lrange('dones', 0, -1)
        self.next_states    = self.redis.lrange('next_states', 0, -1)

    def delete_redis(self):
        self.redis.delete('states')
        self.redis.delete('actions')
        self.redis.delete('rewards')
        self.redis.delete('dones')
        self.redis.delete('next_states')

    def check_if_exists_redis(self):
        return bool(self.redis.exists('states')) and bool(self.redis.delete('actions')) and bool(self.redis.delete('rewards')) and bool(self.redis.delete('dones')) and bool(self.redis.delete('next_states'))
        