from memory.aux_ppg.standard import auxPpgMemory

class auxPpgRedisListMemory(auxPpgMemory):
    def __init__(self, redis, capacity = 100000, datas = None):
        super().__init__(capacity, datas)
        self.redis = redis

    def save_redis(self):
        self.redis.append('states', self.states)

    def load_redis(self):
        self.states = self.redis.lrange('states', 0, -1)

    def delete_redis(self):
        self.redis.delete('states')

    def check_if_exists_redis(self):
        return bool(self.redis.exists('states'))