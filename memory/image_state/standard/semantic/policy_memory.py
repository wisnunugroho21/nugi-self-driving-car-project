import numpy as np

from memory.image_state.standard.rgb.policy_memory import PolicyMemory

class PolicySemanticMemory(PolicyMemory):
    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.images[idx], dtype = np.int8), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(self.next_states[idx], dtype = np.float32), np.array(self.next_images[idx], dtype = np.int8)