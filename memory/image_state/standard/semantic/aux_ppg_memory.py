import numpy as np

from memory.image_state.standard.rgb.aux_ppg_memory import AuxPpgMemory

class AuxPpgSemanticMemory(AuxPpgMemory):
    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.images[idx], dtype = np.int8)