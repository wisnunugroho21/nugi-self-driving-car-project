import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from helper.pytorch import set_device, to_numpy
from agent.image_state.ppg.agent_ppg import AgentPpg

class AgentPpgSemantic(AgentPpg):
    def _update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False, num_workers = 4)

        for _ in range(self.ppo_epochs):       
            for states, images, actions, rewards, dones, next_states, next_images in dataloader: 
                self._training_ppo(states.float().to(self.device), images.long().to(self.device), actions.float().to(self.device), 
                    rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device), next_images.long().to(self.device))

        states, images, _, _, _, _, _ = self.policy_memory.get_all_items()
        self.auxppg_memory.save_all(states, images)
        self.policy_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def _update_auxppg(self):
        dataloader  = DataLoader(self.auxppg_memory, self.batch_size, shuffle = False, num_workers = 4)

        for _ in range(self.auxppg_epochs):       
            for states, images in dataloader:
                self._training_auxppg(states.float().to(self.device), images.long().to(self.device))

        self.auxppg_memory.clear_memory()
        self.policy_old.load_state_dict(self.policy.state_dict())

    def act(self, state, image):
        state, image        = torch.FloatTensor(state).unsqueeze(0).to(self.device), torch.LongTensor(image).unsqueeze(0).to(self.device)
        
        res                 = self.cnn(image)
        action_datas, _     = self.policy(res, state)
        
        if self.is_training_mode:
            action = self.policy_dist.sample(action_datas)
        else:
            action = self.policy_dist.deterministic(action_datas)
              
        return action.detach().cpu().numpy()