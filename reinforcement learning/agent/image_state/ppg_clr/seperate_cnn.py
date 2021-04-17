import copy

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from helpers.pytorch_utils import set_device, to_numpy, to_tensor
from agent.standard.ppg import AgentPPG

class AgentImageStatePPGClr(AgentPPG):
    def __init__(self, projector_policy, projector_value, cnn_policy, cnn_value, policy, value, state_dim, action_dim, distribution, ppo_loss, aux_ppg_loss, aux_clr_loss, ppo_memory, aux_ppg_memory, aux_clr_memory,
            ppo_optimizer, aux_ppg_optimizer, aux_policy_clr_optim, aux_value_clr_optim, PPO_epochs = 10, aux_ppg_epochs = 10, aux_clr_epochs = 10, n_aux_update = 10, is_training_mode = True, policy_kl_range = 0.03, 
            policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, batch_size = 32,  folder = 'model', use_gpu = True):

        super().__init__(policy, value, state_dim, action_dim, distribution, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory, 
            ppo_optimizer, aux_ppg_optimizer, PPO_epochs, aux_ppg_epochs, n_aux_update, is_training_mode, policy_kl_range, 
            policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size,  folder, use_gpu)

        self.cnn_policy             = cnn_policy
        self.projector_policy       = projector_policy

        self.cnn_value              = cnn_value
        self.projector_value        = projector_value

        self.cnn_policy_old         = copy.deepcopy(self.cnn_policy)
        self.projector_policy_old   = copy.deepcopy(self.projector_policy)        

        self.cnn_value_old          = copy.deepcopy(self.cnn_value)
        self.projector_value_old    = copy.deepcopy(self.projector_value)

        self.aux_policy_clr_optim   = aux_policy_clr_optim
        self.aux_policy_clr_scaler  = torch.cuda.amp.GradScaler()

        self.aux_value_clr_optim    = aux_value_clr_optim
        self.aux_value_clr_scaler   = torch.cuda.amp.GradScaler()

        self.aux_clrLoss            = aux_clr_loss
        self.aux_clr_memory         = aux_clr_memory
        self.aux_clr_epochs         = aux_clr_epochs

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.soft_tau = 0.95

        if self.is_training_mode:
            self.cnn_policy.train()
            self.cnn_value.train()
            self.projector_policy.train()            
            self.projector_value.train()
        else:
            self.cnn_policy.eval()
            self.cnn_value.eval()
            self.projector_policy.eval()            
            self.projector_value.eval()

    def _training_ppo(self, images, states, actions, rewards, dones, next_images, next_states):
        self.ppo_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            policy_res          = self.cnn_policy(images)
            action_datas, _     = self.policy(policy_res, states)

            values_res          = self.cnn_value(images)
            values              = self.value(values_res, states)
            
            policy_res_old      = self.cnn_policy_old(images, True)
            old_action_datas, _ = self.policy_old(policy_res_old, states, True)

            value_res_old       = self.cnn_value_old(images, True)
            old_values          = self.value_old(value_res_old, states, True)

            value_next_res      = self.cnn_value(next_images, True)
            next_values         = self.value(value_next_res, next_states, True)

            loss = self.ppoLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        
        self.ppo_scaler.scale(loss).backward()
        self.ppo_scaler.step(self.ppo_optimizer)
        self.ppo_scaler.update()

    def _training_aux_ppg(self, images, states):
        self.aux_ppg_optimizer.zero_grad()        
        with torch.cuda.amp.autocast():
            policy_res              = self.cnn_policy(images)
            action_datas, values    = self.policy(policy_res, states)

            policy_res_old          = self.cnn_policy_old(images, True)
            old_action_datas, _     = self.policy_old(policy_res_old, states, True)            

            values_res              = self.cnn_value(images, True)
            returns                 = self.value(values_res, states, True)

            loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.aux_ppg_scaler.scale(loss).backward()
        self.aux_ppg_scaler.step(self.aux_ppg_optimizer)
        self.aux_ppg_scaler.update()

    def _training_aux_clr(self, input_images, target_images):
        self.aux_policy_clr_optim.zero_grad()
        with torch.cuda.amp.autocast():
            policy_out1        = self.cnn_policy(input_images)
            policy_encoded1    = self.projector_policy(policy_out1)

            policy_out2        = self.cnn_policy_old(target_images, True)
            policy_encoded2    = self.projector_policy_old(policy_out2, True)

            loss = self.aux_clrLoss.compute_loss(policy_encoded1, policy_encoded2)

        self.aux_policy_clr_scaler.scale(loss).backward()
        self.aux_policy_clr_scaler.step(self.aux_policy_clr_optim)
        self.aux_policy_clr_scaler.update()

        self.aux_value_clr_optim.zero_grad()
        with torch.cuda.amp.autocast():
            value_out1        = self.cnn_value(input_images)
            value_encoded1    = self.projector_value(value_out1)

            value_out2        = self.cnn_value_old(target_images, True)
            value_encoded2    = self.projector_value_old(value_out2, True)

            loss = self.aux_clrLoss.compute_loss(value_encoded1, value_encoded2)

        self.aux_value_clr_scaler.scale(loss).backward()
        self.aux_value_clr_scaler.step(self.aux_value_clr_optim)
        self.aux_value_clr_scaler.update()

    def _update_ppo(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        self.cnn_policy_old.load_state_dict(self.cnn_policy.state_dict())
        self.cnn_value_old.load_state_dict(self.cnn_value.state_dict())

        dataloader = DataLoader(self.ppo_memory, self.batch_size, shuffle = False, num_workers = 8)

        for _ in range(self.ppo_epochs):       
            for images, states, actions, rewards, dones, next_images, next_states in dataloader: 
                self._training_ppo(images.to(self.device), states.to(self.device), actions.to(self.device), 
                    rewards.to(self.device), dones.to(self.device), next_images.to(self.device), next_states.to(self.device))

        images, states, _, _, _, _, _ = self.ppo_memory.get_all_items()
        self.aux_ppg_memory.save_all(images, states)
        self.ppo_memory.clear_memory()

    def _update_aux_ppg(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.cnn_policy_old.load_state_dict(self.cnn_policy.state_dict())

        dataloader  = DataLoader(self.aux_ppg_memory, self.batch_size, shuffle = False, num_workers = 8)

        for _ in range(self.aux_ppg_epochs):       
            for images, states in dataloader:
                self._training_aux_ppg(images.to(self.device), states.to(self.device))

        images, _ = self.aux_ppg_memory.get_all_items()
        self.aux_clr_memory.save_all(images)
        self.aux_ppg_memory.clear_memory()

    def _update_aux_clr(self):
        self.cnn_policy_old.load_state_dict(self.cnn_policy.state_dict())
        self.projector_policy_old.load_state_dict(self.projector_policy.state_dict())
        self.cnn_value_old.load_state_dict(self.cnn_value.state_dict())
        self.projector_value_old.load_state_dict(self.projector_value.state_dict())

        dataloader  = DataLoader(self.aux_clr_memory, self.batch_size, shuffle = True, num_workers = 8)

        for _ in range(self.aux_clr_epochs):
            for input_images, target_images in dataloader:
                self._training_aux_clr(input_images.to(self.device), target_images.to(self.device))            

        self.aux_clr_memory.clear_memory()

    def update(self):
        self._update_ppo()
        self.i_update += 1

        if self.i_update % self.n_aux_update == 0:
            self._update_aux_ppg()
            self._update_aux_clr()
            self.i_update = 0

    def save_memory(self, policy_memory):
        images, states, actions, rewards, dones, next_images, next_states = policy_memory.get_all_items()
        self.ppo_memory.save_all(images, states, actions, rewards, dones, next_images, next_states)

    def act(self, image, state):
        image, state        = self.trans(image).unsqueeze(0).to(self.device), torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        res                 = self.cnn_policy(image)
        action_datas, _     = self.policy(res, state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'cnn_policy_state_dict': self.cnn_policy.state_dict(),
            'projector_policy_state_dict': self.projector_policy.state_dict(),
            'cnn_value_state_dict': self.cnn_value.state_dict(),
            'projector_value_state_dict': self.projector_value.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'aux_ppg_optimizer_state_dict': self.aux_ppg_optimizer.state_dict(),
            'aux_policy_clr_optim_state_dict': self.aux_policy_clr_optim.state_dict(),
            'aux_value_clr_optim_state_dict': self.aux_value_clr_optim.state_dict(),
            'ppo_scaler_state_dict': self.ppo_scaler.state_dict(),
            'aux_ppg_scaler_state_dict': self.aux_ppg_scaler.state_dict(),
            'aux_policy_clr_scaler_state_dict': self.aux_policy_clr_scaler.state_dict(),
            'aux_value_clr_scaler_state_dict': self.aux_value_clr_scaler.state_dict(),
        }, self.folder + '/ppg.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/ppg.tar', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.cnn_policy.load_state_dict(model_checkpoint['cnn_policy_state_dict'])
        self.projector_policy.load_state_dict(model_checkpoint['projector_policy_state_dict'])
        self.cnn_value.load_state_dict(model_checkpoint['cnn_value_state_dict'])
        self.projector_value.load_state_dict(model_checkpoint['projector_value_state_dict'])
        self.ppo_optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])        
        self.aux_ppg_optimizer.load_state_dict(model_checkpoint['aux_ppg_optimizer_state_dict'])   
        self.aux_policy_clr_optim.load_state_dict(model_checkpoint['aux_policy_clr_optim_state_dict'])
        self.aux_value_clr_optim.load_state_dict(model_checkpoint['aux_value_clr_optim_state_dict'])
        self.ppo_scaler.load_state_dict(model_checkpoint['ppo_scaler_state_dict'])        
        self.aux_ppg_scaler.load_state_dict(model_checkpoint['aux_ppg_scaler_state_dict'])  
        self.aux_policy_clr_scaler.load_state_dict(model_checkpoint['aux_policy_clr_scaler_state_dict'])
        self.aux_value_clr_scaler.load_state_dict(model_checkpoint['aux_value_clr_scaler_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            self.cnn.train()
            self.projector.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            self.cnn.eval()
            self.projector.eval()
            print('Model is evaluating...')