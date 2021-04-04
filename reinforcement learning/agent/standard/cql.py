import torch
from torch.utils.data import DataLoader
import copy

from helpers.pytorch_utils import set_device

class AgentCql():
    def __init__(self, soft_q, value, policy, state_dim, action_dim, distribution, q_loss, v_loss, policy_loss, memory, 
        soft_q_optimizer, value_optimizer, policy_optimizer, is_training_mode = True, batch_size = 32, epochs = 1, 
        soft_tau = 0.95, folder = 'model', use_gpu = True):

        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.epochs             = epochs
        self.soft_tau           = soft_tau

        self.value              = value
        self.soft_q             = soft_q
        self.policy             = policy

        self.distribution       = distribution
        self.memory             = memory
        
        self.qLoss              = q_loss
        self.vLoss              = v_loss
        self.policyLoss         = policy_loss

        self.device             = set_device(self.use_gpu)
        self.i_update           = 0
        
        self.soft_q_optimizer   = soft_q_optimizer
        self.value_optimizer    = value_optimizer
        self.policy_optimizer   = policy_optimizer

        self.soft_q_scaler      = torch.cuda.amp.GradScaler()
        self.value_scaler       = torch.cuda.amp.GradScaler()
        self.policy_scaler      = torch.cuda.amp.GradScaler()

    def _training_q(self, states, actions, rewards, dones, next_states):
        self.soft_q_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            predicted_actions           = self.policy(states, True)
            next_value                  = self.value(next_states, True)

            naive_predicted_q_value     = self.soft_q(states, predicted_actions)
            predicted_q_value           = self.soft_q(states, actions)

            loss = self.qLoss.compute_loss(naive_predicted_q_value, predicted_q_value, rewards, dones, next_value)        
        
        self.soft_q_scaler.scale(loss).backward()
        self.soft_q_scaler.step(self.soft_q_optimizer)
        self.soft_q_scaler.update()       

    def _training_values(self, states):
        self.value_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            predicted_actions   = self.policy(states, True)
            q_value             = self.soft_q(states, predicted_actions, True)
            
            predicted_value     = self.value(states)

            loss = self.vLoss.compute_loss(predicted_value, q_value)

        self.value_scaler.scale(loss).backward()
        self.value_scaler.step(self.value_optimizer)
        self.value_scaler.update()

    def _training_policy(self, states):
        self.policy_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            predicted_actions   = self.policy(states)
            q_value             = self.soft_q(states, predicted_actions)

            loss = self.policyLoss.compute_loss(q_value)

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()

    def _update_offpolicy(self):
        dataloader  = DataLoader(self.memory, self.batch_size, shuffle = True, num_workers = 8)

        for _ in range(self.epochs):
            for states, images, actions, rewards, dones, next_states, next_images in dataloader:
                self._training_q(states.float().to(self.device), images.float().to(self.device), actions.float().to(self.device), 
                    rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device), next_images.float().to(self.device))

                self._training_values(states.float().to(self.device), images.float().to(self.device))
                self._training_policy(states.float().to(self.device), images.float().to(self.device))

        self.memory.clear_memory(delete_img = False)

    def save_memory(self, policy_memory):
        states, images, actions, rewards, dones, next_states, next_images = policy_memory.get_all_items(get_tensor_images = False)
        self.memory.save_all(states, images, actions, rewards, dones, next_states, next_images, save_tensor_images = False)
        
    def act(self, state):
        state   = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action  = self.policy(state)
                      
        return action.detach().cpu().numpy()

    def update(self):
        self._update_offpolicy()
        
    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'soft_q_state_dict': self.soft_q.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
            'policy_scaler_state_dict': self.policy_scaler.state_dict(),
            'value_scaler_state_dict': self.value_scaler.state_dict(),
            'soft_q_scaler_state_dict': self.soft_q_scaler.state_dict(),
        }, self.folder + '/cql.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/cql.tar', map_location = device)
        
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.soft_q.load_state_dict(model_checkpoint['soft_q_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(model_checkpoint['value_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])
        self.policy_scaler.load_state_dict(model_checkpoint['policy_scaler_state_dict'])        
        self.value_scaler.load_state_dict(model_checkpoint['value_scaler_state_dict'])
        self.soft_q_scaler.load_state_dict(model_checkpoint['soft_q_scaler_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            print('Model is evaluating...')