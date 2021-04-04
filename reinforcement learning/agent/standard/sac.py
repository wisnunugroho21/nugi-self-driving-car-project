import torch
from torch.utils.data import DataLoader
import copy

from helpers.pytorch_utils import set_device, to_numpy, to_tensor

class AgentSAC():
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
        self.target_value       = copy.deepcopy(self.value)
        self.soft_q1            = soft_q
        self.soft_q2            = copy.deepcopy(self.soft_q1)
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

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(param.data)

    def _training_q(self, states, actions, rewards, dones, next_states):
        self.soft_q_optimizer.zero_grad()

        q_value1            = self.soft_q1(states, actions)
        q_value2            = self.soft_q2(states, actions)

        next_value          = self.target_value(next_states, True)

        loss = self.qLoss.compute_loss(q_value1, rewards, dones, next_value) + self.qLoss.compute_loss(q_value2, rewards, dones, next_value)
        loss.backward()

        self.soft_q_optimizer.step()        

    def _training_values(self, states):
        self.value_optimizer.zero_grad()
        
        predicted_value     = self.value(states)

        action_datas        = self.policy(states, True)
        actions             = self.distribution.sample(action_datas).detach()

        q_value1            = self.soft_q1(states, actions, True)
        q_value2            = self.soft_q2(states, actions, True)

        loss = self.vLoss.compute_loss(predicted_value, action_datas, actions, q_value1, q_value2)
        loss.backward()

        self.value_optimizer.step()

    def _training_policy(self, states):
        self.policy_optimizer.zero_grad()

        action_datas    = self.policy(states)
        actions         = self.distribution.sample(action_datas)

        q_value1        = self.soft_q1(states, actions)
        q_value2        = self.soft_q2(states, actions)

        loss = self.policyLoss.compute_loss(action_datas, actions, q_value1, q_value2)
        loss.backward()

        self.policy_optimizer.step()

    def _update_sac(self):
        if len(self.memory) > self.batch_size:
            for _ in range(self.epochs):
                dataloader  = DataLoader(self.memory, self.batch_size, shuffle = True, num_workers = 8)
                states, actions, rewards, dones, next_states = next(iter(dataloader))

                self._training_q(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu), self.soft_q1, self.soft_q1_optimizer)

                self._training_q(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu), self.soft_q2, self.soft_q2_optimizer)

                self._training_values(to_tensor(states, use_gpu = self.use_gpu))
                self._training_policy(to_tensor(states, use_gpu = self.use_gpu))

            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def update(self):
        self._update_sac()

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.memory.save_all(states, actions, rewards, dones, next_states)
        
    def act(self, state):
        state               = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_datas        = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)    

    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'soft_q1_state_dict': self.soft_q1.state_dict(),
            'soft_q2_state_dict': self.soft_q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
        }, self.folder + '/sac.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/cql.tar', map_location = device)
        
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.soft_q1.load_state_dict(model_checkpoint['soft_q1_state_dict'])
        self.soft_q2.load_state_dict(model_checkpoint['soft_q2_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(model_checkpoint['value_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])