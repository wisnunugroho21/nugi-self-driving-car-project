import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from agent.standard.cql import AgentCql

class AgentImageStateCql(AgentCql):
    def __init__(self, cnn, soft_q, value, policy, state_dim, action_dim, distribution, q_loss, v_loss, policy_loss, memory, 
        soft_q_optimizer, value_optimizer, policy_optimizer, is_training_mode = True, batch_size = 32, epochs = 1, 
        soft_tau = 0.95, folder = 'model', use_gpu = True):

        super().__init__(soft_q, value, policy, state_dim, action_dim, distribution, q_loss, v_loss, policy_loss, memory, 
        soft_q_optimizer, value_optimizer, policy_optimizer, is_training_mode, batch_size, epochs, soft_tau, folder, use_gpu)

        self.cnn = cnn

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _training_q(self, images, states, actions, rewards, dones, next_images, next_states):
        self.soft_q_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            res                         = self.cnn(images)
            next_res                    = self.cnn(next_images, True)

            predicted_actions           = self.policy(res, states, True)
            next_value                  = self.value(next_res, next_states, True)

            naive_predicted_q_value     = self.soft_q(res, states, predicted_actions)
            predicted_q_value           = self.soft_q(res, states, actions)

            loss = self.qLoss.compute_loss(naive_predicted_q_value, predicted_q_value, rewards, dones, next_value)        
        
        self.soft_q_scaler.scale(loss).backward()
        self.soft_q_scaler.step(self.soft_q_optimizer)
        self.soft_q_scaler.update()       

    def _training_values(self, images, states):
        self.value_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            res                 = self.cnn(images, True)

            predicted_actions   = self.policy(res, states, True)
            q_value             = self.soft_q(res, states, predicted_actions, True)
            
            predicted_value     = self.value(res, states)

            loss = self.vLoss.compute_loss(predicted_value, q_value)

        self.value_scaler.scale(loss).backward()
        self.value_scaler.step(self.value_optimizer)
        self.value_scaler.update()

    def _training_policy(self, images, states):
        self.policy_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            res                 = self.cnn(images, True)

            predicted_actions   = self.policy(res, states)
            q_value             = self.soft_q(states, predicted_actions)

            loss = self.policyLoss.compute_loss(q_value)

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()

    def _update_offpolicy(self):
        dataloader  = DataLoader(self.policy_memory, self.batch_size, shuffle = True, num_workers = 4)

        for _ in range(self.epochs):
            for images, states, actions, rewards, dones, next_images, next_states in dataloader:
                self._training_q(images.float().to(self.device), states.float().to(self.device), actions.float().to(self.device), 
                    rewards.float().to(self.device), dones.float().to(self.device), next_images.float().to(self.device), next_states.float().to(self.device))

                self._training_values(images.float().to(self.device), states.float().to(self.device))
                self._training_policy(images.float().to(self.device), states.float().to(self.device))

        self.memory.clear_memory()

    def save_memory(self, policy_memory):
        images, states, actions, rewards, dones, next_images, next_states = policy_memory.get_all_items()
        self.memory.save_all(images, states, actions, rewards, dones, next_images, next_states)
        
    def act(self, image, state):
        image, state        = self.trans(image).unsqueeze(0).to(self.device), torch.FloatTensor(state).unsqueeze(0).to(self.device)

        res     = self.cnn(image)
        action  = self.policy(res, state)
                      
        return action.detach().cpu().numpy()

    def update(self):
        self._update_offpolicy()
        
    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'soft_q_state_dict': self.soft_q.state_dict(),
            'cnn_state_dict': self.cnn.state_dict(),
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
        self.cnn.load_state_dict(model_checkpoint['cnn_state_dict'])
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