import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from helper.pytorch import set_device, to_numpy

class AgentPpg():  
    def __init__(self, Policy_Model, Value_Model, CnnModel, state_dim, action_dim, policy_dist, policy_loss, auxppg_loss, 
                policy_memory, auxppg_memory, ppo_epochs = 10, auxppg_epochs = 10, n_aux_update = 2, is_training_mode = True, 
                policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32,  learning_rate = 3e-4, folder = 'model', use_gpu = True):   

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batch_size         = batch_size  
        self.ppo_epochs         = ppo_epochs
        self.auxppg_epochs      = auxppg_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.n_aux_update       = n_aux_update

        self.device             = set_device(self.use_gpu)

        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu).float().to(self.device)
        self.policy_old         = Policy_Model(state_dim, action_dim, self.use_gpu).float().to(self.device)

        self.value              = Value_Model(state_dim).float().to(self.device)
        self.value_old          = Value_Model(state_dim).float().to(self.device)

        self.cnn                = CnnModel().float().to(self.device)

        self.policy_dist        = policy_dist

        self.policy_memory      = policy_memory
        self.auxppg_memory      = auxppg_memory
        
        self.policyLoss         = policy_loss
        self.auxppgLoss         = auxppg_loss
        
        self.i_auxppg_update    = 0
        self.i_ppo_update       = 0

        self.ppo_optimizer      = Adam(list(self.policy.parameters()) + list(self.value.parameters()) + list(self.cnn.parameters()), lr = learning_rate)        
        self.auxppg_optimizer   = Adam(list(self.policy.parameters()), lr = learning_rate)

        self.ppo_scaler         = torch.cuda.amp.GradScaler()
        self.auxppg_scaler      = torch.cuda.amp.GradScaler()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def __training_ppo(self, states, images, actions, rewards, dones, next_states, next_images):         
        self.ppo_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            res                 = self.cnn(images)
            next_res            = self.cnn(next_images, True)
            
            old_action_datas, _ = self.policy_old(res, states, True)
            old_values          = self.value_old(res, states, True)
            next_values         = self.value(next_res, next_states, True)

            action_datas, _     = self.policy(res, states)
            values              = self.value(res, states)

            loss = self.policyLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        
        self.ppo_scaler.scale(loss).backward()
        self.ppo_scaler.step(self.ppo_optimizer)
        self.ppo_scaler.update()

    def __training_auxppg(self, states, images):        
        self.auxppg_optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            res                     = self.cnn(images, True)
            
            returns                 = self.value(res, states, True)
            old_action_datas, _     = self.policy_old(res, states, True)

            action_datas, values    = self.policy(res, states)

            loss = self.auxppgLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.auxppg_scaler.scale(loss).backward()
        self.auxppg_scaler.step(self.auxppg_optimizer)
        self.auxppg_scaler.update()

    def __update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False, num_workers = 4)

        for _ in range(self.ppo_epochs):       
            for states, images, actions, rewards, dones, next_states, next_images in dataloader: 
                self.__training_ppo(states.float().to(self.device), images.float().to(self.device), actions.float().to(self.device), 
                    rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device), next_images.float().to(self.device))

        states, images, _, _, _, _, _ = self.policy_memory.get_all_items()
        self.auxppg_memory.save_all(states, images)
        self.policy_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def __update_auxppg(self):
        dataloader  = DataLoader(self.auxppg_memory, self.batch_size, shuffle = False, num_workers = 4)

        for _ in range(self.auxppg_epochs):       
            for states, images in dataloader:
                self.__training_auxppg(states.float().to(self.device), images.float().to(self.device))

        self.auxppg_memory.clear_memory()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def act(self, state, image):
        state, image        = torch.FloatTensor(state).unsqueeze(0).to(self.device), torch.FloatTensor(self.trans(image)).unsqueeze(0).to(self.device)
        
        res                 = self.cnn(image)
        action_datas, _     = self.policy(res, state)
        
        if self.is_training_mode:
            action = self.policy_dist.sample(action_datas)
        else:
            action = self.policy_dist.deterministic(action_datas)
              
        return action.detach().cpu().numpy()

    def save_memory(self, policy_memory):
        states, images, actions, rewards, dones, next_states, next_images = policy_memory.get_all_items()
        self.policy_memory.save_all(states, images, actions, rewards, dones, next_states, next_images)

    def update(self):
        self.__update_ppo()
        self.i_auxppg_update += 1

        if self.i_auxppg_update % self.n_aux_update == 0 and self.i_auxppg_update != 0:
            self.__update_auxppg()
            self.i_auxppg_update = 0

    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'cnn_state_dict': self.cnn.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'auxppg_optimizer_state_dict': self.auxppg_optimizer.state_dict(),
            'ppo_scaler_state_dict': self.ppo_scaler.state_dict(),
            'auxppg_scaler_state_dict': self.auxppg_scaler.state_dict(),
        }, self.folder + '/ppg.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/ppg.tar', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.cnn.load_state_dict(model_checkpoint['cnn_state_dict'])
        self.ppo_optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])        
        self.auxppg_optimizer.load_state_dict(model_checkpoint['auxppg_optimizer_state_dict'])   
        self.ppo_scaler.load_state_dict(model_checkpoint['ppo_scaler_state_dict'])        
        self.auxppg_scaler.load_state_dict(model_checkpoint['auxppg_scaler_state_dict'])  

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            print('Model is evaluating...')

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def set_weights(self, policy_weights, value_weights):
        self.policy.load_state_dict(policy_weights)
        self.value.load_state_dict(value_weights)