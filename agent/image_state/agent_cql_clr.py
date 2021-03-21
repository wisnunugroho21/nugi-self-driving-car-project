import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from helper.pytorch import set_device, to_tensor, to_numpy

class AgentCqlClr():
    def __init__(self, Policy_Model, Value_Model, Q_Model, CnnModel, ProjectionModel, state_dim, action_dim, policy_dist, q_loss, v_loss, policy_loss, auxclr_loss, 
        policy_memory, auxclr_memory, is_training_mode = True, batch_size = 32, cql_epochs = 4, auxclr_epochs = 4, learning_rate = 3e-4, 
        folder = 'model', use_gpu = True):

        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.cql_epochs         = cql_epochs
        self.auxclr_epochs      = auxclr_epochs

        self.device             = set_device(self.use_gpu)
        
        self.soft_q             = Q_Model(state_dim, action_dim).float().to(self.device)
        self.value              = Value_Model(state_dim).float().to(self.device)
        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu).float().to(self.device)

        self.cnn                = CnnModel().float().to(self.device)
        self.auxclr_projection  = ProjectionModel().float().to(self.device)

        self.policy_dist        = policy_dist

        self.policy_memory      = policy_memory
        self.auxclr_memory      = auxclr_memory
        
        self.qLoss              = q_loss
        self.vLoss              = v_loss
        self.policyLoss         = policy_loss
        self.auxclrLoss         = auxclr_loss
              
        self.soft_q_optimizer   = Adam(list(self.soft_q.parameters()) + list(self.cnn.parameters()), lr = learning_rate)        
        self.value_optimizer    = Adam(self.value.parameters(), lr = learning_rate)
        self.policy_optimizer   = Adam(self.policy.parameters(), lr = learning_rate)
        self.auxclr_optimizer   = Adam(list(self.cnn.parameters()) + list(self.auxclr_projection.parameters()), lr = learning_rate)

        self.soft_q_scaler      = torch.cuda.amp.GradScaler()
        self.value_scaler       = torch.cuda.amp.GradScaler()
        self.policy_scaler      = torch.cuda.amp.GradScaler()
        self.auxclr_scaler      = torch.cuda.amp.GradScaler()

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __training_q(self, states, images, actions, rewards, dones, next_states, next_images):
        self.soft_q_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            res                         = self.cnn(images)
            next_res                    = self.cnn(next_images, True)

            action_datas                = self.policy(res, states, True)
            predicted_actions           = self.policy_dist.sample(action_datas).detach()
            next_value                  = self.value(next_res, next_states, True)

            naive_predicted_q_value     = self.soft_q(res, states, predicted_actions)
            predicted_q_value           = self.soft_q(res, states, actions)

            loss = self.qLoss.compute_loss(naive_predicted_q_value, predicted_q_value, rewards, dones, next_value)        
        
        self.soft_q_scaler.scale(loss).backward()
        self.soft_q_scaler.step(self.soft_q_optimizer)
        self.soft_q_scaler.update()       

    def __training_values(self, states, images):
        self.value_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            res                 = self.cnn(images, True)

            action_datas        = self.policy(res, states, True)
            predicted_actions   = self.policy_dist.sample(action_datas).detach()
            q_value             = self.soft_q(res, states, predicted_actions, True)
            
            predicted_value     = self.value(res, states)

            loss = self.vLoss.compute_loss(predicted_value, q_value)

        self.value_scaler.scale(loss).backward()
        self.value_scaler.step(self.value_optimizer)
        self.value_scaler.update()

    def __training_policy(self, states, images):
        self.policy_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            res                 = self.cnn(images, True)

            action_datas        = self.policy(res, states)
            predicted_actions   = self.policy_dist.sample(action_datas)

            q_value             = self.soft_q(states, predicted_actions)

            loss = self.policyLoss.compute_loss(q_value)

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()

    def __training_auxclr(self, first_images, second_images):
        self.auxclr_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out1        = self.cnn(first_images)
            encoded1    = self.auxclr_projection(out1)

            out2        = self.cnn(second_images)
            encoded2    = self.auxclr_projection(out2)

            loss = (self.auxclrLoss.compute_loss(encoded1, encoded2) + self.auxclrLoss.compute_loss(encoded2, encoded1)) / 2.0

        self.auxclr_scaler.scale(loss).backward()
        self.auxclr_scaler.step(self.auxclr_optimizer)
        self.auxclr_scaler.update()

    def __update_offpolicy(self):
        dataloader  = DataLoader(self.policy_memory, self.batch_size, shuffle = True, num_workers = 2)

        for _ in range(self.epochs):
            for states, images, actions, rewards, dones, next_states, next_images in dataloader:
                self.__training_q(states.float().to(self.device), images.float().to(self.device), actions.float().to(self.device), 
                    rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device), next_images.float().to(self.device))

                self.__training_values(states.float().to(self.device), images.float().to(self.device))
                self.__training_policy(states.float().to(self.device), images.float().to(self.device))

        states, images, _, _, _, _, _ = self.policy_memory.get_all_items(get_tensor_images = False)
        self.auxclr_memory.save_all(images, save_tensor_images = True)
        self.policy_memory.clear_memory(delete_img = False)

    def __update_auxclr(self):
        dataloader  = DataLoader(self.auxclr_memory, self.batch_size, shuffle = True, num_workers = 2)

        for _ in range(self.AuxClr_epochs):
            for first_images, second_images in dataloader:
                self.__training_auxclr(to_tensor(first_images, use_gpu = self.use_gpu), to_tensor(second_images, use_gpu = self.use_gpu))

        self.auxclr_memory.clear_memory()

    def save_memory(self, policy_memory):
        states, images, actions, rewards, dones, next_states, next_images = policy_memory.get_all_items(get_tensor_images = False)
        self.policy_memory.save_all(states, images, actions, rewards, dones, next_states, next_images, save_tensor_images = False)
        
    def act(self, state, image):
        state, image        = state.unsqueeze(0).float().to(self.device), self.trans(image).unsqueeze(0).float().to(self.device)

        res                 = self.cnn(image)
        action_datas, _     = self.policy(res, state)
        
        if self.is_training_mode:
            action = self.policy_dist.sample(action_datas)
        else:
            action = self.policy_dist.act_deterministic(action_datas)
              
        return action.detach().cpu().numpy()

    def update(self):
        self.__update_offpolicy()
        self.__update_auxclr()
        
    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'soft_q1_state_dict': self.soft_q1.state_dict(),
            'soft_q2_state_dict': self.soft_q2.state_dict(),
            'cnn_state_dict': self.cnn.state_dict(),
            'auxclr_pro_state_dict': self.auxclr_projection.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
            'auxclr_optimizer_state_dict': self.auxclr_optimizer.state_dict(),
            'policy_scaler_state_dict': self.policy_scaler.state_dict(),
            'value_scaler_state_dict': self.value_scaler.state_dict(),
            'soft_q_scaler_state_dict': self.soft_q_scaler.state_dict(),
            'auxclr_scaler_state_dict': self.auxclr_scaler.state_dict()
            }, self.folder + '/cql.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/cql.tar', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.soft_q1.load_state_dict(model_checkpoint['soft_q1_state_dict'])        
        self.soft_q2.load_state_dict(model_checkpoint['soft_q2_state_dict'])
        self.cnn.load_state_dict(model_checkpoint['cnn_state_dict'])
        self.auxclr_projection.load_state_dict(model_checkpoint['auxclr_pro_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(model_checkpoint['value_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])
        self.auxclr_optimizer.load_state_dict(model_checkpoint['auxclr_optimizer_state_dict'])   
        self.policy_scaler.load_state_dict(model_checkpoint['policy_scaler_state_dict'])        
        self.value_scaler.load_state_dict(model_checkpoint['value_scaler_state_dict'])
        self.soft_q_scaler.load_state_dict(model_checkpoint['soft_q_scaler_state_dict'])
        self.auxclr_scaler.load_state_dict(model_checkpoint['auxclr_scaler_state_dict'])     

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            print('Model is evaluating...')