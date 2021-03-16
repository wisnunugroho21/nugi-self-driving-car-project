import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from gym.spaces.box import Box
import numpy as np

import glob
import os
import sys
import datetime
import time
import numpy as np
import cv2
import math
import queue
import random
from PIL import Image

try:
    sys.path.append(glob.glob("/home/nugroho/Projects/Simulator/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg" % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def set_device(use_gpu = True):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

def to_numpy(datas, use_gpu = True):
    if use_gpu:
        if torch.cuda.is_available():
            return datas.detach().cpu().numpy()
        else:
            return datas.detach().numpy()
    else:
        return datas.detach().numpy()

def to_tensor(datas, use_gpu = True, first_unsqueeze = False, last_unsqueeze = False, detach = False):
    if isinstance(datas, tuple):
        datas = list(datas)
        for i, data in enumerate(datas):
            data    = torch.FloatTensor(data).to(set_device(use_gpu))
            if first_unsqueeze: 
                data    = data.unsqueeze(0)
            if last_unsqueeze:
                data    = data.unsqueeze(-1) if data.shape[-1] != 1 else data
            if detach:
                data    = data.detach()
            datas[i] = data
        datas = tuple(datas)

    elif isinstance(datas, list):
        for i, data in enumerate(datas):
            data    = torch.FloatTensor(data).to(set_device(use_gpu))
            if first_unsqueeze: 
                data    = data.unsqueeze(0)
            if last_unsqueeze:
                data    = data.unsqueeze(-1) if data.shape[-1] != 1 else data
            if detach:
                data    = data.detach()
            datas[i] = data
        datas = tuple(datas)

    else:
        datas   = torch.FloatTensor(datas).to(set_device(use_gpu))
        if first_unsqueeze: 
            datas   = datas.unsqueeze(0)
        if last_unsqueeze:
            datas   = datas.unsqueeze(-1) if datas.shape[-1] != 1 else datas
        if detach:
            datas   = datas.detach()
    
    return datas

class CarlaEnv():
    def __init__(self, im_height = 480, im_width = 480, im_preview = False, max_step = 512):
        self.cur_step           = 0
        self.collision_hist     = []
        self.crossed_line_hist  = []
        self.actor_list         = []
        self.init_pos           = [
            [-149.1, -94.7, 89.3],
            [-152.3, -96.5, 90.8],
            [-141.9, -33.6, -89.0],
            [-145.5, -33.6, -89.6],
        ]

        self.im_height              = im_height
        self.im_width               = im_width
        self.im_preview             = im_preview
        self.max_step               = max_step

        self.observation_space      = Box(low = -1.0, high = 1.0, shape = (im_height, im_width))
        self.action_space           = Box(low = -1.0, high = 1.0, shape = (2, 1))

        client              = carla.Client('127.0.0.1', 2000)
        self.world          = client.get_world()
        blueprint_library   = self.world.get_blueprint_library()

        self.model_3        = blueprint_library.filter('model3')[0]
        self.rgb_cam        = blueprint_library.find('sensor.camera.rgb')
        self.col_detector   = blueprint_library.find('sensor.other.collision')
        self.crl_detector   = blueprint_library.find('sensor.other.lane_invasion')        

        self.rgb_cam.set_attribute('image_size_x', f'{im_height}')
        self.rgb_cam.set_attribute('image_size_y', f'{im_width}')

        settings                        = self.world.get_settings()
        settings.synchronous_mode       = True
        settings.fixed_delta_seconds    = 0.05
        self.world.apply_settings(settings)

        self.cam_queue  = queue.Queue()         
        
    def __del__(self):
        for actor in self.actor_list:
            actor.destroy()
        del self.actor_list[:]

    def __process_image(self, image):
        # image.convert(carla.ColorConverter.CityScapesPalette)        
        
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, -1))
        i = i[:, :, :3]

        if self.im_preview:
            cv2.imshow('', i)
            cv2.waitKey(1)

        i = Image.fromarray(i)
        return i

    def __process_collision(self, event):
        self.collision_hist.append(event)

    def __process_crossed_line(self, event):
        self.crossed_line_hist.append(event)

    def __tick_env(self):
        self.world.tick()
        time.sleep(0.05)

    def is_discrete(self):
        return False

    def get_obs_dim(self):
        return self.im_height * self.im_width
            
    def get_action_dim(self):
        return 2

    def reset(self):
        for actor in self.actor_list:
            actor.destroy()
        del self.actor_list[:]        

        idx_pos = np.random.randint(len(self.init_pos))
        pos     = carla.Transform(carla.Location(x = self.init_pos[idx_pos][0], y = self.init_pos[idx_pos][1], z = 1.0), 
            carla.Rotation(pitch = 0, yaw = self.init_pos[idx_pos][2], roll = 0))
        
        self.vehicle    = self.world.spawn_actor(self.model_3, pos)        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0, brake = 1.0, steer = 0))        
                
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, carla.Transform(carla.Location(x = 1.6, z = 1.7)), attach_to = self.vehicle)
        self.cam_sensor.listen(self.cam_queue.put)

        for _ in range(4):
            self.__tick_env()
        
        self.col_sensor = self.world.spawn_actor(self.col_detector, carla.Transform(), attach_to = self.vehicle)
        self.col_sensor.listen(lambda event: self.__process_collision(event))
        
        self.crl_sensor = self.world.spawn_actor(self.crl_detector, carla.Transform(), attach_to = self.vehicle)
        self.crl_sensor.listen(lambda event: self.__process_crossed_line(event))
        
        self.actor_list.append(self.cam_sensor)
        self.actor_list.append(self.col_sensor)
        self.actor_list.append(self.crl_sensor)
        self.actor_list.append(self.vehicle)

        self.cur_step = 0

        image = self.__process_image(self.cam_queue.get())
        del self.collision_hist[:]
        del self.crossed_line_hist[:] 
        
        return image, np.array([0, 0])

    def step(self, action):
        prev_loc    = self.vehicle.get_location()

        steer       = -1 if action[0] < -1 else 1 if action[0] > 1 else action[0]
        throttle    = 0 if action[1] < 0 else 1 if action[1] > 1 else action[1]
        brake       = 0 if action[2] < 0 else 1 if action[2] > 1 else action[2]
        self.vehicle.apply_control(carla.VehicleControl(steer = float(steer), throttle = float(throttle), brake = float(brake)))

        self.__tick_env()
        self.cur_step   += 1

        v       = self.vehicle.get_velocity()
        kmh     = math.sqrt(v.x ** 2 + v.y ** 2)
                
        loc     = self.vehicle.get_location()
        dif_x   = loc.x - prev_loc.x if loc.x - prev_loc.x >= 0.05 else 0
        dif_y   = loc.y - prev_loc.y if loc.y - prev_loc.y >= 0.05 else 0
        dif_loc = math.sqrt(dif_x ** 2 + dif_y ** 2)

        done    = False
        reward  = (dif_loc * 100) - 1.0        
        
        image   = self.__process_image(self.cam_queue.get())
        if len(self.crossed_line_hist) > 0 or len(self.collision_hist) > 0 or loc.x >= -100 or loc.y >= -10 or self.cur_step >= self.max_step:
            done = True
        
        return image, np.array([kmh, steer]), reward, done, None

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = True, depth_multiplier = 1):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.nn_layer = nn.Sequential(
            nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin),
            nn.Conv2d(nin * depth_multiplier, nout, kernel_size = 1, bias = bias)
        )

    def forward(self, x):
        return self.nn_layer(x)


class SpatialAtrousExtractor(nn.Module):
    def __init__(self, dim, rate):
        super(SpatialAtrousExtractor, self).__init__()        

        self.spatial_atrous = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = rate, dilation = rate, bias = False, groups = dim),            
            nn.ReLU()
		)

    def forward(self, x):
        x = self.spatial_atrous(x)
        return x

class AtrousSpatialPyramidConv2d(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(AtrousSpatialPyramidConv2d, self).__init__()        

        self.extractor1 = SpatialAtrousExtractor(dim_in, 1)
        self.extractor2 = SpatialAtrousExtractor(dim_in, 4)
        self.extractor3 = SpatialAtrousExtractor(dim_in, 8)

        self.out = nn.Sequential(
            DepthwiseSeparableConv2d(3 * dim_in, dim_out, kernel_size = 1)
        )

    def forward(self, x):
        x1 = self.extractor1(x)
        x2 = self.extractor2(x)
        x3 = self.extractor3(x) 

        xout = torch.cat((x1, x2, x3), 1)
        xout = self.out(xout)

        return xout

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__()   

      self.bn1 = nn.BatchNorm2d(32)
      self.bn2 = nn.BatchNorm2d(64)

      self.conv1 = nn.Sequential(
        AtrousSpatialPyramidConv2d(3, 8),
        nn.ReLU(),
        DepthwiseSeparableConv2d(8, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      )

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(16, 16, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
      )

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(16, 32, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
      )

      self.conv4 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 32, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
      )

      self.conv5 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 64, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
      )

      self.conv_out = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      )
        
    def forward(self, image, detach = False):
      i1  = self.conv1(image)
      i2  = self.conv2(i1)
      i3  = self.conv3(i1)
      i23 = self.bn1(i2 + i3)
      i4  = self.conv4(i23)
      i5  = self.conv5(i23)
      i45 = self.bn2(i4 + i5)
      out = self.conv_out(i45)
      out = out.mean([-1, -2])

      if detach:
        return out.detach()
      else:
        return out

class ProjectionModel(nn.Module):
    def __init__(self):
      super(ProjectionModel, self).__init__()

      self.nn_layer   = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
      )

    def forward(self, image, detach = False):      
      if detach:
        return self.nn_layer(image).detach()
      else:
        return self.nn_layer(image)

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Policy_Model, self).__init__()

      self.std                  = torch.FloatTensor([1.0, 0.5, 0.5]).to(set_device(use_gpu))

      self.state_extractor      = nn.Sequential( nn.Linear(2, 64), nn.ReLU() )
      self.nn_layer             = nn.Sequential( nn.Linear(320, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU() )

      self.critic_layer         = nn.Sequential( nn.Linear(64, 1) )
      self.actor_tanh_layer     = nn.Sequential( nn.Linear(64, 1), nn.Tanh() )
      self.actor_sigmoid_layer  = nn.Sequential( nn.Linear(64, 2), nn.Sigmoid() )            
        
    def forward(self, image, state, detach = False):
      s   = self.state_extractor(state)
      x   = torch.cat([image, s], -1)
      x   = self.nn_layer(x)

      action_tanh     = self.actor_tanh_layer(x)
      action_sigmoid  = self.actor_sigmoid_layer(x)
      action          = torch.cat((action_tanh, action_sigmoid), -1)

      if detach:
        return (action.detach(), self.std.detach()), self.critic_layer(x).detach()
      else:
        return (action, self.std), self.critic_layer(x)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim):
      super(Value_Model, self).__init__()

      self.state_extractor      = nn.Sequential( nn.Linear(2, 64), nn.ReLU() )
      self.nn_layer             = nn.Sequential( nn.Linear(320, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU() )
      self.critic_layer         = nn.Sequential( nn.Linear(64, 1) )
        
    def forward(self, image, state, detach = False):
      s   = self.state_extractor(state)
      x   = torch.cat([image, s], -1)
      x   = self.nn_layer(x)

      if detach:
        return self.critic_layer(x).detach()
      else:
        return self.critic_layer(x)

class BasicContinous():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def sample(self, datas):
        mean, std = datas

        distribution    = Normal(0, 1)
        rand            = distribution.sample().float().to(set_device(self.use_gpu))
        return (mean + std * rand).squeeze(0)
        
    def entropy(self, datas):
        mean, std = datas
        
        distribution = Normal(mean, std)
        return distribution.entropy().float().to(set_device(self.use_gpu))
        
    def logprob(self, datas, value_data):
        mean, std = datas

        distribution = Normal(mean, std)
        return distribution.log_prob(value_data).float().to(set_device(self.use_gpu))

    def kldivergence(self, datas1, datas2):
        mean1, std1 = datas1
        mean2, std2 = datas2

        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)
        return kl_divergence(distribution1, distribution2).float().to(set_device(self.use_gpu))

    def deterministic(self, datas):
        mean, _ = datas
        return mean.squeeze(0)

class GeneralizedAdvantageEstimation():
    def __init__(self, gamma = 0.99):
        self.gamma  = gamma

    def compute_advantages(self, rewards, values, next_values, dones):
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values          
        for step in reversed(range(len(rewards))):  
            gae = delta[step] + (1.0 - dones[step]) * (1.0 - self.gamma) * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)

class CLR():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def compute_loss(self, first_encoded, second_encoded):
        indexes     = torch.arange(first_encoded.shape[0]).long().to(set_device(self.use_gpu))   
        
        similarity  = torch.nn.functional.cosine_similarity(first_encoded.unsqueeze(1), second_encoded.unsqueeze(0), dim = 2)
        return torch.nn.functional.cross_entropy(similarity, indexes)

class JointAuxPpg():
    def __init__(self, distribution):
        self.distribution       = distribution

    def compute_loss(self, action_datas, old_action_datas, values, returns):
        Kl                  = self.distribution.kldivergence(old_action_datas, action_datas).mean()
        auxppg_loss         = ((returns - values).pow(2) * 0.5).mean()

        return auxppg_loss + Kl

class TrulyPPO():
    def __init__(self, distribution, advantage_function, policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, vf_loss_coef = 1.0, entropy_coef = 0.01, gamma = 0.95):
        self.policy_kl_range    = policy_kl_range
        self.policy_params      = policy_params
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.advantage_function = advantage_function
        self.distribution       = distribution

    def compute_loss(self, action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones):
        advantages      = self.advantage_function.compute_advantages(rewards, values, next_values, dones)
        returns         = (advantages + values).detach()
        advantages      = ((advantages - advantages.mean()) / (advantages.std() + 1e-5)).detach()       

        logprobs        = self.distribution.logprob(action_datas, actions) + 1e-5
        old_logprobs    = (self.distribution.logprob(old_action_datas, actions) + 1e-5).detach()

        ratios          = (logprobs - old_logprobs).exp()       
        Kl              = self.distribution.kldivergence(old_action_datas, action_datas) + 1e-5

        pg_targets  = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * advantages - self.policy_params * Kl,
            ratios * advantages
        )
        pg_loss         = pg_targets.mean()
        dist_entropy    = self.distribution.entropy(action_datas).mean()

        if self.value_clip is None:
            critic_loss     = ((returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
            critic_loss     = ((returns - vpredclipped).pow(2) * 0.5).mean()

        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

class AuxPpgMemory(Dataset):
    def __init__(self, capacity = 100000):
        self.capacity   = capacity
        self.images     = []
        self.states     = []

        self.trans  = transforms.Compose([
            transforms.ToTensor()
        ])    

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        images  = self.trans(self.images[idx])
        return np.array(self.states[idx], dtype = np.float32), images.detach().cpu().numpy()

    def save_eps(self, state, image):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.images[0]

        self.states.append(state)
        self.images.append(image)

    def save_replace_all(self, states, images):
        self.clear_memory()
        self.save_all(states, images)

    def save_all(self, states, images):
        for state, image in zip(states, images):
            self.save_eps(state, image)

    def get_all_items(self):         
        return self.states, self.images

    def clear_memory(self):
        del self.states[:]
        del self.images[:]

class PolicyMemory(Dataset):
    def __init__(self, capacity = 100000):
        self.capacity       = capacity
        self.position       = 0

        self.states         = []
        self.images         = []
        self.actions        = []
        self.rewards        = []
        self.dones          = []
        self.next_states    = []
        self.next_images    = []

        self.trans  = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        images      = self.trans(self.images[idx])
        next_images = self.trans(self.next_images[idx])

        return np.array(self.states[idx], dtype = np.float32), images.detach().cpu().numpy(), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(self.next_states[idx], dtype = np.float32), next_images.detach().cpu().numpy()       

    def save_eps(self, state, image, action, reward, done, next_state, next_image):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.images[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]
            del self.next_images[0]

        self.states.append(state)
        self.images.append(image)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.next_images.append(next_image)

    def save_replace_all(self, states, images, actions, rewards, dones, next_states, next_images):
        self.clear_memory()
        self.save_all(states, images, actions, rewards, dones, next_states, next_images)

    def save_all(self, states, images, actions, rewards, dones, next_states, next_images):
        for state, image, action, reward, done, next_state, next_image in zip(states, images, actions, rewards, dones, next_states, next_images):
            self.save_eps(state, image, action, reward, done, next_state, next_image)

    def get_all_items(self):         
        return self.states, self.images, self.actions, self.rewards, self.dones, self.next_states, self.next_images

    def clear_memory(self):
        del self.states[:]
        del self.images[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        del self.next_images[:]

class ClrMemory(Dataset):
    def __init__(self, capacity = 10000, first_trans = None, second_trans = None):        
        self.images         = []
        self.capacity       = capacity
        self.first_trans    = first_trans
        self.second_trans   = second_trans

        if self.first_trans is None:
            self.first_trans = transforms.Compose([
                transforms.RandomCrop(270),
                transforms.Resize(320),
                transforms.GaussianBlur(3),
                transforms.ToTensor()
            ])

        if self.second_trans is None:
            self.second_trans = transforms.Compose([                
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
                transforms.RandomGrayscale(p = 0.2),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images          = self.images[idx] # torch.FloatTensor(self.images[idx])

        first_inputs    = self.first_trans(images)
        second_inputs   = self.second_trans(images)

        return (first_inputs.detach().cpu().numpy(), second_inputs.detach().cpu().numpy())

    def save_eps(self, image):
        if len(self) >= self.capacity:
            del self.images[0]

        self.images.append(image)

    def save_replace_all(self, images):
        self.clear_memory()

        for image in images:
            self.save_eps(image)

    def save_all(self, images):
        for image in images:
            self.save_eps(image)

    def get_all_items(self):         
        return self.images

    def clear_memory(self):
        del self.images[:]

class AgentPpgClr():  
    def __init__(self, Policy_Model, Value_Model, CnnModel, ProjectionModel, state_dim, action_dim, policy_dist, policy_loss, auxppg_loss, clr_loss, 
                policy_memory, auxppg_memory, clr_memory, PPO_epochs = 10, AuxPpg_epochs = 10, Clr_epochs = 10, n_auxppg_update = 10, 
                is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32,  learning_rate = 3e-4, folder = 'model', use_gpu = True):   

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batch_size         = batch_size  
        self.PPO_epochs         = PPO_epochs
        self.AuxPpg_epochs      = AuxPpg_epochs
        self.Clr_epochs         = Clr_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.n_auxppg_update    = n_auxppg_update

        self.device             = set_device(self.use_gpu)

        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu).float().to(self.device)
        self.policy_old         = Policy_Model(state_dim, action_dim, self.use_gpu).float().to(self.device)
        self.policy_cnn         = CnnModel().float().to(self.device)
        self.policy_cnn_old     = CnnModel().float().to(self.device)
        self.policy_projection  = ProjectionModel().float().to(self.device)

        self.value              = Value_Model(state_dim).float().to(self.device)
        self.value_old          = Value_Model(state_dim).float().to(self.device)
        self.value_cnn          = CnnModel().float().to(self.device)
        self.value_cnn_old      = CnnModel().float().to(self.device)
        self.value_projection   = ProjectionModel().float().to(self.device)

        self.policy_dist        = policy_dist

        self.policy_memory      = policy_memory
        self.auxppg_memory      = auxppg_memory
        self.clr_memory         = clr_memory
        
        self.policyLoss         = policy_loss
        self.auxppgLoss         = auxppg_loss
        self.clrLoss            = clr_loss
        
        self.i_auxppg_update    = 0
        self.i_ppo_update       = 0

        self.ppo_optimizer      = Adam(list(self.policy_cnn.parameters()) + list(self.policy.parameters()) + list(self.value_cnn.parameters()) + list(self.value.parameters()), lr = learning_rate)        
        self.auxppg_optimizer   = Adam(list(self.policy_cnn.parameters()) + list(self.policy.parameters()), lr = learning_rate)
        self.clr_optimizer      = Adam(list(self.policy_cnn.parameters()) + list(self.policy_projection.parameters()) + list(self.value_cnn.parameters()) + list(self.value_projection.parameters()), lr = learning_rate) 

        self.ppo_scaler         = torch.cuda.amp.GradScaler()
        self.auxppg_scaler      = torch.cuda.amp.GradScaler()
        self.clr_scaler         = torch.cuda.amp.GradScaler()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())
        self.value_cnn_old.load_state_dict(self.value_cnn.state_dict())       

        self.trans  = transforms.Compose([
            transforms.ToTensor()
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
            out1                = self.policy_cnn(images)
            action_datas, _     = self.policy(out1, states)

            out2                = self.value_cnn(images)
            values              = self.value(out2, states)

            out3                = self.policy_cnn_old(images, True)
            old_action_datas, _ = self.policy_old(out3, states, True)

            out4                = self.value_cnn_old(images, True)
            old_values          = self.value_old(out4, states, True)

            out5                = self.value_cnn(next_images, True)
            next_values         = self.value(out5, next_states, True)

            loss = self.policyLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        
        self.ppo_scaler.scale(loss).backward()
        self.ppo_scaler.step(self.ppo_optimizer)
        self.ppo_scaler.update()

    def __training_auxppg(self, states, images):        
        self.auxppg_optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            out1                    = self.policy_cnn(images)
            action_datas, values    = self.policy(out1, states)

            out2                    = self.value_cnn(images, True)
            returns                 = self.value(out2, states, True)

            out3                    = self.policy_cnn_old(images, True)
            old_action_datas, _     = self.policy_old(out3, states, True)

            loss = self.auxppgLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.auxppg_scaler.scale(loss).backward()
        self.auxppg_scaler.step(self.auxppg_optimizer)
        self.auxppg_scaler.update()

    def __training_clr(self, first_images, second_images):
        self.clr_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out1        = self.policy_cnn(first_images)
            encoded1    = self.policy_projection(out1)

            out2        = self.value_cnn(second_images)
            encoded2    = self.value_projection(out2)

            out3        = self.value_cnn(first_images)
            encoded3    = self.value_projection(out3)

            out4        = self.policy_cnn(second_images)
            encoded4    = self.policy_projection(out4)

            loss = self.clrLoss.compute_loss(encoded1, encoded2) + self.clrLoss.compute_loss(encoded3, encoded4)

        self.clr_scaler.scale(loss).backward()
        self.clr_scaler.step(self.clr_optimizer)
        self.clr_scaler.update()

    def __update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False, num_workers = 2)

        for _ in range(self.PPO_epochs):       
            for states, images, actions, rewards, dones, next_states, next_images in dataloader: 
                self.__training_ppo(to_tensor(states, use_gpu = self.use_gpu), to_tensor(images, use_gpu = self.use_gpu), actions.float().to(self.device), 
                    rewards.float().to(self.device), dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu), to_tensor(next_images, use_gpu = self.use_gpu))

        states, images, _, _, _, _, _ = self.policy_memory.get_all_items()
        self.auxppg_memory.save_all(states, images)
        self.policy_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())
        self.value_cnn_old.load_state_dict(self.value_cnn.state_dict())

    def __update_auxppg(self):
        dataloader  = DataLoader(self.auxppg_memory, self.batch_size, shuffle = False, num_workers = 2)

        for _ in range(self.AuxPpg_epochs):       
            for states, images in dataloader:
                self.__training_auxppg(to_tensor(states, use_gpu = self.use_gpu), to_tensor(images, use_gpu = self.use_gpu))

        states, images = self.auxppg_memory.get_all_items()
        self.clr_memory.save_all(images)
        self.auxppg_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())

    def __update_clr(self):
        dataloader  = DataLoader(self.clr_memory, self.batch_size, shuffle = True, num_workers = 2)

        for _ in range(self.Clr_epochs):
            for first_images, second_images in dataloader:
                self.__training_clr(to_tensor(first_images, use_gpu = self.use_gpu), to_tensor(second_images, use_gpu = self.use_gpu))

        self.clr_memory.clear_memory()
        
        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())
        self.value_cnn_old.load_state_dict(self.value_cnn.state_dict())

    def act(self, state, image):
        state, image        = to_tensor(state, use_gpu = self.use_gpu, first_unsqueeze = True, detach = True), to_tensor(self.trans(image), use_gpu = self.use_gpu, first_unsqueeze = True, detach = True)

        out1                = self.policy_cnn(image)
        action_datas, _     = self.policy(out1, state)
        
        if self.is_training_mode:
            action = self.policy_dist.sample(action_datas)
        else:
            action = self.policy_dist.deterministic(action_datas)
              
        return to_numpy(action)

    def save_memory(self, policy_memory):
        states, images, actions, rewards, dones, next_states, next_images = policy_memory.get_all_items()
        self.policy_memory.save_all(states, images, actions, rewards, dones, next_states, next_images)
        self.clr_memory.save_all(images)

    def update(self):
        self.__update_ppo()                
        self.i_auxppg_update += 1

        if self.i_auxppg_update % self.n_auxppg_update == 0 and self.i_auxppg_update != 0:
            self.__update_auxppg()
            self.__update_clr()
            self.i_auxppg_update = 0

    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_cnn_state_dict': self.policy_cnn.state_dict(),
            'value_cnn_state_dict': self.value_cnn.state_dict(),
            'policy_pro_state_dict': self.policy_projection.state_dict(),
            'value_pro_state_dict': self.value_projection.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'auxppg_optimizer_state_dict': self.auxppg_optimizer.state_dict(),
            'clr_optimizer_state_dict': self.clr_optimizer.state_dict(),
            'ppo_scaler_state_dict': self.ppo_scaler.state_dict(),
            'auxppg_scaler_state_dict': self.auxppg_scaler.state_dict(),
            'clr_scaler_state_dict': self.clr_scaler.state_dict()
            }, self.folder + '/model.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/model.tar', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.policy_cnn.load_state_dict(model_checkpoint['policy_cnn_state_dict'])        
        self.value_cnn.load_state_dict(model_checkpoint['value_cnn_state_dict'])
        self.policy_projection.load_state_dict(model_checkpoint['policy_pro_state_dict'])        
        self.value_projection.load_state_dict(model_checkpoint['value_pro_state_dict'])
        self.ppo_optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])        
        self.auxppg_optimizer.load_state_dict(model_checkpoint['auxppg_optimizer_state_dict'])
        self.clr_optimizer.load_state_dict(model_checkpoint['clr_optimizer_state_dict'])   
        self.ppo_scaler.load_state_dict(model_checkpoint['ppo_scaler_state_dict'])        
        self.auxppg_scaler.load_state_dict(model_checkpoint['auxppg_scaler_state_dict'])
        self.clr_scaler.load_state_dict(model_checkpoint['clr_scaler_state_dict'])     

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

class CarlaRunner():
    def __init__(self, agent, env, memory, training_mode, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100):
        self.env                = env
        self.agent              = agent
        self.render             = render
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0
        
        self.images, self.states    = self.env.reset()
        self.memories               = memory        

    def run(self):
        self.memories.clear_memory()       

        for _ in range(self.n_update):
            action                                      = self.agent.act(self.states, self.images)
            next_image, next_state, reward, done, _     = self.env.step(action)
            
            if self.training_mode:
                self.memories.save_eps(self.states.tolist(), self.images, action, reward, float(done), next_state.tolist(), next_image)
                
            self.images         = next_image
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                self.images, self.states    = self.env.reset()
                self.total_reward           = 0
                self.eps_time               = 0

        # print('Updating agent..')
        return self.memories

class Executor():
    def __init__(self, agent, n_iteration, runner, save_weights = False, n_saved = 10, load_weights = False, is_training_mode = True):
        self.agent              = agent
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.is_training_mode   = is_training_mode

        if load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

    def execute(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                memories  = self.runner.run()
                self.agent.save_memory(memories)

                if self.is_training_mode:
                    self.agent.update()

                    if self.save_weights:
                        if i_iteration % self.n_saved == 0:
                            self.agent.save_weights()
                            print('weights saved')

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            finish = time.time()
            timedelta = finish - start
            print('\nTimelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))

############## Hyperparameters ##############

load_weights            = True # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
render                  = False # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 1000000 # How many episode you want to run
n_memory_clr            = 10000
n_update                = 256 # How many episode before you update the Policy 
n_auxppg_update         = 2
n_saved                 = n_auxppg_update

policy_kl_range         = 0.03
policy_params           = 5
value_clip              = 5.0
entropy_coef            = 0.0
vf_loss_coef            = 1.0
batch_size              = 32
PPO_epochs              = 5
AuxPpg_epochs           = 5
Clr_epochs              = 4
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4

folder                  = 'weights/carla1'
env                     = CarlaEnv(im_height = 320, im_width = 320, im_preview = False, max_step = 512) # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim           = None
action_dim          = None
max_action          = 1

Policy_Model        = Policy_Model
Value_Model         = Value_Model
Cnn_Model           = CnnModel
ProjectionModel     = ProjectionModel
Policy_Dist         = BasicContinous
Runner              = CarlaRunner
Executor            = Executor
Policy_loss         = TrulyPPO
AuxPpg_loss         = JointAuxPpg
Clr_loss            = CLR
Wrapper             = env
Policy_Memory       = PolicyMemory
AuxPpg_Memory       = AuxPpgMemory
Clr_Memory          = ClrMemory
Advantage_Function  = GeneralizedAdvantageEstimation

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

if state_dim is None:
    state_dim = Wrapper.get_obs_dim()
print('state_dim: ', state_dim)

if Wrapper.is_discrete():
    print('discrete')
else:
    print('continous')

if action_dim is None:
    action_dim = Wrapper.get_action_dim()
print('action_dim: ', action_dim)

policy_dist         = Policy_Dist(use_gpu)
advantage_function  = Advantage_Function(gamma)
auxppg_memory       = AuxPpg_Memory()
policy_memory       = Policy_Memory()
runner_memory       = Policy_Memory()
clr_memory          = Clr_Memory(n_memory_clr)
auxppg_loss         = AuxPpg_loss(policy_dist)
policy_loss         = Policy_loss(policy_dist, advantage_function, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma)
clr_loss            = Clr_loss(use_gpu)

agent = AgentPpgClr( Policy_Model, Value_Model, CnnModel, ProjectionModel, state_dim, action_dim, policy_dist, policy_loss, auxppg_loss, clr_loss, 
                policy_memory, auxppg_memory, clr_memory, PPO_epochs, AuxPpg_epochs, Clr_epochs, n_auxppg_update, 
                is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, 
                batch_size,  learning_rate, folder, use_gpu)

# ray.init()
runner      = Runner(agent, Wrapper, runner_memory, is_training_mode, render, n_update, Wrapper.is_discrete, max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()