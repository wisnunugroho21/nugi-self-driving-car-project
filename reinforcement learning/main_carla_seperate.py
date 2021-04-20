import gym
import random
import numpy as np
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adam import Adam

from eps_runner.iteration.carla import CarlaRunner
from train_executor.executor import Executor
from agent.image_state.ppg.seperate_cnn import AgentImageStatePPG
from distribution.basic_continous import BasicContinous
from environment.custom.carla.carla_rgb import CarlaEnv
from loss.other.joint_aux import JointAux
from loss.ppo.truly_ppo import TrulyPPO
from policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from model.ppg.carla.cnn_model import CnnModel
from model.ppg.carla.policy_std_model import PolicyModel
from model.ppg.carla.value_model import ValueModel
from model.ppg.carla.projection_model import ProjectionModel
from memory.policy.image_state.standard import ImageStatePolicyMemory
from memory.aux_ppg.image_state.standard import ImageStateAuxPpgMemory

from helpers.pytorch_utils import set_device

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
render                  = False # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 1000000 # How many episode you want to run
n_update                = 256 # How many episode before you update the Policy 
n_aux_update            = 2
n_saved                 = n_aux_update

policy_kl_range         = 0.03
policy_params           = 5
value_clip              = 5.0
entropy_coef            = 1.0
vf_loss_coef            = 1.0
batch_size              = 32
ppo_epochs              = 5
aux_ppg_epochs          = 5
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4

folder                  = 'weights/carla1'
# env                     = gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim           = None
action_dim          = None
max_action          = 1

Cnn_Model           = CnnModel
Policy_Model        = PolicyModel
Value_Model         = ValueModel
Projection_Model    = ProjectionModel
Policy_Dist         = BasicContinous
Runner              = CarlaRunner
Executor            = Executor
Policy_loss         = TrulyPPO
Aux_loss            = JointAux
Wrapper             = CarlaEnv
Policy_Memory       = ImageStatePolicyMemory
Aux_Memory          = ImageStateAuxPpgMemory
Advantage_Function  = GeneralizedAdvantageEstimation
Agent               = AgentImageStatePPG

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

environment = Wrapper(im_height = 320, im_width = 320, im_preview = False, max_step = 512)

if state_dim is None:
    state_dim = environment.get_obs_dim()
print('state_dim: ', state_dim)

if environment.is_discrete():
    print('discrete')
else:
    print('continous')

if action_dim is None:
    action_dim = environment.get_action_dim()
print('action_dim: ', action_dim)

policy_dist         = Policy_Dist(use_gpu)
advantage_function  = Advantage_Function(gamma)
aux_ppg_memory      = Aux_Memory()
ppo_memory          = Policy_Memory()
runner_memory       = Policy_Memory()
aux_ppg_loss        = Aux_loss(policy_dist)
ppo_loss            = Policy_loss(policy_dist, advantage_function, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma)

policy              = Policy_Model(state_dim, action_dim, use_gpu).float().to(set_device(use_gpu))
value               = Value_Model(state_dim).float().to(set_device(use_gpu))

cnn_policy          = Cnn_Model().float().to(set_device(use_gpu))
projector_policy    = Projection_Model().float().to(set_device(use_gpu))

cnn_value           = Cnn_Model().float().to(set_device(use_gpu))
projector_value     = Projection_Model().float().to(set_device(use_gpu))

ppo_optimizer       = Adam(list(policy.parameters()) + list(value.parameters()) + list(cnn_policy.parameters()) + list(cnn_value.parameters()), lr = learning_rate)        
aux_ppg_optimizer   = Adam(list(policy.parameters()) + list(cnn_policy.parameters()), lr = learning_rate)

agent = Agent(cnn_policy, cnn_value, policy, value, state_dim, action_dim, policy_dist, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory,
            ppo_optimizer, aux_ppg_optimizer, ppo_epochs, aux_ppg_epochs, n_aux_update, is_training_mode, policy_kl_range, 
            policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size,  folder, use_gpu)

runner      = Runner(agent, environment, runner_memory, is_training_mode, render, n_update, environment.is_discrete(), max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()