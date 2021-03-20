import os
import numpy as np
import random
import torch
import ray
from torch.utils.tensorboard import SummaryWriter

from environment.carla import CarlaEnv
from model.carla.policy_model import PolicyModel
from model.carla.value_model import ValueModel
from model.carla.cnn_model import CnnModel
from model.carla.projection_model import ProjectionModel
from model.carla.q_model import QModel
from distribution.normal import NormalDist
from executor.multi_agent_central_learner.central_learner import CentralLearnerExecutor
from executor.multi_agent_central_learner.child import ChildExecutor
from runner.carla_runner import CarlaRunner
from loss.ppo.truly_ppo import TrulyPPO
from loss.cql.cql import Cql
from loss.cql.policy import OffPolicyLoss
from loss.cql.value import OffVLoss
from loss.other.clr import Clr
from loss.other.joint_aux_ppg import JointAuxPpg
from memory.aux_clr_memory import AuxClrMemory
from memory.aux_ppg_memory import AuxPpgMemory
from memory.policy_memory import PolicyMemory
from rl_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from agent.agent_ppg_clr import AgentPpgClr
from agent.agent_cql_clr import AgentCqlClr

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
render                  = False # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 1000000 # How many episode you want to run
n_memory_auxclr         = 10000
n_update                = 256 # How many episode before you update the Policy 
n_ppo_update            = 1
n_aux_update            = 2
n_saved                 = n_ppo_update * n_aux_update
n_agent                 = 2

policy_kl_range         = 0.03
policy_params           = 5
value_clip              = 20.0
entropy_coef            = 0.0
vf_loss_coef            = 1.0
batch_size              = 32
ppo_epochs              = 5
auxppg_epochs           = 5
auxclr_epochs           = 5
cql_epochs              = 5
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4

folder                  = 'weights/carla2'

state_dim           = None
action_dim          = None
max_action          = 1

Policy_Model        = PolicyModel
Value_Model         = ValueModel
Cnn_Model           = CnnModel
Q_Model             = QModel
Projection_Model    = ProjectionModel
Policy_Dist         = NormalDist
Runner              = CarlaRunner
Central_Executor    = CentralLearnerExecutor
Child_Executor      = ChildExecutor
Policy_loss         = TrulyPPO
AuxPpg_loss         = JointAuxPpg
AuxClr_loss         = Clr
Cql_loss            = Cql
OffValue_loss       = OffVLoss
OffPolicy_loss      = OffPolicyLoss
Policy_Memory       = PolicyMemory
AuxPpg_Memory       = AuxPpgMemory
AuxClr_Memory       = AuxClrMemory
Advantage_Function  = GeneralizedAdvantageEstimation
Agent               = AgentPpgClr

env                 = [CarlaEnv(im_height = 320, im_width = 320, im_preview = False, max_step = 512) for _ in range(n_agent)] # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]
Wrapper             = env
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

policy_dist         = [Policy_Dist(use_gpu) for _ in range(n_agent + 1)]
advantage_function  = [Advantage_Function(gamma) for _ in range(n_agent)]
auxppg_memory       = [AuxPpg_Memory() for _ in range(n_agent)]
policy_memory       = [Policy_Memory() for _ in range(n_agent + 1)]
runner_memory       = [Policy_Memory() for _ in range(n_agent)]
auxclr_memory       = [AuxClr_Memory(n_memory_auxclr) for _ in range(n_agent + 1)]
auxppg_loss         = [AuxPpg_loss(policy_dist[i]) for i in range(n_agent)]
policy_loss         = [Policy_loss(policy_dist[i], advantage_function[i], policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma) for i in range(n_agent)]
auxclr_loss         = [AuxClr_loss(use_gpu) for _ in range(n_agent + 1)]
cql_loss            = Cql_loss()
offvalue_loss       = OffValue_loss()
offpolicy_loss      = OffPolicy_loss()

agentPgg = [
    AgentPpgClr( Policy_Model, Value_Model, Cnn_Model, Projection_Model, state_dim, action_dim, policy_dist[i], policy_loss[i], auxppg_loss[i], auxclr_loss[i], policy_memory[i], 
            auxppg_memory[i], auxclr_memory[i], ppo_epochs, auxppg_epochs, auxclr_epochs, n_aux_update, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size,  learning_rate, folder, use_gpu)
        for i in range(n_agent)
]

agentCql = AgentCqlClr( Policy_Model, Value_Model, Q_Model, Cnn_Model, Projection_Model, state_dim, action_dim, policy_dist, cql_loss, offvalue_loss, offpolicy_loss, auxclr_loss[-1], 
        policy_memory[-1], auxclr_memory[-1], is_training_mode, batch_size, cql_epochs, auxclr_epochs, learning_rate, folder, use_gpu)

# ray.init()
runners = ray.put([
    Runner(agentPgg, Wrapper, runner_memory, is_training_mode, render, n_update, Wrapper.is_discrete, max_action, SummaryWriter(), n_plot_batch)
        for _ in range(n_agent)
])

child_executors     = [Child_Executor.remote(agentPgg[i], runners[i], i, load_weights, save_weights) for i in range(n_agent)]
central_executor    = Central_Executor(agentCql, n_iteration, child_executors, save_weights, n_saved)

central_executor.execute()