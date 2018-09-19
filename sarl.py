import torch
import configparser
import numpy as np
from dynav.policy.sarl import SARL
from gym_crowd.envs.utils.state import ObservableState, FullState, JointState


def predict(input_state, b):
    policy = SARL()
    policy_config = configparser.RawConfigParser()
    policy_config.read('policy.config')
    policy.configure(policy_config)
    policy.set_device(torch.device('cpu'))
    policy.set_phase('test')
    policy.time_step = 0.25
    policy.model.load_state_dict(torch.load('data/sarl_circle_5p_visible/rl_model.pth'))

    state = JointState(FullState(0, -4, 0, 1, 0.3, 0, 4, 1, 1.7), [ObservableState(-2, 2, 0, 1, 0.3)])
    action = policy.predict(state)
    
    v = np.linalg.norm(action)
    rot = np.arctan2(action.vy, action.vx) - np.arctan2(state.self_state.vy, state.self_state.vx)
    
    return v, rot

