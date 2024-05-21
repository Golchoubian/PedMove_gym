from PedMove_gym.arguments import get_args
import numpy as np


class BaseConfig(object):
    def __init__(self):
        pass

class Config(object):

    args = get_args()
 
    # general configs for OpenAI gym env
    env = BaseConfig()
    # The following time_step shouldn't be changed
    # It is based on HBS dataset's fps
    env.time_step = 0.5 # 2 fps

    # config for reward function
    reward = BaseConfig()
    reward.success_reward = 10
    reward.collision_penalty = -20
    reward.veh_collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 1 # meter
    reward.discomfort_penalty_factor = 40 
    # smooth continuous transition to collision penality of -20 considering that the discomfort distance is 1.

    # config for simulation
    sim = BaseConfig()
    sim.max_human_num = 60  # the maximum number of pedestrians in the HBS scenarios is 60
    sim.max_vehicle_num = 15  # the maximum number of vehicles in the HBS scenarios is 15
    # the number of frames beyond the exsting frames in ground truth data for continuing the simulation
    sim.extended_time_out = 30 

    # 'none': no prediction involved
    # 'truth': ground truth future traj 
    # 'inferred': inferred future traj from UAW-PCG prediction model
    sim.predict_method = 'inferred'
    sim.predict_network = 'CollisionGrid' # 'VanillaLSTM' or 'CollisionGrid' (UAW-PCG)
    sim.render = True # render the simulation or not

    sim.obs_len = args.obs_length
    sim.pred_len = args.pred_length

   
    # action space of the robot
    action_space = BaseConfig()
    action_space.kinematics = 'unicycle'  # 'unicycle', 'holonomic', 'double_integrator_unicycle', 'bicycle'

    # robot config
    robot = BaseConfig()
    # whether to use the human driver actual trajectory in the dataset 
    # as an upper bound to navigtion algorithm performance
    robot.human_driver = True 
    robot.radius = 1 # meter
    robot.v_pref = 15/3.6  # 20 km/h is the maximun speed allowed in the shared space of the HBS dataset
    robot.max_steering_angle = 25 * np.pi/180 
    robot.L = 1.75 # polaris gem e2 wheelbase from documents
    robot.FOV = 2
    robot.sensor_range = 15 # radius of perception range

    # ped config
    peds = BaseConfig()
    peds.radius = 0.3 # meters
    peds.v_pref = 1

    # other veh config
    vehs = BaseConfig()
    vehs.radius = 0.7
    vehs.v_pref = 20/3.6  # 20 km/h is the maximun speed allowed in the shared space of the HBS dataset

    # config for data collection
    data = BaseConfig()
    data.pred_timestep = 0.5
    data.visual_save_path = 'Simulated_scenarios'