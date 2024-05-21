import numpy as np
import torch
import gym
import os
from numpy.linalg import norm
from PedMove_gym.ped_pred.DataLoader import DataLoader
from PedMove_gym.ped_pred.pedestrian_trajectory import traj_prediction
from PedMove_gym.envs.utils.robot import Robot
from PedMove_gym.envs.utils.peds import Peds
from PedMove_gym.envs.utils.vehs import Vehs
from PedMove_gym.ped_pred.helper import getCoef, cov_mat_generation
from PedMove_gym.envs.utils.info import *


class PedSimPred(gym.Env):

    def __init__(self):

        self.frame = None

        self.robot = None
        self.peds = None
        self.vehs = None
        self.start_pos = None
        self.goal_pos = None
        self.scen_max_human_num = None

        self.config = None
        self.dataloader = None
        self.ped_traj_pred = None

        self.scenario_ind = None
        self.scenario_num = None
        self.scenario_length = None
        self.time_out_f = None
        self.extended_time_out = None

        self.observation_space = None
        self.action_space = None

        self.nenv = None  # the number of env will be set when the env is created.
        self.phase = None # the phase will be set when the env is created.
        self.test_case = None  # the test scenario number to be visualized.

        # for render
        self.render_axis = None
        self.render_figure = None
        self.epo_info = None

        self.robot_planned = None
        self.robot_fov = None # limit FOV
     
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None


    def reset(self, phase='train', test_case=None):
        """
        Reset the environment
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        self.scenario_ind += 1

        if self.scenario_ind == 0:
            # for the first step of training
            scen_data = self.dataloader.get_scenario(self.scenario_ind, first_time=True, 
                                                     test_case = test_case)

        elif self.scenario_ind >= self.dataloader.num_scenarios:
            '''
            When one round of training on all scenarios is done
            reset the cycle of scenario extraction with shuffle set to True
            we shuffle the scenarios only when starting the scenarios all over again
            '''
            self.scenario_ind = 0
            scen_data = self.dataloader.get_scenario(self.scenario_ind, shuffle=True)
        else:
            scen_data = self.dataloader.get_scenario(self.scenario_ind)

        (ego_data, EgoList, ped_data, PedsList, veh_data, VehsList, scenario_num) = scen_data

        self.scenario_num = scenario_num
        self.scenario_length = len(ego_data)
   
        pos_ego_scen, mask_ego_scen = self.dataloader.convert_proper_array(ego_data, 
                                                            EgoList, self.scenario_length)
        pos_ped_scen, mask_ped_scen = self.dataloader.convert_proper_array(ped_data,
                                                            PedsList, self.scenario_length)
        pos_veh_scen, mask_veh_scen = self.dataloader.convert_proper_array(veh_data,
                                                            VehsList, self.scenario_length)

        self.scen_max_human_num = pos_ped_scen.shape[1]
        self.scen_max_vehicle_num = pos_veh_scen.shape[1]

        # setting the goal position of the robot before extending the trajectories
        self.robot.gx = pos_ego_scen[-1, 0, 0]
        self.robot.gy = pos_ego_scen[-1, 0, 1]

        self.simulation_lenght = self.scenario_length - self.obs_length
        self.time_out_f = self.scenario_length + self.extended_time_out
        ext_len = self.extended_time_out + self.pred_length + 1 
        # +1 for geberate_ob to work in step function at the last extended scenario

        pos_veh_scen_ext, mask_veh_ext = self.dataloader.extend_traj(pos_veh_scen,
                                                                    mask_veh_scen, ext_len)
        # The extension for the ego will be used as a ground truth
        pos_ego_scen_ext, mask_ego_ext = self.dataloader.extend_traj(pos_ego_scen,
                                                                    mask_ego_scen, ext_len)

        if self.config.sim.predict_method == 'truth': 
            # providing ground truth prediction by extending the ped traj with constant velocity model
            pos_ped_scen, mask_ped_scen = self.dataloader.extend_traj(pos_ped_scen, 
                                                                    mask_ped_scen, ext_len)

        self.robot.set_scen_data(pos_ego_scen_ext, mask_ego_ext)
        self.peds.set_scen_data(pos_ped_scen, mask_ped_scen)
        self.vehs.set_scen_data(pos_veh_scen_ext, mask_veh_ext)

        self.start_pos = self.robot.get_position()
        self.goal_pos = self.robot.get_goal_position()

        # the simulation starts at frame = obs_lenght to have enoguht obervation at the start
        self.frame = self.obs_length - 1

        ob = {}
        ob['robot'] = self.robot.get_curr_ob(self.frame)
        ob['peds'] = self.peds.get_curr_ob(self.frame, reset=True)
        ob['vehs'] = self.vehs.get_curr_ob(self.frame)

        # initialize potential and angular potential
        rob_goal_vec = np.array([self.robot.gx, self.robot.gy]) - \
                         np.array([self.robot.px, self.robot.py])
        self.potential = -abs(np.linalg.norm(rob_goal_vec))

        # A robot plan is required for the prediction.
        # We use a constant velocity for this initial plan
        self.robot_planned = self.robot.planned_traj()
        
        if self.config.sim.predict_method != 'none':
            # Generate the pedestrians' predicted trajectories and 
            # update the relevant variable in peds class.
            self.ped_pred(ob)
    

        _ob = self.generate_ob(ob)

        return _ob 


    def configure(self, config):

        self.config = config

        self.obs_length = config.sim.obs_len
        self.pred_length = config.sim.pred_len
        self.seq_length = self.obs_length + self.pred_length

        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.veh_collision_penalty = config.reward.veh_collision_penalty

        self.robot_fov = np.pi * config.robot.FOV

        self.dataloader = DataLoader(phase=self.phase)
        self.scenario_ind = -1

        self.case_size = {self.phase: self.dataloader.num_scenarios}

        rob_RL = Robot(config, 'robot')
        self.set_robot(rob_RL)

        self.peds = Peds(config, 'ped')
        self.vehs = Vehs(config, 'veh')

        # we extend the scenarios from the datset to give the robot more time to reach the goal
        self.extended_time_out = self.config.sim.extended_time_out
        self.time_step = config.env.time_step

        self.ped_traj_pred = traj_prediction(config)

    
    def set_robot(self, robot):
        self.robot = robot

        d = {}
        # robot node: px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,),
                                                                dtype=np.float32)
        # temporal edges for the robot from time t-1 to t
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), 
                                                                dtype=np.float32)
       
        if self.config.sim.predict_method == 'inferred':
            # (dx, dy, cov11, cov12, cov21, cov22) of current and future time
            self.spatial_edge_dim = int((6*(self.pred_length+1))) 
        elif self.config.sim.predict_method == 'truth':
            self.spatial_edge_dim = int(2*(self.pred_length+1)) # (dx, dy) of current and future time
        else:
            self.spatial_edge_dim = 2 # (dx, dy) of current time

        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.max_human_num,
                                                    self.spatial_edge_dim), dtype=np.float32)

        # whether each human is visible to robot when considering limited FOV and sensor range
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.max_human_num,),
                                            dtype=np.bool)

        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,),
                                                                    dtype=np.float32)

        if self.config.args.consider_veh:
            d['spatial_edges_veh']  = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.max_vehicle_num, 2),
                                            dtype=np.float32)
            # number of vehicles detected at each timestep
            d['detected_vehicle_num'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                        shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

 

    def step(self, action):

        """
        step function
        Execute robto's actions,
        detect collision, update environment and return (ob, reward, done, info)
        """

        self.frame += 1

        # apply action and update the robot's state
        self.robot.step(action)

        ob = {}
    
        ob['robot'] = self.robot.get_curr_ob(self.frame)

        if self.config.robot.human_driver:
            '''
            To reproduce the exact human driver's trajectory in the dataset
            without any model inaccuracies intervening, we can directly pdate the
            robot's state to the human driver's ground truth next state. This is 
            solely for illustrative purposes and should be disabled when training 
            the robot policy by setting robot.human_driver to False in the config.py file.
            '''

            human_driver_next_pose = self.robot.true_pose[self.frame,:,:]
            # overwrting the updated pose through calling robot.step
            self.robot.px = human_driver_next_pose[0,0]
            self.robot.py = human_driver_next_pose[0,1]
            self.robot.vx = human_driver_next_pose[0,2]

            curr_obs = self.robot.true_pose[self.frame-self.obs_length+1:self.frame+1,:,:]
            self.robot.curr_obs = curr_obs
            curr_mask = self.robot.mask[self.frame-self.obs_length+1:self.frame+1,:]
            curr_ob = {'pos': curr_obs,
                    'mask': curr_mask}
            ob['robot'] = curr_ob


        # Update pedestrian states within the simulation environment by
        # replaying the real trajectories of pedestrians from the dataset.
        
        if (self.frame > self.scenario_length-1):
            # if we go beyond the length of the scenario in the dataset
            # (no more ground truth data) we rely on a constant velocity model pedestrians
            next_state = torch.zeros((1, self.scen_max_human_num ,5))
            next_state[0,:,0] = self.peds.true_pose[-1,:,0] +  \
                                self.peds.true_pose[-1,:,2] * self.time_step
            next_state[0,:,1] = self.peds.true_pose[-1,:,1] + \
                                self.peds.true_pose[-1,:,3] * self.time_step  
            next_state[0,:,2] = self.peds.true_pose[-1,:,2]
            next_state[0,:,3] = self.peds.true_pose[-1,:,3]
            # storing the next state from const vel as the ground turth  
            self.peds.true_pose = torch.cat((self.peds.true_pose, next_state),0) 
            self.peds.mask = torch.cat((self.peds.mask, 
                                    self.peds.mask[-1, :self.scen_max_human_num].unsqueeze(0)), 0)

        ob['peds'] = self.peds.get_curr_ob(self.frame)
        ob['vehs'] = self.vehs.get_curr_ob(self.frame)

        reward, done, episode_info = self.calc_reward(action)
        info = {'info': episode_info}

        # A robot plan is required for the prediction.
        # We use a constant velocity for this initial plan
        self.robot_planned = self.robot.planned_traj()
        
        if self.config.sim.predict_method != 'none':
            # Generate the pedestrians' predicted trajectories and 
            # update the relevant variable in peds class.
            self.ped_pred(ob)
    

        _ob = self.generate_ob(ob)

        return _ob, reward, done, info


    def ped_pred(self, ob):
        '''
        This should be called before making descision on the robot's action
        the data of the pedestrians predicted trajectory will be used for action decision
        '''

        pred_pos = torch.zeros((self.config.sim.pred_len, self.config.sim.max_human_num, 5))
        pred_cov = torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(self.config.sim.pred_len,
                                                                self.config.sim.max_human_num,1,1) 

        if self.config.sim.predict_method == 'truth':
            # return the ground truth prediction of the dataset. Used as an upper bound 
            pred_mask = torch.zeros((6, self.config.sim.max_human_num), dtype=torch.int)
            # with ground truth prediction, we should keep track of the maks over the prediction horizon

            GT_pred = self.peds.get_true_prediction(self.frame)
            pred_pos[:,:self.scen_max_human_num,:] = GT_pred['pos']
            pred_mask[:, :self.scen_max_human_num] = GT_pred['mask']

        else: # Generating predictions for pedestrains future trajecotry using UAW-PCG model

            pred_mask = torch.zeros((self.config.sim.max_human_num), dtype=torch.int)

            ob_ped_pos, ob_ped_mask, col_ind_pres_peds = self.peds.filter_curr_ob(ob['peds']['pos'],
                                                                                    ob['peds']['mask'])
            ob_veh_pos, ob_veh_mask, col_ind_pres_vehs = self.vehs.filter_curr_ob(ob['vehs']['pos'],
                                                                                    ob['vehs']['mask'])
            robot_pos, robot_plan = ob['robot']['pos'], self.robot_planned.clone()

            
            with torch.no_grad():
                # ped_pred: (pred_seq_len, num_peds, 5)
                ped_pred, dist_param = self.ped_traj_pred.forward(
                                                            ob_ped_pos.cpu(), ob_ped_mask.cpu(),
                                                            ob_veh_pos.cpu(), ob_veh_mask.cpu(),
                                                            robot_pos.cpu(), robot_plan.cpu(),
                                                            self.dataloader.timestamp)
                
                mux, muy, sx, sy, corr = getCoef(dist_param.cpu())
                scaled_param_dist = torch.stack((mux, muy, sx, sy, corr),2) 
                cov = cov_mat_generation(scaled_param_dist)

                pred_pos[:, col_ind_pres_peds, :] = ped_pred.cpu()
                pred_cov[:, col_ind_pres_peds, :, :] = cov.cpu()

                pred_mask[:self.scen_max_human_num] = ob['peds']['mask'][-1,:]
        
        
        # update the pedestrains predicted data
        self.peds.predicted_pos = pred_pos[:,:,:2]
        self.peds.predicted_cov = pred_cov
        self.peds.prediction_mask = pred_mask

        return True

    
    def get_human_visibility(self, ob, veh=False):
        '''
        returns a list of True/False values in the index position of pos tensor,
        True where the pedestrian is present in the robot's FOV (visible to the robot).
        '''

        mask = ob['mask']

        if veh:
            pres_time_mask = mask[self.obs_length-1, :]  # just looking at the current step, 
            # for vehicle the mask is of size (max_vehicle_num, obs_len+pred_len) 
            complete_visib_mask = torch.zeros((1, self.config.sim.max_vehicle_num),
                                                dtype=torch.int8)
            neighbor_agent = ob['pos'][self.obs_length-1,:,:]
            neigbor_radius = self.vehs.radius
            agent_num = self.scen_max_vehicle_num
        else:
            pres_time_mask = mask[-1, :]  # just looking at the current step
            # for pedestrian the mask is of size (max_ped_num, obs_len)
            complete_visib_mask = torch.zeros((1, self.config.sim.max_human_num), 
                                                dtype=torch.int8)
            neighbor_agent = ob['pos'][-1,:,:]
            neigbor_radius = self.peds.radius
            agent_num = self.scen_max_human_num
        
        for i in range(agent_num):
            if pres_time_mask[i]:
                complete_visib_mask[0, i] = int(self.detect_visible(self.robot,
                                                                    neighbor_agent[i,:], 
                                                                    neigbor_radius))

        neighbor_visib_ind = complete_visib_mask.bool().squeeze().tolist()
        num_visib_neighbor = torch.sum(complete_visib_mask).numpy()
       

        return neighbor_visib_ind, num_visib_neighbor
    
  
    def detect_visible(self, state1, state2, neigbor_radius, custom_fov=None,
                        custom_sensor_range=None):
        '''
        # Caculate whether agent2 is in agent1's FOV
        # Not the same as whether agent1 is in agent2's FOV!
        # arguments:
        # state1; robots state (class of robot)
        # state2; other agents state (in form of pos tensor)
        # return value:
        # return True if state2 is visible to state1, else return False
        '''
      
        real_theta = state1.theta

        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2[0] - state1.px, state2[1] - state1.py]
        
        # angle between center of FOV and agent 2
        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)
        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))

        if custom_fov:
            fov = custom_fov
        else:
            fov = self.robot_fov
          
        if np.abs(offset) <= fov / 2:
            inFov = True
        else:
            inFov = False

        # detect whether state2 is in state1's sensor_range
        dist = np.linalg.norm(
                [state1.px - state2[0], state1.py - state2[1]]) - neigbor_radius - state1.radius
        if custom_sensor_range:
            inSensorRange = dist <= custom_sensor_range
        else:
            inSensorRange = dist <= self.robot.sensor_range
           
        return (inFov and inSensorRange)
    

    def generate_ob(self, ob, sort=False):
        '''
        Generate observation for reset and step functions
        '''
        
        _ob = {}
        parent_ob = {}

        _ob['robot_node'] = np.array(self.robot.get_full_state_list_noV())

        _ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        if self.config.sim.predict_method == 'none':
            third_dim = 2
        else:
            # for 'inferred: 6 (x, y, cov11, cov12, cov21, covv22) at each timestep
            # for 'truth' : 2 (x, y) at each timestep
            third_dim = int(self.spatial_edge_dim / (self.pred_length+1)) 

        all_spatial_edges = np.ones((self.config.sim.max_human_num, third_dim)) * np.inf

        human_visibility, num_available_human = self.get_human_visibility(ob['peds'])

        for i in range(self.scen_max_human_num):
            if human_visibility[i]:
                relative_pos = np.array([ob['peds']['pos'][-1, i, 0] - self.robot.px, 
                                            ob['peds']['pos'][-1, i, 1] - self.robot.py])
                all_spatial_edges[i, :2] = relative_pos

                if self.config.sim.predict_method == 'inferred':
                    # arbitrary covariance matrix for the current time step (not using it in the model)
                    all_spatial_edges[i, 2:] = np.eye(2).flatten() 

        _ob['visible_masks'] = np.zeros(self.config.sim.max_human_num, dtype=np.bool)

        parent_ob['spatial_edges'] = all_spatial_edges
        _ob['visible_masks'][:self.scen_max_human_num] = human_visibility[:self.scen_max_human_num]

        constant_value = 5 * self.robot.sensor_range
        parent_ob['spatial_edges'][np.isinf(parent_ob['spatial_edges'])] = constant_value

        _ob['detected_human_num'] = num_available_human
        
        if self.config.sim.predict_method != 'none':
            # Add the prediction of pedestrians trajectory to the spatial edge
            _ob['spatial_edges'] = np.tile(parent_ob['spatial_edges'], self.pred_length+1)
            _ob = self.augment_prediction(_ob)
        else:
            _ob['spatial_edges'] = parent_ob['spatial_edges']


        if sort:
            # sort all humans by distance to robot
            hr_dist_cur = np.linalg.norm(_ob['spatial_edges'][:, :2], axis=-1)
            sorted_idx = np.argsort(hr_dist_cur, axis=0)
            _ob['spatial_edges'] = _ob['spatial_edges'][sorted_idx]
            # update the visiblity_mask after sorting
            _ob['visible_masks'] = np.zeros(self.config.sim.max_human_num, dtype=np.bool)
            if num_available_human > 0:
                _ob['visible_masks'][:num_available_human] = True


        # vehicle spatial edges
        if self.config.args.consider_veh:
            all_spatial_edges_veh = np.ones((self.config.sim.max_vehicle_num, 2)) * np.inf
            vehicle_availability, num_available_vehicle = self.get_human_visibility(ob['vehs'],
                                                                                    veh=True)

            for i in range(self.scen_max_vehicle_num):
                if vehicle_availability[i]:
                    relative_pos_veh = np.array(
                                    [ob['vehs']['pos'][self.obs_length-1, i, 0] - self.robot.px,
                                      ob['vehs']['pos'][self.obs_length-1, i, 1] - self.robot.py])
                    all_spatial_edges_veh[i, :2] = relative_pos_veh
            # I will alway sort vehicles spatial edges since for vehciles we never do prediction 
            _ob['spatial_edges_veh'] = np.array(sorted(all_spatial_edges_veh,
                                                        key=lambda x: np.linalg.norm(x)))
            _ob['spatial_edges_veh'][np.isinf(_ob['spatial_edges_veh'])] = constant_value

            _ob['detected_vehicle_num'] = num_available_vehicle
            # if no vehicle is detected, assume there is one dummy vehicle
            if _ob['detected_vehicle_num'] == 0:
                _ob['detected_vehicle_num'] = np.array([1])

        _ob = self.covnert_to_torch(_ob)

        return _ob
    
    
    def augment_prediction(self, _ob):
        '''
        Fill out the prediction part of the spatial edges
        The prediction data contains the x,y position of the pedestrians
        over the prediction horizon
        In 'truth' case, this will be the ground truth future position data
        In 'inferred' case, the prediction will also include the covaraince matrices
        of the pedestrians predicted trajecotry over the prediction length

        '''

        max_human_num = self.config.sim.max_human_num
        visibility_mask = np.expand_dims(_ob['visible_masks'], axis=1)
        third_dim = int(self.spatial_edge_dim / (self.pred_length+1))
        visibility_mask = visibility_mask.repeat(self.pred_length * third_dim, axis=1)
        pred_spatial_edges = self.peds.predicted_pos.permute(1,0,2)
        pred_spatial_edges = pred_spatial_edges - \
                                     torch.tensor([self.robot.px, self.robot.py]) # relative position

        if self.config.sim.predict_method == 'inferred':
            
            pred_spatial_edges_cov = self.peds.predicted_cov.permute(1,0,2,3)
            pred_spatial_edges_cov = pred_spatial_edges_cov.reshape(self.config.sim.max_human_num,
                                                                    self.pred_length,4)
            pred_spatial_edges =  torch.cat((pred_spatial_edges, pred_spatial_edges_cov), dim=2)

        
        pred_spatial_edges = pred_spatial_edges.reshape(max_human_num, -1)
        _ob['spatial_edges'][:, third_dim:][visibility_mask] = pred_spatial_edges[visibility_mask] 

        return _ob



    def calc_reward(self, action):
        
        # collision detection
        dmin = float('inf')
        speedAtdmin = float('inf')

        danger_dists = []
        collision = False

        # collision check with humans
        for i in range(self.scen_max_human_num):
            if self.peds.curr_mask[-1, i] != 0:  # The pedestrian is present at this time step
                dx = self.peds.px[i] - self.robot.px
                dy = self.peds.py[i] - self.robot.py
                closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - \
                                    self.peds.radius - self.robot.radius
                closest_dist = closest_dist.numpy()

                if closest_dist < self.discomfort_dist:
                    danger_dists.append(closest_dist)
                if closest_dist < 0:
                    collision = True
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist
                    speedAtdmin = self.robot.v
        
        # collision check with vehicles
        if self.config.args.consider_veh:
            collision_veh = False
            veh_closest_dist = []
            for i in range(self.scen_max_vehicle_num):
                if self.vehs.curr_mask[self.obs_length-1, i] != 0:  # The vehicle is present 
                    dx_veh = self.vehs.px[i] - self.robot.px
                    dy_veh = self.vehs.py[i] - self.robot.py
                    closest_dist_veh = (dx_veh ** 2 + dy_veh ** 2) ** (1 / 2) - self.vehs.radius - self.robot.radius
                    closest_dist_veh = closest_dist_veh.numpy()
                    veh_closest_dist.append(closest_dist_veh)

                    if closest_dist_veh < 0: 
                        # closest_dist_veh threshold value should be ideally different
                        # for vehicles in the same line and those in different lines but
                        # lets keep it simple for now by reducing the safety zone to zero 
                        # to not detected conflicts between cars in different lane
                        collision_veh = True
                        break

        # check if reaching the goal
        goal_radius = 2 
        reaching_goal = norm(np.array(self.robot.get_position()) - 
                             np.array(self.robot.get_goal_position())) < goal_radius

        danger_cond = dmin < self.discomfort_dist
        min_danger_dist = dmin

        if ((self.frame) >= self.time_out_f):
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            episode_info = Collision(self.robot.v)
        elif self.config.args.consider_veh and collision_veh:
            reward = self.veh_collision_penalty
            done = True
            episode_info = Collision_Vehicle()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()

        elif danger_cond:
            # only penalize agent for getting too close if it's visible
            reward = (dmin-self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(min_danger_dist, speedAtdmin)

        else:
            # potential reward
            pot_factor = 1 
            potential_cur = np.linalg.norm(np.array([self.robot.px, self.robot.py]) - \
                                           np.array(self.robot.get_goal_position()))
            reward = pot_factor * (-abs(potential_cur) - self.potential)
            self.potential = -abs(potential_cur)

            done = False
            episode_info = Nothing()

        if self.robot.kinematics == 'unicycle':
            # add a rotational penalty
            r_spin = -200 * action.r ** 2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.

            reward = reward + r_spin + r_back
        
        if self.robot.kinematics in {'double_integrator_unicycle', 'bicycle'}:
            # add a rotational penalty
            if self.robot.kinematics == 'double_integrator_unicycle':
                r_spin = -200 * action.u_alpha ** 2  # the coefficient should be adjusted
                reward = reward + r_spin
            elif self.robot.kinematics == 'bicycle':
                r_spin_action = -2000 * action.steering_change_rate ** 2
                r_spin = -10 * self.robot.phi ** 2
                reward = reward + r_spin_action + r_spin
            
        if self.phase == 'test':  # there is only one env to render
            self.epo_info = episode_info

        return reward, done, episode_info
    

    def covnert_to_torch(self, ob, device ='cpu'):
        # covert to pytorch tensor
        if isinstance(ob, dict):
            for key in ob:
                ob[key] = torch.from_numpy(ob[key]).to(device)
        else:
            ob = torch.from_numpy(ob).float().to(device)
        return ob


    def talk2Env(self, data):
        """
        Call this function when you want extra information to send to/recive from the env
        """       
        return True


    def render(self):

        from matplotlib import pyplot as plt

        ax = self.render_axis
        fig = self.render_figure

        fig.suptitle(f'Scenario #{self.scenario_num}', fontsize=15)
        ax.cla()
        ax.set_xlim(15, 95)  # for HBS
        ax.set_ylim(5, 60)  # for HBS
        ax.set_xlabel('x(m)', fontsize=15)
        ax.set_ylabel('y(m)', fontsize=15)

        Robot_obs_traj, _ = self.robot.get_traj_history()
        Peds_pos_obs, Peds_mask_obs = self.peds.get_traj_history()
        other_Vehs_traj, Vehs_mask = self.vehs.get_traj()
       
        if self.config.sim.predict_method != 'none':
            ped_pred_traj = self.peds.predicted_pos.permute(1,0,2)
            ped_pred_cov = self.peds.predicted_cov.permute(1,0,2,3)
            ped_pred_mask = self.peds.prediction_mask
    
        planned_traj = self.robot_planned
        ego_traj = torch.cat((Robot_obs_traj, planned_traj), 0)

        num_peds = Peds_pos_obs.shape[1]
        num_vehs = other_Vehs_traj.shape[1]

        max_marker_size = 6
        min_marker_size = 2
        max_marker_size_ego = 10
        min_alpha = 0.2

        for ped_ind in range(num_peds):
            PedPresFrame = []
            for t in range(self.obs_length):
                if Peds_mask_obs[t, ped_ind] == 1:
                    PedPresFrame.append(t)
                    # plotting the observed trajectory for this time step
                    marker_size = min_marker_size + \
                                    ((max_marker_size-min_marker_size)/self.seq_length * t)
                    alpha_val = min_alpha + ((1-min_alpha)/self.obs_length * t)
                    ax.plot(Peds_pos_obs[t, ped_ind, 0], Peds_pos_obs[t, ped_ind, 1], c='r',
                            marker='o', markersize=marker_size, alpha=alpha_val)
            
            # label = ped_ind
            # ax.annotate(label, # this is the text
            #     (Peds_pos_obs[self.obs_length-1, ped_ind,0], Peds_pos_obs[self.obs_length-1, ped_ind,1]), # these are the coordinates to position the label
            #     textcoords="offset points",
            #     xytext=(0,1), # distance from text to points (x,y)
            #     ha='center', fontsize=10) # horizontal alignment can be left, right or center
            
            # plotting the raduis and the personall space of the pedestrian at its current position
            if Peds_mask_obs[self.obs_length-1, ped_ind] == 1:
                PedRadius = plt.Circle((Peds_pos_obs[self.obs_length-1, ped_ind, 0],
                                         Peds_pos_obs[self.obs_length-1, ped_ind, 1]),
                                         self.peds.radius ,fill = False, ec='r',
                                         linestyle='--', linewidth=0.5)
                ax.add_artist(PedRadius)

                # static PS
                PS_static = plt.Circle((Peds_pos_obs[self.obs_length-1, ped_ind, 0],
                                         Peds_pos_obs[self.obs_length-1, ped_ind, 1]),
                                         self.discomfort_dist + self.peds.radius ,
                                         fill = False, ec='y', linestyle='--', linewidth=0.5)
                ax.add_artist(PS_static)
        
        # plot predicted pedestrian positions
        if self.config.sim.predict_method == 'inferred':
            for ped_ind in range(num_peds):
                if ped_pred_mask[ped_ind] == 1:
                    ax.plot(ped_pred_traj[ped_ind, :, 0], ped_pred_traj[ped_ind, :, 1], c='y',
                            marker='o', markersize=marker_size, alpha=alpha_val)
                    # plot the covariance of the predicted positions
                    for t in range(self.pred_length):
                        mean = ped_pred_traj[ped_ind, t, :2]
                        cov = ped_pred_cov[ped_ind, t, :]
                        self.plot_bivariate_gaussian3(mean, cov, ax, 1)

        elif self.config.sim.predict_method == 'truth':
            for ped_ind in range(num_peds):
                for t in range(self.pred_length):
                    if ped_pred_mask[t, ped_ind] == 1:
                        ax.plot(ped_pred_traj[ped_ind, :, 0], ped_pred_traj[ped_ind, :, 1], c='y',
                            marker='o', markersize=marker_size, alpha=alpha_val)


        if self.config.args.consider_veh:
            # plotting the trajectory of other vehicles
            for veh_ind in range(num_vehs):
                VehPresFrame = []
                for t in range(self.obs_length):
                    # plotting the trajecotry up to this time
                    marker_size = min_marker_size + \
                                ((max_marker_size-min_marker_size)/self.obs_length * t)
                    alpha_val = min_alpha + ((1-min_alpha)/self.obs_length * t)
                    if Vehs_mask[t, veh_ind] == 1:
                        VehPresFrame.append(t)
                        ax.plot(other_Vehs_traj[t, veh_ind, 0], other_Vehs_traj[t, veh_ind, 1],
                                c='c', marker='o', markersize=marker_size, alpha=alpha_val)
                ax.plot(other_Vehs_traj[VehPresFrame, veh_ind, 0],
                        other_Vehs_traj[VehPresFrame, veh_ind, 1], c='c', linewidth=1.0, alpha=0.5)

        # plotting the trajecotry of ego
        for t in range(self.seq_length):
            if t < self.obs_length:
                marker_size_ego = min_marker_size + \
                                    ((max_marker_size_ego-min_marker_size)/self.obs_length * t)
                alpha_val = min_alpha + ((1-min_alpha)/self.obs_length * t)
                ax.plot(ego_traj[t, 0, 0], ego_traj[t, 0, 1], c='k',
                        marker='o', markersize=marker_size_ego, alpha=alpha_val)
            else:
                ax.plot(ego_traj[t, 0, 0], ego_traj[t, 0, 1], c='g',
                        marker='o', markersize=marker_size_ego, alpha=alpha_val)
        ax.plot(ego_traj[:self.obs_length, 0, 0], ego_traj[:self.obs_length, 0, 1],
                 c='k', linewidth=1.0, alpha=0.5)
        ax.plot(ego_traj[self.obs_length:, 0, 0], ego_traj[self.obs_length:, 0, 1], 
                 c='g', linewidth=1.0, alpha=0.5)

        # drawing the raduis of the robot
        RobotRadius = plt.Circle((ego_traj[self.obs_length-1, 0, 0], 
                                  ego_traj[self.obs_length-1, 0, 1]),
                                  self.robot.radius, fill = False, 
                                  ec='k', linestyle='--', linewidth=0.5)
        ax.add_artist(RobotRadius)

        # drawing the visibility range of the robot
        visibility_circle = plt.Circle((ego_traj[self.obs_length-1, 0, 0],
                                         ego_traj[self.obs_length-1, 0, 1]),
                                         self.robot.sensor_range, fill = False,
                                         ec='k', linestyle='--', linewidth=0.5)
        ax.add_artist(visibility_circle)

        # Indicating the start and goal position of the ego for this scenario
        ax.plot(self.start_pos[0], self.start_pos[1], c='m', marker='P', markersize=15)
        ax.plot(self.goal_pos[0], self.goal_pos[1], c='m', marker='*', markersize=20)

        # legends
        ax.plot(-100, -100, c='k', marker='o', label='AV traj')
        ax.plot(-100, -100, c='g', marker='o', label='AV planned traj')
        ax.plot(-100, -100, c='r', marker='o', label='Ped observed traj')
        ax.plot(-100, -100, c='y', marker='o', label='Ped predicted traj')
        ax.plot(-100, -100, c='b', ls='-', label='Predicted traj $1\sigma$ std')
        if self.config.args.consider_veh:
            ax.plot(-100, -100, c='c', marker='o', label='Other veh current traj')
        ax.scatter(-100, -100, c='m', marker='P', label='Start')
        ax.scatter(-100, -100, c='m', marker='*', label='Goal')
        legend = ax.legend(loc="upper left", prop={'size': 9}, ncol=1)
        legend.legendHandles[-1]._sizes = [120]
        legend.legendHandles[-2]._sizes = [110]

        ax.text(78, 8, 'frame: '+ str(self.frame), fontsize=14, color='0.3')

        # Writing the episode info on the plot
        if self.epo_info.__str__() == 'Collision':
            c = 'red'
        elif self.epo_info.__str__() == 'Vehicle Collision':
            c = 'red'
        elif self.epo_info.__str__() == 'Reaching goal':
            c = 'green'
        elif self.epo_info.__str__() == 'Timeout':
            c = 'blue'
        elif self.epo_info.__str__() == 'Intrusion':
            c = 'orange'
        else:
            c = 'black'

        if len(self.epo_info.__str__()) != 0 and self.epo_info.__str__() != 'None':
            ax.text(40, 8, self.epo_info.__str__(), fontsize=20, color=c)
            plt.pause(1)

        plt.pause(0.2)
        save_path = os.path.join(self.config.data.visual_save_path, 'plots')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'img_{self.frame}.png')
        plt.savefig(save_path)

    def plot_bivariate_gaussian3(self, mean, cov, ax, max_nstd=3, c='b'):
        
        from matplotlib.patches import Ellipse

        vals, vecs = self.eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

        for j in range(1, max_nstd+1):
            # Width and height are "full" widths, not radius
            width, height = 2 * j * np.sqrt(vals)
            ellip = Ellipse(xy=mean, width=width, height=height,
                             angle=theta, edgecolor=c, fill=False)
            ax.add_artist(ellip)
 
        return ellip

    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]