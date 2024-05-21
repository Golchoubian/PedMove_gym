import torch

class Agent(object):
    def __init__(self, config, section):
        '''
        Base class for robot (AV) and human.
        '''

        if section == 'robot':
            subconfig = config.robot
        elif section == 'ped':
            subconfig = config.peds
        elif section == 'veh':
            subconfig = config.vehs

        self.obs_length = config.sim.obs_len
        self.pred_length = config.sim.pred_len

        self.policy = None
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius

        self.px = None
        self.py = None
        self.vx = None
        self.vy = None
        self.gx = None
        self.gy = None
        self.theta = None
        self.time_step = config.env.time_step

        self.true_pose = None
        self.lookup = None
        self.agent_type = section

        self.curr_obs = None
        self.curr_mask = None   

    def set_scen_data(self, pos_scen, mask_scen):
        
        self.true_pose = pos_scen
        self.mask = mask_scen
    
        self.px = self.true_pose[self.obs_length-1,:,0] # for all pedestrians and all vehicles
        self.py = self.true_pose[self.obs_length-1,:,1] 

        self.vx = self.true_pose[self.obs_length-1,:,2] 
        self.vy = self.true_pose[self.obs_length-1,:,3] 


    def get_traj_history(self):
        return self.curr_obs, self.curr_mask
    
    def get_position(self):
        return [self.px, self.py]
    
    def get_velocity(self):
        return [self.vx, self.vy]
    
    def calculate_theta(self, vx, vy):
        '''
        calculatig the heading angle in radian
        given the velocity components
        '''
        theta = torch.atan2(vy, vx)
        theta = theta % (2 * torch.pi)
        return theta

    def filter_curr_ob(self, pos, mask):
        '''
        This function removes those columns in pos and mask
        that are assosicated to agents that are not present 
        in any of the frames during this current sequence length
        that we are looking at in the scenario 
        '''
        num_frame = pos.shape[0]
        # colums with value of zero are associated to those peds not availabe in this whole sequence
        num_avail_fram = torch.sum(mask, 0)
        columns_to_keep = (num_avail_fram != 0).nonzero()  # of shape (num_valid_columns, 1)
        # expanding this valid colum number to all time rows in the pos and mask (first dimension)
        columns_to_keep_rp = torch.transpose(columns_to_keep, 1, 0).repeat(num_frame, 1)
        mask_filter = torch.gather(mask, dim=1, index=columns_to_keep_rp)
        columns_to_keep_rp2 = columns_to_keep_rp.unsqueeze(2).repeat(1, 1, pos.shape[2])
        pos_filter = torch.gather(pos, dim=1, index=columns_to_keep_rp2)

        col_indexs_to_keep = columns_to_keep_rp[0, :]

        return pos_filter, mask_filter, col_indexs_to_keep
    
