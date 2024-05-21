
import torch
import numpy as np
from PedMove_gym.ped_pred.model_CollisionGrid import CollisionGridModel
from PedMove_gym.ped_pred.model_VanillaLSTM import VLSTMModel


def position_change(x_seq, avail_mask):
    #substract each frame value from its previous frame to create displacement data.

    num_peds = x_seq.shape[1]
    vectorized_x_seq = x_seq.clone()
    
    # find the row index of the first frame each pedestrian apears in
    first_frame_apearing = torch.argmax(avail_mask.to(torch.int), dim=0) # shape: (num_ped)

    # Find pedestrains that exist in at least one frame of this sequence
    ExistOrNot = torch.sum(avail_mask, dim=0)
    existing_ped_indexes = torch.nonzero(ExistOrNot).squeeze(-1)

    first_values_dict = dict(zip(existing_ped_indexes.tolist(),
                                  x_seq[first_frame_apearing[existing_ped_indexes].tolist(),
                                         existing_ped_indexes.tolist(), :2]))
    
    vectorized_x_seq[1:,:,:2] = x_seq[1:,:,:2] - x_seq[:-1,:,:2]
    # the first frame of each pedestrian is (0,0)
    vectorized_x_seq[first_frame_apearing[existing_ped_indexes].tolist(),
                               existing_ped_indexes.tolist(), :2] = torch.zeros(len(existing_ped_indexes), 2)
    
    # make sure when mask is zero, the displacement is also zero  
    avail_mask_rp = avail_mask.unsqueeze(2).repeat(1,1,x_seq.shape[2]).float()
    vectorized_x_seq = vectorized_x_seq * avail_mask_rp
    
    return vectorized_x_seq, first_values_dict


def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def get_model(index, arguments, infer = False):
    # return a model given index and arguments
   
    if index == 4:
        return CollisionGridModel(arguments, infer)
    elif index == 3:
        return VLSTMModel(arguments, infer)
    

def sample_gaussian_2d(mux, muy, sx, sy, corr, mask):
    '''
    Parameters
    ==========

    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation

    mask : a tensor of zero and ones determining which ped is present
           based on position index in the actual tensor of position or mux,..

    Returns
    =======

    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    # converted_node_present = [look_up[node] for node in nodesPresent]
    converted_node_present = [i for i in range(numNodes) if mask[i]==1]
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], 
                [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y # The value of next_x and next_y for those agent not present in this frame is zero


def revert_postion_change(x_seq, avail_mask, first_values_dict, orig_x_seq, obs_length, infer=False, use_cuda=False):
    
    num_peds = x_seq.shape[1]
    absolute_x_seq = x_seq.clone()
    # prepare the first position values of all pedestrians in a tensor format
    # first_values_list = [first_values_dict[i] if i in first_values_dict.keys() else torch.zeros(2) for i in range(num_peds)]
    first_values_list = [first_values_dict[key] for key in sorted(first_values_dict.keys())]
    first_values = torch.stack(first_values_list) # shape: (num_ped, 2)

    # observed part:
    absolute_x_seq[1:obs_length,:,:2] = x_seq[1:obs_length,:,:2] + orig_x_seq[:obs_length-1,:,:2] 
    # add the first value dict for each pedestrian to the column it first apears in
    # find the column index of the first frame each pedestrian apears in
    first_frame_apearing = torch.argmax(avail_mask.to(torch.int), dim=0) # shape: (num_ped)
    absolute_x_seq[first_frame_apearing, range(num_peds), :2] = first_values

    # prediction part
    if len(x_seq)>obs_length: # if we have prediction part
        if infer==True:
            absolute_x_seq[obs_length,:,:2] = absolute_x_seq[obs_length,:,:2] + orig_x_seq[obs_length-1,:,:2] # using the last observed pos
            absolute_x_seq[obs_length:,:,:2] = torch.cumsum(absolute_x_seq[obs_length:,:,:2], dim=0)
        else:
            absolute_x_seq[obs_length:,:,:2] = absolute_x_seq[obs_length:,:,:2] + orig_x_seq[obs_length-1:-1,:,:2] 
  
    avail_mask_rp = avail_mask.unsqueeze(2).repeat(1,1,x_seq.shape[2])
    absolute_x_seq = absolute_x_seq * avail_mask_rp.float()

    return absolute_x_seq
  

def KF_covariance_generator(x_seq, mask, dt, plot_bivariate_gaussian3=None, ax2=None,
                            x_orig=None, Pedlist=None, lookup=None, use_cuda=None,
                            first_values_dict=None, obs_length=None):

    '''
    This function is used to generate a distribution around each
    ground truth data using Kalman Filter.
    It ouptus the filtered states and the covariance matrix of the distribution
    Outputs:
    filtered_states: tensor of shape (seq_length, num_peds, 2)
    filtered_covariances: tensor of shape (seq_length, num_peds, 2, 2)
    '''
    # Note: for now we are not using the mask data,
    # considering that the data of all the peds are available in each frame !

    # parameters to adjust later
    process_noise_std = 0.5
    measurement_noise_std = 0.7
    
    # this should be done before making the x_seq as position change
    seq_length = x_seq.shape[0]
    num_peds = x_seq.shape[1]

    # Initial state and covariance for each ped
    initial_state = x_seq[0,:,:4]
    initial_covariance = torch.tile(torch.eye(4), (num_peds, 1, 1))

    # Process and measurement noise covariance matrices
    process_noise = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(num_peds, 1, 1) * process_noise_std**2
    measurement_noise = torch.eye(2, dtype=torch.float32).view(1, 2, 2).repeat(num_peds, 1, 1) * measurement_noise_std**2

    # State transition matrix and measurement matrix
    F = torch.tensor([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32).view(1, 4, 4).repeat(num_peds, 1, 1)
    H = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32).view(1, 2, 4).repeat(num_peds, 1, 1)

    measurements = x_seq[1:, :, :2] # only position data is used for measurement

    # Run Kalman filter
    filtered_states, filtered_covariances = kalman_filter(
        initial_state, initial_covariance, measurements, measurement_noise, process_noise, F, H)
    
    return filtered_states[:,:,:2], filtered_covariances[:,:,:2,:2] # only passing the position covariance matrix



def kalman_filter(x, P, measurements, R, Q, F, H): 

    """
    Kalman Filter implementation

    Parameters:
    - x: Initial state estimate
    - P: Initial state covariance matrix
    - measurements: List of measurements over time !!!! this is a torch tensor not a list !!!!!!!!!!!!!!!
    - R: Measurement noise covariance matrix
    - Q: Process noise covariance matrix
    - F: State transition matrix
    - H: Measurement matrix

    Returns:
    - filtered_states: tensor of filtered state estimates of dimension (seq_len, num_peds, 4)
    - filtered_covariances: tensor of filtered state covariances of dieemnsion (seq_len, num_peds, 4, 4)
    """
    x = x.unsqueeze(2) 
    filtered_states = [x]
    filtered_covariances = [P]
 

    for z in measurements: # this is going step by step over the sequence length

        z = z.unsqueeze(2)

        # Prediction Step
        x = F @ x
        P = F @ P @ F.transpose(-1, -2) + Q

        # Update Step
        innovation = z - H @ x
        S = H @ P @ H.transpose(-1, -2) + R
        K = P @ H.transpose(-1, -2) @ torch.inverse(S)


        x = x + K @ innovation
        P = (torch.eye(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1,1)- K @ H) @ P 

        filtered_states.append(x)
        filtered_covariances.append(P)


    # covnert the list of tensors to a single tensor of size (seq_len, num_peds, 2,2) for the convariance
    filtered_covariances = torch.stack(filtered_covariances, dim=0)
    # also convert the list of tesors for the filtered state to a single tensor of size (seq_len, num_peds, 4)
    filtered_states = torch.stack(filtered_states, dim=0)

    return filtered_states.squeeze(-1), filtered_covariances


def cov_mat_generation(dist_param): # the input should be dist_param after the scaling has been done in the gen_Coef
        
    '''
    Generating the covanriance matrix from the distribution parameters
    dist_param: numpy array of shape: (pred_seq_len, num_peds, 5)
    bi-varainat gaussian distribution parameters in the third dimesnion are ordered as follows:
    [mu_x, mu_y, sigma_x, sigma_y, rho]
    Output: numpy array of shape: (pred_seq_len, num_peds, 2, 2)
    '''
    
    mu_x = dist_param[:, :, 0]
    mu_y = dist_param[:, :, 1]
    sigma_x = dist_param[:, :, 2]
    sigma_y = dist_param[:, :, 3]
    rho = dist_param[:, :, 4]

    # compute the element of the covariance matrix
    sigma_x2 = sigma_x ** 2
    sigma_y2 = sigma_y ** 2
    sigma_xy = sigma_x * sigma_y
    rho_sigma_xy = rho * sigma_xy

    # create the convanriance matrix tensor
    cov_mat = torch.stack([
        torch.stack([sigma_x2, rho_sigma_xy], dim=-1),
        torch.stack([rho_sigma_xy, sigma_y2], dim=-1)
        ], dim=-2)

    # I guess there is no need for worrying about the non-existing peds of the corrent steps in the output
    # as the distribution parameters for those peds are all zeros and the cov_mat will be all zeros as well
    # each row is the cov_mat of the prediction we have for that specifc time step
    
    return cov_mat


