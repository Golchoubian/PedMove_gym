import gym
import os
from matplotlib import pyplot as plt
from PedMove_gym.arguments import get_args
from PedMove_gym.config import Config
from PedMove_gym.ped_pred.plot_scenario import Create_gif, Create_video


def main():
    """
    main function for training a robot policy network
    """
    # read arguments
    args = get_args()
    config = Config()

    # for visualization
    if config.sim.render:
        fig, ax = plt.subplots()
        plt.ion()
        plt.show()
    else:
        ax = None
        fig = None

    env_name = args.env_name
    env = gym.make(env_name)
   
    env.nenv = 1
    env.phase = args.phase

    env.configure(config)

    if ax:
        env.render_axis = ax
        env.render_figure = fig

    test_size = env.dataloader.num_scenarios
    if args.test_case >= 0:
        env.test_case = args.test_case
        test_size = 1


    for k in range(test_size):
        done = False
        obs = env.reset()
        while not done:

            # call your action calculation function here
            # Here to only show a demontration of the simulation environment we are 
            # creating an action to closly follow the humans driven trajecoty in the dataset
            action = env.robot.human_driver_policy(env.frame)

            if config.sim.render:
                env.render()

            obs, reward, done, infos = env.step(action)


            # # =======================================================================
            # #                      printing some of the observations
            # # =======================================================================

            # print(' <><><><><><><><><><> frame:', env.frame, ' <><><><><><><><><><>')
            
            # robot_pos = obs['robot_node'][:2]
            # robot_vel = obs['temporal_edges']

            # if config.sim.predict_method != 'no_pred':
            #     third_dim = int(env.spatial_edge_dim / (env.pred_length+1))
            #     peds_state = obs['spatial_edges'].reshape(config.sim.max_human_num,
            #                                                 config.sim.pred_len+1, third_dim) 
            # else:
            #     peds_state = obs['spatial_edges'].clone()

            # peds_visibility = obs['visible_masks']
            # peds_state_visible = peds_state[peds_visibility]
            # peds_pos = peds_state_visible[..., :2] + robot_pos # back to absolute pos

            # if config.sim.predict_method == 'inferred':
            #     peds_pred_cov = peds_state_visible[:, 1:, 2:]
            #     print(peds_pred_cov.shape)
            #     peds_pred_cov = peds_pred_cov.reshape(peds_pred_cov.shape[0], 
            #                                           config.sim.pred_len, 2, 2)
                
            # print('robot pos:', robot_pos)
            # print('robot vel:', robot_vel)
            # # x,y position postion of current and predicted future paths 
            # # of visible pedestrians to the AV/robot
            # print('peds pos:', peds_pos) # [num_visible_peds, pred_length+1, 2] 
            # if config.sim.predict_method == 'inferred':
            #     print('peds cov:', peds_pred_cov)

            # # =======================================================================
            # # =======================================================================

        
        # render the last frame when doen
        if config.sim.render:
                env.render()


        # a scenario ends!
        print('')
        print('scenario #', env.scenario_num, 'ends at frame:', env.frame)

        if config.sim.render:
            # save the sceario as a gif
            early_stop_frame = env.frame
            gif_folder = os.path.join(config.data.visual_save_path, 'gifs')
            plot_folder = os.path.join(config.data.visual_save_path, 'plots')
            if not os.path.exists(gif_folder):
                os.makedirs(gif_folder, exist_ok=True)
            Create_gif(plot_folder, gif_folder, env.simulation_lenght, 
                        env.obs_length, env.scenario_num,
                        early_stop_frame, fps=3)
            # Create_video(plot_folder, gif_folder, env.simulation_lenght,
            #             env.obs_length-1, env.scenario_num,
            #             early_stop_frame, fps=3)



if __name__ == '__main__':
    main()
