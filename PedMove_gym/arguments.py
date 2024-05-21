import argparse
import torch


def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--phase',
        default='test', # train, val, test
        help='scenario sets within the specified phase will be run')
    
    # if -1, it will run all scenarios in the test set; 
    # if >=0, it will run the specified test case
    # valid scenario numbers: train: 49-247, val: 0-48, test: 248-310
    parser.add_argument('--test_case', type=int, default=-1)

    # "PedSimPred-v0": HBS dataset scenarios coupled with PCG pedestrian trajecotry prediction model
    parser.add_argument(
        '--env-name',
        default='PedSimPred-v0',
        help='name of the environment')
    
    parser.add_argument('--consider-veh',
                        default=False,
                        action='store_true',
                        help='whether to consider robot-vehicle spatial egdes in the model or not')
    
    # parser.add_argument(
    #     '--seed', type=int, default=425, help='random seed (default: 1)')
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    # ===========================================================
    #                   params of prediction model
    # ===========================================================
    parser.add_argument('--pred_length', type=int, default=6,
                        help='prediction length')
    parser.add_argument('--obs_length', type=int, default=6,
                        help='Observed length of the trajectory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size of the dataset. Its 1 for inference')
    # ===========================================================
    # ===========================================================


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
