#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import cv2
import torch
import os
import agent
import numpy as np
from data_loader import Generator
from parameters import Parameters
import test
import evaluation
import util
import copy

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training():
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')

    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
        lane_agent.load_weights(50, "tensor(0.2378)", False) # quan model
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(50, "tensor(0.2378)") # quan model

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    sampling_list = None
    for epoch in range(p.n_epoch):
        lane_agent.training_mode()

        for inputs, target_lanes, target_h, test_image, data_list in loader.Generate(sampling_list):
            #training
            print("epoch : " + str(epoch))
            print("step : " + str(step))
            loss_p = lane_agent.train(inputs, target_lanes, target_h, epoch, lane_agent, data_list)
            torch.cuda.synchronize()
            loss_p = loss_p.cpu().data
            step += 1

        sampling_list = copy.deepcopy(lane_agent.get_data_list())
        lane_agent.sample_reset()
        
        #evaluation
        if epoch >= 0 and epoch%1 == 0:
            print("evaluation")
            lane_agent.evaluate_mode()
            th_list = [0.8]
            index = [3]
            lane_agent.save_model(int(step/100), loss_p)

            for idx in index:
                print("generate result")
                test.evaluation(loader, lane_agent, index = idx, name="./eval_res/test_result_"+str(epoch)+"_"+str(idx)+".json")

            for idx in index:
                print("compute score")
                with open("./eval_res/eval_acc.txt", 'a') as make_file:
                    make_file.write( "epoch : " + str(epoch) + " loss : " + str(loss_p.cpu().data) )
                    make_file.write(evaluation.LaneEval.bench_one_submit("./eval_res/test_result_"+str(epoch)+"_"+str(idx)+".json", "test_label.json"))
                    make_file.write("\n")

        if int(step)>50000:
            break
        
        if epoch > 20:
            # Freeze quantizer parameters
            lane_agent.d_observer()
        if epoch > 20:
            # Freeze batch norm mean and variance estimates
            lane_agent.freeze_bn()

    
if __name__ == '__main__':
    Training()

