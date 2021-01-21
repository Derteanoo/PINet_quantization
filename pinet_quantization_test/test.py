#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
import agent
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import os
import util
from tqdm import tqdm
import csaps

p = Parameters()

###############################################################
##
## Testing
## 
###############################################################
def Testing():
    print('Testing')
    
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
    ## testing
    ##############################
    print('Testing loop')
    lane_agent.evaluate_mode()

    if p.mode == 0 : # check model with test data
        for test_image, _, _, _, i in loader.Generate_Test():#loader.Generate()
            _, _, ti = test(lane_agent, np.array([test_image]))
            cv2.imwrite("./eval/{}_test.png".format(i), ti[0])

    elif p.mode == 1: # check model with video
        test_videos = 'test_videos.txt'
        save_dir = 'test_video_result'
        video_list = open(test_videos).readlines()
        use_ori = True
        for video in video_list:
            video = video.strip()
            
            save_video_dir = save_dir + "/".join(video.split("/")[-3:-1])
            if not os.path.exists(save_video_dir):
                os.makedirs(save_video_dir)
            
            cap = cv2.VideoCapture(video)
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),

                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(size) # wh
            h,w = size[1],size[0]
            hcrop = h//5
            tmp = hcrop*2
            videoWriter = cv2.VideoWriter(save_video_dir + '/{}.avi'.format(video.split('/')[-1][:-4]), cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
            count = 0
            while(cap.isOpened()):
                ret, ori_frame = cap.read()
                count += 1
                if ret==False:
                    break
                
                crop_frame = ori_frame[tmp:,:]
                prevTime = time.time()
                frame = cv2.resize(crop_frame, (256,256))/255.0
                frame = np.rollaxis(frame, axis=2, start=0)
                if not use_ori:
                    _, _, ti = test(lane_agent, np.array([frame]))
                if use_ori:
                    ratio_w = p.x_size * 1.0 / (size[0])
                    ratio_h = p.y_size * 1.0 / (size[1]-tmp)
                    _, _, ti = test_ori(tmp,lane_agent,ori_frame, np.array([frame]), ratio_w, ratio_h,draw_type = 'point', thresh=p.threshold_point)
                curTime = time.time()
                sec = curTime - prevTime
                fps = 1/(sec)
                s = "FPS : "+ str(round(fps, 3))
                cv2.putText(ti[0], s, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                videoWriter.write(ti[0])
            cap.release()
            print("video writer finished")

    elif p.mode == 2: # check model with a picture
        test_images = "./test_pic"
        save = "./test_pic_res/"
        img_list = os.listdir(test_images)
        img_list = [img for img in img_list if '.jpg' in img]
        use_ori = True
        print("image test")
        for img in img_list:
            ori_image = cv2.imread(test_images + '/' + img) #hw
            h,w = ori_image.shape[0],ori_image.shape[1]
            hcrop = h//5
            tmp = hcrop*2
            crop_image = ori_image[tmp:,:]
            
            test_image = cv2.resize(crop_image, (256, 256)) / 255.0
            test_image = np.rollaxis(test_image, axis=2, start=0)

            if not use_ori:
                _, _, ti = test(lane_agent, np.array([test_image]))
            if use_ori:
                ratio_w = p.x_size * 1.0 / ori_image.shape[1]
                ratio_h = p.y_size* 1.0 / (ori_image.shape[0]-tmp)
                _, _, ti = test_ori(tmp,lane_agent, ori_image, np.array([test_image]), ratio_w, ratio_h,draw_type = 'point',thresh=p.threshold_point)
            cv2.imwrite(save+img, ti[0])

    elif p.mode == 3: #evaluation
        print("evaluate")
        evaluation(loader, lane_agent)

def fitting(x, y, target_h, ratio_w, ratio_h):
    out_x = []
    out_y = []
    count = 0
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h

    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []

            jj = []
            pre = -100
            for temp in j[::-1]:
                if temp > pre:
                    jj.append(temp)
                    pre = temp
                else:
                    jj.append(pre+0.00001)
                    pre = pre+0.00001
            sp = csaps.CubicSmoothingSpline(jj, i[::-1], smooth=0.0001)

            last = 0
            last_second = 0
            last_y = 0
            last_second_y = 0
            for h in target_h[count]:
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    temp_x.append( sp([h])[0] )
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    if len(temp_x)<2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                else:
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch) 
        count += 1

    return out_x, out_y

############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(loader, lane_agent, index= -1, thresh = p.threshold_point, name = None):
    result_data = deepcopy(loader.test_data)
    progressbar = tqdm(range(loader.size_test))
    for test_image, target_h, ratio_w, ratio_h, testset_index, gt in loader.Generate_Test():
        x, y, _ = test(lane_agent, test_image, thresh, index)
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)

        x_, y_ = fitting(x_, y_, target_h, ratio_w, ratio_h)
        result_data = write_result_json(result_data, x_, y_, testset_index)
        progressbar.update(1)

    progressbar.close()
    if name == None:
        save_result(result_data, "test_result.json")
    else:
        save_result(result_data, name)

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh = p.threshold_point, index= -1):
    result = lane_agent.predict_lanes_test(test_images)
    #print(result[index][0].shape)
    torch.cuda.synchronize()
    confidences, offsets, instances = result[-1]
    
    num_batch = len(test_images)
    
    out_x = []
    out_y = []
    out_images = []
    
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)
        
    return out_x, out_y,  out_images

############################################################################
## write result
############################################################################
def write_result_json(result_data, x, y, testset_index):
    for index, batch_idx in enumerate(testset_index):
        result_data[batch_idx]['lanes'] = []
        for i in x[index]:
            result_data[batch_idx]['lanes'].append(i)
            result_data[batch_idx]['run_time'] = 1
    return result_data

############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image,and show result on original output.
############################################################################
def test_ori(h_crop,lane_agent,ori_image, test_images,ratio_w, ratio_h, draw_type, thresh=p.threshold_point):  # p.threshold_point:0.81
    result = lane_agent.predict_lanes_test(test_images)
    confidences, offsets, instances = result[-1]
    num_batch = len(test_images)
    out_x = []
    out_y = []
    out_images = []

    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0) * 255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)

        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)
        
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)  #
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
        in_x, in_y = util.sort_along_y(in_x, in_y)   

        if draw_type == 'line':
            result_image = util.draw_lines_ori(in_x, in_y, ori_image,ratio_w, ratio_h)  # 将最后且后处理后的坐标点绘制与原图上.
        elif draw_type == 'point':
            result_image = util.draw_point_ori(in_x, in_y, ori_image,ratio_w, ratio_h,h_crop)  # 将最后且后处理后的坐标点绘制与原图上.
        else:
            result_image = util.draw_points(in_x, in_y, ori_image,ratio_w, ratio_h)  # 将最后且后处理后的坐标点绘制与原图上.
        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)

    return out_x, out_y, out_images


############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>5:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh):
    
    mask = confidance > thresh
    
    grid = p.grid_location[mask] # 获取含有车道线的点的网格坐标
    offset = offsets[mask]
    feature = instance[mask]
    
    tmp = confidance[mask]
    xy_confidence = []
    x = []
    y = []
    lane_feature = []
    
    for i in range(len(grid)):                                        # 有多少个含有车道线的点
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)   # 还原点到原图上
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            
            con = tmp[i]
            
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
                
                xy_confidence.append([con])
            
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                
                if min_feature_dis <= p.threshold_instance:  
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                    
                    xy_confidence[min_feature_index].append(con)
                    
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])           
                    
                    xy_confidence.append([con])
    return x, y

def calc_k(i,j):
    '''
    Calculate the direction of lanes
    '''
    line_x = i
    line_y = j

    p = np.polyfit(line_x, line_y,deg = 1)
    rad = np.arctan(p[0])
    
    return rad
if __name__ == '__main__':
    Testing()
