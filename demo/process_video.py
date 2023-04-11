import os
import sys
import csv
import numpy as np
from numpy.lib.format import open_memmap

def create_skeleton_list(skeleton_path, len_sub_video):
    skeleton_list = []
    with open(skeleton_path, "r") as f:
        data = list(csv.reader(f))
    for i in range(0,len(data),len_sub_video*25):  # 1 sec = 30 frames = 30*25 joints
        sub_skeleton = data[i:i+len_sub_video*25]    # each sub skeleton = 6 secs video
        skeleton_list.append(sub_skeleton)

        if len(sub_skeleton) < len_sub_video*25:
            break
    return skeleton_list

def read_skeleton(sub_skeleton):
    skeleton_sequence = {}
    skeleton_sequence['numFrame'] = len(sub_skeleton) // 25
    skeleton_sequence['frameInfo'] = []
    joint_info_key = ['jointName', 'x', 'y', 'z', 'state']

    for f in range(skeleton_sequence['numFrame']):
        frame_info = {}
        frame_info['numBody'] = 1
        frame_info['bodyInfo'] = []

        for b in range(frame_info['numBody']):
            body_info = {}
            body_info['numJoint'] = 25
            body_info['jointInfo'] = []

            for j in range(body_info['numJoint']):
                joint_info_value = sub_skeleton[25 * f + j]
                joint_info = {k: v for k, v in zip(joint_info_key, joint_info_value)}

                body_info['jointInfo'].append(joint_info)
            frame_info['bodyInfo'].append(body_info)
        skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence

def read_xyz(sub_skeleton, max_body=2, num_joint=25):
    seq_info = read_skeleton(sub_skeleton)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for nf, f in enumerate(seq_info['frameInfo']):
        for nb, b in enumerate(f['bodyInfo']):
            for nj, j in enumerate(b['jointInfo']):
                if nb < max_body and nj < num_joint:
                    data[:, nf, nj, nb] = [j['x'], j['y'], j['z']]
                else:
                    pass
    return data

def gendata(out_path, skeleton_list):
    max_body = 2
    num_joint = 25
    max_frame = 300
    fp = open_memmap(
        '{}\demo_data.npy'.format(out_path),
        dtype='float32',
        mode='w+',
        shape=(len(skeleton_list), 3, max_frame, num_joint, max_body))

    for i, sub_skeleton in enumerate(skeleton_list):
        print('({:>5}/{:<5}) Processing aip_demo data: '.format(
                          i + 1, len(skeleton_list)))
        data = read_xyz(sub_skeleton, max_body=max_body, num_joint=num_joint)
        print(data.shape)
        fp[i, :, 0:data.shape[1], :, :] = data

def gen_skeleton_path(video_path):
    element_path = video_path.split('_')
    skeleton_path = element_path[0]+"_Skeleton.csv"
    return skeleton_path
