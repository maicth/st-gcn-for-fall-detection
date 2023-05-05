
import argparse

import torch
from processor.io import IO
from demo.process_video import *
import cv2
import time

class AIPdemo(IO):

    def start(self):
        # gen data
        skeleton_path = gen_skeleton_path(self.arg.video)
        skeleton_list = create_skeleton_list(skeleton_path, self.arg.len_sub_video)
        gendata(self.arg.out_path, skeleton_list)
        # load data
        dataset = torch.from_numpy(np.load("./demo/demo_data.npy"))
        # print("len dataset:", len(dataset))
        play_text_count = 0
        play_text_flag = 0
        # start capture video
        video_capture = cv2.VideoCapture(self.arg.video)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        print("num frames:", num_frames)
        frame_index = 0
        num_check = 0
        # text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 5
        org = (300,300)
        color = (0,0,255)   # red
        thickness = 15

        while (video_capture.isOpened()):
            ret, cur_frame = video_capture.read()
            frame_index += 1

            if frame_index == self.arg.len_sub_video*(num_check+1) or frame_index == num_frames:
                C, T, V, M = dataset[num_check].shape
                print('M=',M)
                data = dataset[num_check].resize_(1,C,T,V,M)    # N,C,T,V,M
                data = data.float().to(self.dev)

                # predict
                with torch.no_grad():
                    output = self.model(data)
                if output[0][0]<0:
                    if output[0][0] > -2:
                        print("----------FALL WARNING----------")
                        play_text_flag = 2
                        play_text_count = 0
                    else:
                        print("----------FALL----------")
                        play_text_flag = 1
                        play_text_count = 0
                else:
                    print("Non Fall")
                num_check += 1

            # display text in 2 secs
            if play_text_flag == 1:
                cv2.putText(cur_frame, 'FALL', org, font, font_size, color, thickness)
                play_text_count += 1
            # elif play_text_flag == 2:
            #     cv2.putText(cur_frame, 'FALL WARNING', org, font, font_size, (0,204,255), thickness)
            #     play_text_count += 1
            if play_text_count >= fps*2:
                play_text_flag = 0
                play_text_count = 0

            if cv2.waitKey(int(1000/50)) & 0xFF == ord('q'):
                break
            cv2.imshow('video', cur_frame)
            cv2.waitKey(int(100/self.arg.model_fps))
            if num_check == len(dataset):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for AIP490')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/skateboarding.mp4',
                            help='Path to video')
        parser.add_argument('--len_sub_video',
                            default=120,
                            type=int,
                            help='Length of sub-video to gen data')
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--batch_size',
                            type=int,
                            default=256,
                            help='training batch size')
        parser.add_argument('--out_path',
                            default='./demo',
                            help='Path to .npy file')
        parser.set_defaults(
            config='./config/st_gcn/aip_demo/aip_demo.yaml')
        parser.set_defaults(print_log=True)
        # endregion yapf: enable

        return parser
