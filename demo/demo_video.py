import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from common.base import Demoer
from common.utils.human_models import smpl_x
from common.utils.preprocessing import process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj
from config import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--video_file', type=str, default='./input.mp4')  # change to video_file
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal',
                        choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='pretrained_models/osx_l.pth.tar')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def process_image(frame, frame_number, detector, demoer, args):
    # prepare input image
    transform = transforms.ToTensor()

    original_img = frame
    original_img_height, original_img_width = original_img.shape[:2]
    os.makedirs(args.output_folder, exist_ok=True)

    with torch.no_grad():
        results = detector(original_img)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    class_ids, confidences, boxes = [], [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.02, 0.2)
    vis_img = original_img.copy()
    for num, indice in enumerate(indices):
        bbox = boxes[indice]  # x,y,h,w
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
        
        # CVCHI Practical Course: Extract SMPL-H Pose
        root_joint = out['smplx_root_pose']
        body_joints = out['smplx_body_pose']
        l_hand_joints = out['smplx_lhand_pose']
        r_hand_joints = out['smplx_rhand_pose']
        pose = torch.cat((out['smplx_root_pose'], out['smplx_body_pose'], out['smplx_lhand_pose'], out['smplx_rhand_pose']), 1).cpu()
        # TODO: Get translation!
        trans = [[0,0,0]]

        mesh = mesh[0]

        # save mesh 
        save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'{frame_number}_person_{num}.obj'))

        # render mesh
        focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0],
                   cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})

    # save rendered image
    cv2.imwrite(os.path.join(args.output_folder, f'{frame_number}.jpg'), vis_img[:, :, ::-1])
    if len(indices) == 0:
        pose = None
        trans = None
    return pose, trans

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    # load model
    cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting,
                            pretrained_model_path=args.pretrained_model_path)

    demoer = Demoer()
    demoer._make_model()

    model_path = args.pretrained_model_path
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))

    demoer.model.eval()

    # detect human bbox with yolov5s
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open video file
    cap = cv2.VideoCapture(args.video_file)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    poses = []
    trans = []

    frame_number = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pose, frame_translation = process_image(rgb_frame, frame_number, detector, demoer, args)
            if frame_pose == None:
                frame_pose = poses[-1:]
            if frame_translation == None:
                frame_translation = trans[-1:]
            if len(poses) > 0:
                poses = np.concatenate((poses, frame_pose))
            else:
                poses = frame_pose
            if len(trans) > 0:
                trans = np.concatenate((trans, frame_translation))
            else:
                trans = frame_translation                
            frame_number += 1
            print(f"Frame {frame_number}")
        else:
            break
    
    mocap_framerate = [cap.get(cv2.CAP_PROP_FPS)]
    gender = ["male"]
    np.savez("fbx_input.npz", poses=poses, mocap_framerate=mocap_framerate, gender=gender, trans=trans)

if __name__ == "__main__":
    main()
