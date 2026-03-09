'''
 @FileName    : Demo_AlphaPose.py
 @EditTime    : 2024-04-05 16:04:00
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description :
'''
from utils.FileLoaders import save_keypoints
import os
import cv2
from alphapose_core.alphapose_core import AlphaPose_Predictor
from yolox.yolox import Predictor
import argparse
import sys
sys.path.append('./')

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run YOLOX + AlphaPose and save keypoints and overlays.')
    parser.add_argument('--input_folder', default='data/images',
                        help='Input image root folder: <seq>/<camera>/<image>.jpg')
    parser.add_argument('--keypoint_output_folder', default='data/keypoints',
                        help='Output root folder for keypoint json files')
    parser.add_argument('--overlay_output_folder', default='output/alphapose',
                        help='Output root folder for keypoint overlay images')
    parser.add_argument('--viz', action='store_true',
                        help='Show AlphaPose visualization windows while processing')
    parser.add_argument('--yolox_model',
                        default='pretrained/yolox_data/bytetrack_x_mot17.pth.tar',
                        help='Path to YOLOX pretrained model')
    parser.add_argument('--alpha_config',
                        default='alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
                        help='Path to AlphaPose config yaml')
    parser.add_argument('--alpha_checkpoint',
                        default='pretrained/alphapose_data/halpe26_fast_res50_256x192.pth',
                        help='Path to AlphaPose checkpoint')
    parser.add_argument('--yolox_thres', type=float, default=0.23,
                        help='YOLOX detection threshold')
    parser.add_argument('--alpha_thres', type=float, default=0.1,
                        help='AlphaPose keypoint threshold')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_folder):
        raise FileNotFoundError(
            'Input folder does not exist: {}'.format(args.input_folder))

    yolox_predictor = Predictor(args.yolox_model, args.yolox_thres)
    alpha_predictor = AlphaPose_Predictor(
        args.alpha_config, args.alpha_checkpoint, args.alpha_thres)

    seqs = sorted(os.listdir(args.input_folder))
    for seq in seqs:
        seq_dir = os.path.join(args.input_folder, seq)
        if not os.path.isdir(seq_dir):
            continue

        cameras = sorted(os.listdir(seq_dir))
        for camera in cameras:
            cam_dir = os.path.join(seq_dir, camera)
            if not os.path.isdir(cam_dir):
                continue

            imgs = sorted(
                name for name in os.listdir(cam_dir)
                if name.lower().endswith(IMG_EXTS)
            )

            for name in imgs:
                img_path = os.path.join(cam_dir, name)
                img = cv2.imread(img_path)
                if img is None:
                    print('Skip unreadable image: {}'.format(img_path))
                    continue

                results, _ = yolox_predictor.predict(img, viz=False)
                bboxes = results.get('bbox', []) if isinstance(
                    results, dict) else []
                if len(bboxes) == 0:
                    print('No person detected, skip: {}'.format(img_path))
                    continue

                pose = alpha_predictor.predict(img, bboxes)[:1]
                if len(pose) == 0:
                    print('No keypoints predicted, skip: {}'.format(img_path))
                    continue

                # format: coco17, halpe
                result_img = alpha_predictor.visualize(
                    img, pose, format='coco17', viz=args.viz)

                stem, _ = os.path.splitext(name)
                keypoint_path = os.path.join(
                    args.keypoint_output_folder, seq, camera,
                    stem + '_keypoints.json')
                os.makedirs(os.path.dirname(keypoint_path), exist_ok=True)
                save_keypoints(pose, keypoint_path)
                print('Save keypoints: {}'.format(keypoint_path))

                overlay_path = os.path.join(
                    args.overlay_output_folder, seq, camera,
                    stem + '_overlay.jpg')
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                cv2.imwrite(overlay_path, result_img)
                print('Save overlay: {}'.format(overlay_path))


if __name__ == '__main__':
    main()
