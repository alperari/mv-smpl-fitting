'''
 @FileName    : Demo_AlphaPose.py
 @EditTime    : 2024-04-05 16:04:00
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description :
'''
# Ensure imports from repository root work when running: python code/keypoint_predict.py
import cv2
import argparse
import os
import sys
import torch
from utils.FileLoaders import save_keypoints
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# fmt: off
from alphapose_core.alphapose_core import AlphaPose_Predictor
from yolox.yolox import Predictor
# fmt: on

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def draw_bbox_confidences(img, bboxes, confs):
    if img is None or bboxes is None or confs is None:
        return img

    n = min(len(bboxes), len(confs))
    for i in range(n):
        x1, y1, x2, y2 = bboxes[i]
        score = float(confs[i])
        label = '#{} {:.3f}'.format(i, score)

        x = int(round(x1))
        y = int(round(y1))

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text_top = max(0, y - th - baseline - 4)
        y_text_bottom = y_text_top + th + baseline + 4
        x_text_right = min(img.shape[1] - 1, x + tw + 8)

        cv2.rectangle(
            img, (max(0, x), y_text_top), (x_text_right, y_text_bottom),
            (0, 0, 0), -1)
        cv2.putText(
            img, label, (max(0, x + 4), y_text_bottom - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run YOLOX + AlphaPose and save keypoints and overlays.')
    parser.add_argument('--input_folder', default='data/images',
                        help='Input image root folder: <seq>/<camera>/<image>.jpg')
    parser.add_argument('--keypoint_output_folder', default='data/keypoints',
                        help='Output root folder for keypoint json files')
    parser.add_argument('--overlay_output_folder', default='output/alphapose',
                        help='Output root folder for keypoint overlay images')
    parser.add_argument('--bbox_output_folder', default='output/yolox_bbox',
                        help='Output root folder for YOLOX bbox overlay images')
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
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device, e.g. cuda:0, cuda:1, or cpu')
    parser.add_argument('--pose_batch_size', type=int, default=20,
                        help='AlphaPose pose batch size; lower this to reduce GPU memory')
    parser.add_argument('--yolox_input_h', type=int, default=800,
                        help='YOLOX inference input height')
    parser.add_argument('--yolox_input_w', type=int, default=1440,
                        help='YOLOX inference input width')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_folder):
        raise FileNotFoundError(
            'Input folder does not exist: {}'.format(args.input_folder))

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA device requested but torch.cuda.is_available() is False')

    device = torch.device(args.device)
    print('Using device: {}'.format(device))

    yolox_predictor = Predictor(
        args.yolox_model, args.yolox_thres, device=device)
    yolox_predictor.test_size = (args.yolox_input_h, args.yolox_input_w)
    alpha_predictor = AlphaPose_Predictor(
        args.alpha_config, args.alpha_checkpoint, args.alpha_thres, device=device)
    alpha_predictor.posebatch = max(1, args.pose_batch_size)
    print('AlphaPose posebatch: {}'.format(alpha_predictor.posebatch))
    print('YOLOX input size: {}x{}'.format(
        args.yolox_input_h, args.yolox_input_w))

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

                results, bbox_img = yolox_predictor.predict(img, viz=args.viz)
                print('YOLOX results: {}'.format(results))
                bboxes = results.get('bbox', []) if isinstance(
                    results, dict) else []
                bboxes_conf = results.get('bboxes_conf', []) if isinstance(
                    results, dict) else []
                if len(bboxes) == 0:
                    print('No person detected, skip: {}'.format(img_path))
                    continue

                stem, _ = os.path.splitext(name)
                bbox_path = os.path.join(
                    args.bbox_output_folder, seq, camera,
                    stem + '_bbox.jpg')
                os.makedirs(os.path.dirname(bbox_path), exist_ok=True)
                bbox_img = draw_bbox_confidences(bbox_img, bboxes, bboxes_conf)
                cv2.imwrite(bbox_path, bbox_img)
                print('Save bbox overlay: {}'.format(bbox_path))

                pose = alpha_predictor.predict(img, bboxes)[:1]
                if len(pose) == 0:
                    print('No keypoints predicted, skip: {}'.format(img_path))
                    continue

                # format: coco17, halpe
                result_img = alpha_predictor.visualize(
                    img, pose, format='coco17', viz=args.viz)

                keypoint_path = os.path.join(
                    args.keypoint_output_folder, seq, camera,
                    stem + '_keypoints.json')
                os.makedirs(os.path.dirname(keypoint_path), exist_ok=True)
                save_keypoints(pose, keypoint_path)
                print('Save keypoints: {}'.format(keypoint_path))

                overlay_path = os.path.join(
                    args.overlay_output_folder, seq, camera,
                    stem + '_keypoints_overlay.jpg')
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                cv2.imwrite(overlay_path, result_img)
                print('Save overlay: {}'.format(overlay_path))


if __name__ == '__main__':
    main()
