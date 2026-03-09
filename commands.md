### VISUALIZE KEYPOINTS

python visualize_keypoints.py \
 --keypoint_folder data/keypoints_test_data/keypoints \
 --image_folder data/keypoints_test_data/raw_images \
 --output_folder data/keypoints_test_data/keypoint_overlays

### Run demo headless,

xvfb-run -s "-screen 0 1280x1024x24" python code/main.py --config cfg_files/fit_smpl.yaml

### COCO-17 keypoint order

Nose, Leye, Reye, Lear, Rear, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA

# SINGLE VIEW CAM EST & INFERENCE

### Run a small standalone script in the repo root that takes an image path,

### infers width/height, and writes a valid single-camera parameter file in this project’s expected format.

python generate_single_cam_params.py \
 --image data_single_view_mpi/images/0000/Camera00/cam_8.jpg \
 --output data_single_view_mpi/cam.txt

### And run inference with that camera

> Add CUDA_VISIBLE_DEVICES=2 to the command if you want to specify a GPU.

xvfb-run -s "-screen 0 1280x1024x24" \
 python code/main.py \
 --config cfg_files/fit_smpl.yaml \
 --cam_param data_mpi_4_view_s1/cam.txt \
 --data_folder data_mpi_4_view_s1

# Keypoint detection with AlphaPose

python code/keypoint_predict.py \
 --input_folder data_mpi_4_view_s1/images \
 --keypoint_output_folder data_mpi_4_view_s1/keypoints \
 --overlay_output_folder data_mpi_4_view_s1/keypoint_overlays \
 --bbox_output_folder data_mpi_4_view_s1/bbox_overlays \
 --yolox_model ./pretrained/yolox_data/bytetrack_x_mot17.pth.tar \
 --alpha_checkpoint ./pretrained/alphapose_data/halpe26_fast_res50_256x192.pth
