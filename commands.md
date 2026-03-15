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

### Run inference with that camera

> Add CUDA_VISIBLE_DEVICES=2 to the command if you want to specify a GPU.

xvfb-run -s "-screen 0 1280x1024x24" \
 python code/main.py \
 --config cfg_files/fit_smpl.yaml \
 --cam_param data_alp_4_view_1_cam/cam.txt \
 --data_folder data_alp_4_view_1_cam

### With sequential mode inference

xvfb-run -s "-screen 0 1280x1024x24" \
python code/main.py \
 --config cfg_files/fit_smpl.yaml \
 --data_folder data_steve_4_view_seq \
 --cam_param data_steve_4_view_seq/cam.txt \
 --is_seq true \

# Keypoint detection with AlphaPose

python code/keypoint_predict.py \
 --input_folder data_alp_1_view_frontal/images \
 --keypoint_output_folder data_alp_1_view_frontal/keypoints \
 --overlay_output_folder data_alp_1_view_frontal/keypoint_overlays \
 --bbox_output_folder data_alp_1_view_frontal/bbox_overlays \
 --yolox_model ./pretrained/yolox_data/bytetrack_x_mot17.pth.tar \
 --alpha_checkpoint ./pretrained/alphapose_data/halpe26_fast_res50_256x192.pth

# Convert MPI camera parameters to OpenCV format for AlphaPose

python convert_camera_calibration.py \
 --input data_mpi_4_view_s1/camera.calibration \
 --output data_mpi_4_view_s1/cam.txt \
 --scene_dir data_mpi_4_view_s1/images/scene_1

# Generate single camera parameters for a single view image

python generate_single_cam_params.py \
 --image data_steve_4_view/images/scene1/iphone1/P1_back.jpg \
 --output data_steve_4_view/cam.txt


### AlphaPose HALPE26 keypoints
 //26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    //face
    {26-93, 68 Face Keypoints}
    //left hand
    {94-114, 21 Left Hand Keypoints}
    //right hand
    {115-135, 21 Right Hand Keypoints}