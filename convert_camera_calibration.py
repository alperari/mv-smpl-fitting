import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import glob


def convert_xcp_to_camtxt(xcp_filepath, output_filepath):
    # Parse the XML file
    try:
        tree = ET.parse(xcp_filepath)
        root = tree.getroot()
    except Exception as e:
        print(f"Error reading {xcp_filepath}: {e}")
        return

    output_lines = []
    camera_index = 0

    # Iterate through all cameras in the file
    for camera in root.findall('Camera'):
        # Only process the specific Blackfly S RGB cameras
        display_type = camera.get('DISPLAY_TYPE', '')
        if display_type == "VideoInputDevice:Blackfly S BFS-U3-23S3C":

            # Find the KeyFrame tag which contains the actual calibration math
            keyframes = camera.find('KeyFrames')
            if keyframes is not None:
                keyframe = keyframes.find('KeyFrame')
                if keyframe is not None:

                    # --- 1. Extract Intrinsics ---
                    focal_length = float(keyframe.get('FOCAL_LENGTH'))
                    cx, cy = map(float, keyframe.get(
                        'PRINCIPAL_POINT').split())

                    # Construct K matrix
                    K = np.array([
                        [focal_length, 0, cx],
                        [0, focal_length, cy],
                        [0, 0, 1]
                    ])

                    # --- 2. Extract Extrinsics ---
                    # Position (C) -> Convert from mm to meters
                    tx, ty, tz = map(float, keyframe.get('POSITION').split())
                    C = np.array([tx, ty, tz]) / 1000.0

                    # Orientation (Quaternion)
                    qx, qy, qz, qw = map(
                        float, keyframe.get('ORIENTATION').split())

                    # Convert Quaternion to 3x3 Rotation Matrix (R)
                    rot = R.from_quat([qx, qy, qz, qw])
                    R_mat = rot.as_matrix()

                    # Calculate Translation vector (t) for OpenCV: t = -R * C
                    t = -R_mat @ C

                    # --- 3. Format the Output ---
                    output_lines.append(f"{camera_index}")

                    # K Matrix formatting
                    output_lines.append(f"{K[0, 0]} {K[0, 1]} {K[0, 2]}")
                    output_lines.append(f"{K[1, 0]} {K[1, 1]} {K[1, 2]}")
                    output_lines.append(f"{K[2, 0]} {K[2, 1]} {K[2, 2]}")

                    # Distortion formatting (hardcoded to 0 0 based on your example)
                    output_lines.append("0 0")

                    # R Matrix and t Vector formatting
                    output_lines.append(
                        f"{R_mat[0, 0]} {R_mat[0, 1]} {R_mat[0, 2]} {t[0]}")
                    output_lines.append(
                        f"{R_mat[1, 0]} {R_mat[1, 1]} {R_mat[1, 2]} {t[1]}")
                    output_lines.append(
                        f"{R_mat[2, 0]} {R_mat[2, 1]} {R_mat[2, 2]} {t[2]}")
                    output_lines.append("")  # Empty line between cameras

                    camera_index += 1

    # Write the formatted data to the text file
    if len(output_lines) > 0:
        with open(output_filepath, 'w') as f:
            f.write("\n".join(output_lines))
        print(f"Successfully converted {camera_index} Blackfly S cameras.")
        print(f"Saved to: {output_filepath}")
    else:
        print("No cameras matching 'VideoInputDevice:Blackfly S BFS-U3-23S3C' were found.")


def find_first_xcp_in_cwd():
    files = glob.glob("*.xcp")
    if not files:
        print("No .xcp files found in current directory.")
        return None
    return files[0]


if __name__ == "__main__":
    xcp_file = find_first_xcp_in_cwd()
    if xcp_file:
        output_file = "cam_converted.txt"
        convert_xcp_to_camtxt(xcp_file, output_file)
