import xml.etree.ElementTree as ET
import numpy as np

tree = ET.parse('Vicon_lab_calibration.xcp')
root = tree.getroot()
cams = []
for cam in root.findall('Camera'):
    if cam.attrib.get('DISPLAY_TYPE') == 'VideoInputDevice:Blackfly S BFS-U3-23S3C':
        kf = cam.find('./KeyFrames/KeyFrame')
        if kf is None:
            continue
        fx = float(kf.attrib['FOCAL_LENGTH'])
        cx, cy = map(float, kf.attrib['PRINCIPAL_POINT'].split())
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
        D = [0, 0, 0, 0, 0]
        pos = np.array(list(map(float, kf.attrib['POSITION'].split())))
        q = np.array(list(map(float, kf.attrib['ORIENTATION'].split())))
        # Convert quaternion to rotation matrix
        w, x, y, z = q
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])
        Rt = np.hstack([R, pos.reshape(3, 1)])
        cams.append((K, D, Rt))
        if len(cams) == 2:
            break

with open('Vicon_lab_calibration.txt', 'w') as f:
    for idx, (K, D, Rt) in enumerate(cams):
        f.write(f"{idx}\n")
        for row in K:
            f.write(' '.join(map(str, row)) + '\n')
        # Write only 2 zeros for distortion
        f.write('0 0\n')
        for row in Rt:
            f.write(' '.join(map(str, row)) + '\n')
        f.write('\n')
