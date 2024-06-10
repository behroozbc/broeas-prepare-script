from pyboreas import BoreasDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from pyboreas.utils.utils import get_inverse_tf
masirFolderDataset = '/mnt/h/test-datasets' # Path of root of datasets.
bd = BoreasDataset(masirFolderDataset, verbose=True)
paresh=5 #skip between frames.
masirFolderZakhire='./result3' # path of saving file
if not os.path.exists(masirFolderZakhire):
    os.mkdir(masirFolderZakhire)
seq = bd.sequences[0]
seq.synchronize_frames(ref='camera')  # simulates having synchronous measurements
idxs = [2000,14,500,30]  # try different frame indices!
for idx in range(len(seq.lidar_frames)):
    if(idx%paresh!=0):
        continue
    cam = seq.get_camera(idx)
    lid = seq.get_lidar(idx)
    lid.remove_motion(lid.body_rate)
    T_enu_camera = cam.pose
    T_enu_lidar = lid.pose
    T_camera_lidar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_lidar)
    lid.transform(T_camera_lidar)
    lid.passthrough([-75, 75, -20, 10, 0, 40])  # xmin, xmax, ymin, ymax, zmin, zmax
    # Project lidar points onto the camera image, using the projection matrix, P0.
    uv, colors, _ = lid.project_onto_image(seq.calib.P0)
    fig = plt.figure(figsize=(24.48, 20.48), dpi=100)
    ax = fig.add_subplot()
    ax.imshow(cam.img) # you can change it to this `np.zeros(cam.img.shape, dtype=np.uint8)` for black background
    ax.set_xlim(0, 2448)
    ax.set_ylim(2048, 0)
    ax.scatter(uv[:, 0], uv[:, 1], c=colors, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
    ax.set_axis_off()
    plt.savefig(os.path.join(masirFolderZakhire,f'{idx}.jpg'))
    plt.close()
    cam.unload_data()
    lid.unload_data()