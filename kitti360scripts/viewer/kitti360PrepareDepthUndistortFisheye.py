#!/usr/bin/python
# -*- coding: utf-8 -*-


#################
## Import modules
#################
import sys
# walk directories
import glob
# access to OS functionality
import os
# copy things
import copy
# numpy
import numpy as np
# matplotlib for colormaps
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
# struct for reading binary ply files
import struct
from kitti360scripts.helpers.csHelpers import Rodrigues
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid

CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        from kitti360scripts.helpers import curlVelodyneData
    except:
        CSUPPORT = False
        print('CSUPPORT is required for unwrapping the velodyne data!')
        print('Run ``CYTHONIZE_EVAL= python setup.py build_ext --inplace`` to build with cython')
        sys.exit(-1)


# the main class that loads raw 3D scans
class Kitti360Viewer3DRaw(object):

    # Constructor
    def __init__(self, seq=0, mode='velodyne'):

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')

        if mode=='velodyne':
            self.sensor_dir='velodyne_points'
        elif mode=='sick':
            self.sensor_dir='sick_points'
        else:
            raise RuntimeError('Unknown sensor type!')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath  = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')

        self.kitti360Path = kitti360Path
        self.sequence = sequence
        self.loadPoses()
        self.loadExtrinsics()

    # poses are required to unwrap velodyne points to compensate for ego-motion
    def loadPoses(self):
        # load poses
        filePoses = os.path.join(self.kitti360Path, 'data_poses', self.sequence, 'poses.txt')
        poses = np.loadtxt(filePoses)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        self.Tr_pose_world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses): 
            pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
            self.Tr_pose_world[frame] = pose

    def loadExtrinsics(self):
        # cam_0 to velo
        fileCameraToVelo = os.path.join(self.kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
        TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

        # all cameras to system center 
        fileCameraToPose = os.path.join(self.kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
        TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)
  
        self.TrVeloToPose = TrCamToPose['image_00'] @ np.linalg.inv(TrCam0ToVelo)

        # velodyne to all cameras
        self.TrVeloToCam = {}
        for k, v in TrCamToPose.items():
            # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
            TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
            TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
            # Tr(velo -> cam_k)
            self.TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,4])
        return pcd 

    def loadSickData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,2])
        pcd = np.concatenate([np.zeros_like(pcd[:,0:1]), -pcd[:,0:1], pcd[:,1:2]], axis=1)
        return pcd 

    def curlParameterFromPoses(self, frame):
        Tr_pose_pose = np.eye(4)

        if frame in self.Tr_pose_world.keys():
            if frame==1:
                if frame+1 in self.Tr_pose_world.keys():
                    Tr_pose_pose = np.linalg.inv(self.Tr_pose_world[frame+1]) @ self.Tr_pose_world[frame]
            else:
                if frame-1 in self.Tr_pose_world.keys():
                    Tr_pose_pose = np.linalg.inv(self.Tr_pose_world[frame]) @ self.Tr_pose_world[frame-1]
        Tr_delta = np.linalg.inv(self.TrVeloToPose) @ Tr_pose_pose @ self.TrVeloToPose
        
        r = Rodrigues(Tr_delta[0:3,0:3])
        t = Tr_delta[0:3,3]
        return r.flatten(),t


    def curlVelodyneData(self, frame, pcd):
        pcd=pcd.astype(np.float64)
        pcd_curled = np.copy(pcd) 
        # get curl parameters 
        r,t = self.curlParameterFromPoses(frame)
        # unwrap points to compensate for ego motion
        pcd_curled = curlVelodyneData.cCurlVelodyneData(pcd, pcd_curled, r, t)
        return pcd_curled.astype(np.float32)
        

def SaveVeloToImage(cam_id=0, seq=0, f_tgt=None, out_file=None, vis=False):
    from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
    from PIL import Image
    import matplotlib.pyplot as plt

    if f_tgt is None:
        f_new = h / 2
    else:
        f_new = f_tgt

    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')
    
    sequence = '2013_05_28_drive_%04d_sync'%seq

    # perspective camera
    if cam_id in [0,1]:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye camera
    elif cam_id in [2,3]:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
        grid_fisheye = np.load(os.path.join(kitti360Path, 'fisheye', f'grid_fisheye_0{cam_id}.npy'))
    else:
        raise RuntimeError('Unknown camera ID!')

    # object for parsing 3d raw data 
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq)
    
    # take the rectification into account for perspective cameras
    if cam_id==0 or cam_id == 1:
        TrVeloToRect = np.matmul(camera.R_rect, velo.TrVeloToCam['image_%02d' % cam_id])
    else:
        TrVeloToRect = velo.TrVeloToCam['image_%02d' % cam_id]

    # color map for visualizing depth map
    # cm = plt.get_cmap('jet')

    sub_dir = 'data_rect' if cam_id in [0,1] else 'data_rgb'
    img_dir = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir)
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    min_id = int(os.path.splitext(os.path.basename(img_files[0]))[0])
    max_id = int(os.path.splitext(os.path.basename(img_files[-1]))[0])
    depth_dir_undistorted = os.path.join(kitti360Path, 'proj_depth_undistorted', sequence, 'image_%02d' % cam_id, f'f{f_new}')
    # create depth directory if it doesn't exist
    if not os.path.exists(depth_dir_undistorted):
        os.makedirs(depth_dir_undistorted)
    img_dir_undistorted = os.path.join(kitti360Path, 'data_2d_undistorted', sequence, 'image_%02d' % cam_id, f'f{f_new}')
    # create undistorted image directory if it doesn't exist
    if not os.path.exists(img_dir_undistorted):
        os.makedirs(img_dir_undistorted)

    # visualize a set of frame
    # for each frame, load the raw 3D scan and project to image plane
    for frame in range(min_id, max_id, 10):
        print(f'Processing f_new {f_new}, seq {seq}, cam {cam_id}, frame {frame} in [{min_id}, {max_id}]')
        
        points = velo.loadVelodyneData(frame)
        # curl velodyne
        points = velo.curlVelodyneData(frame, points)

        points[:,3] = 1

        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:,:3]
        # project to image space (this depth is distance from camera center for fisheye cameras, but zbuffer for perspective cameras)
        u, v, depth= camera.cam2image(pointsCam.T)
        u = u.astype(np.int32)
        v = v.astype(np.int32)
        
        # undistort fisheye image 
        imagePath = os.path.join(img_dir, '%010d.png' % frame)
        if not os.path.isfile(imagePath):
            raise RuntimeError('Image file %s does not exist!' % imagePath)
        colorImage = np.array(Image.open(imagePath))
        h, w = camera.height, camera.width
        K = np.array([[camera.fi['projection_parameters']['gamma1'], 0., camera.fi['projection_parameters']['u0']],
                      [0., camera.fi['projection_parameters']['gamma2'], camera.fi['projection_parameters']['v0']],
                      [0., 0., 1.]], dtype=np.float32)
        distCoeffs = np.array([camera.fi['distortion_parameters']['k1'], camera.fi['distortion_parameters']['k2'],
                               camera.fi['distortion_parameters']['p1'], camera.fi['distortion_parameters']['p2']], dtype=np.float32)
        xi = camera.fi['mirror_parameters']['xi']
        
        K_new = np.array([[f_new, 0, w / 2],
                          [0, f_new, h / 2],
                          [0, 0, 1]]).astype(np.float32)
        colorImageUndistort = cv2.omnidir.undistortImage(colorImage, K, distCoeffs, np.array(xi).astype(np.float32), cv2.omnidir.RECTIFY_PERSPECTIVE, Knew=K_new)
        undistort_imgPath = os.path.join(img_dir_undistorted, '%010d.png' % frame)
        Image.fromarray(colorImageUndistort).save(undistort_imgPath)

        # prepare depth map for visualization
        depthMapUndistort = np.zeros((camera.height, camera.width))
        # convert u, v to undistorted fisheye image coordinates
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
        depth = depth[mask]
        coords_norm = grid_fisheye[v[mask], u[mask], :]
        u_new = coords_norm[:, 0] / coords_norm[:, 2] * f_new + w / 2
        v_new = coords_norm[:, 1] / coords_norm[:, 2] * f_new + h / 2
        u_new = u_new.astype(np.int32)
        v_new = v_new.astype(np.int32)
        
        mask = np.logical_and(np.logical_and(np.logical_and(u_new>=0, u_new<camera.width), v_new>=0), v_new<camera.height)
        # visualize points within 80 meters
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<80)
        depthMapUndistort[v_new[mask],u_new[mask]] = depth[mask]
        
        # save depth map
        depthPath = os.path.join(depth_dir_undistorted, '%010d.png' % frame)
        depthMapUndistort = (depthMapUndistort * 256).astype(np.int16)
        Image.fromarray(depthMapUndistort).save(depthPath)
        
        # write to out_file
        imagePathRel = os.path.join('data_2d_undistorted', sequence, 'image_%02d' % cam_id, f'f{f_new}', '%010d.png' % frame)
        depthPathRel = os.path.join('proj_depth_undistorted', sequence, 'image_%02d' % cam_id, f'f{f_new}', '%010d.png' % frame)

        out_str = f'{imagePathRel} {depthPathRel} {f_new:.4f}'
        if os.path.isfile(out_file):
            mode = 'a'  # append to existing file
        else:
            mode = 'w'  # create new file
        
        with open(out_file, mode) as f:
            f.write(out_str + '\n')
        
        if vis:
            layout = (2,1) if cam_id in [0,1] else (1,2)
            fig, axs = plt.subplots(*layout, figsize=(18,12))
            cm = plt.get_cmap('jet')

            # load RGB image for visualization
            # color map for visualizing depth map
            depthMapUndistort[depthMapUndistort>(30*256)] = 0 # only visualize points within 30 meters
            depthImageUndistort = cm(depthMapUndistort/depthMapUndistort.max())[...,:3]
            colorImageUndistort = colorImageUndistort.astype(np.float64) / 255.
            colorImageUndistort[depthMapUndistort>0] = depthImageUndistort[depthMapUndistort>0]

            axs[0].imshow(depthMapUndistort, cmap='jet', interpolation='none')
            axs[0].title.set_text('Projected Depth')
            axs[0].axis('off')
            axs[1].imshow(colorImageUndistort)
            axs[1].title.set_text('Projected Depth Overlaid on Image')
            axs[1].axis('off')
            plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
            plt.show()


if __name__=='__main__':

    train_seq = [3, 4, 5, 6, 7, 9, 10]
    val_seq = [0, 2]
    # set cam_id to 0 or 1 for projection to perspective images
    #               2 or 3 for projecting to fisheye images
    cam_ids = [2, 3]
    vis=False

    # prepare training and validation data for fisheye cameras (can be implemented faster to deal all f_tgt in SaveVeloToImage function)
    for f_tgt in [700, 350, 175]:
        out_train_file = os.path.join(os.environ['KITTI360_DATASET'], f'kitti360_train_fisheye_undistort_f{f_tgt}.txt')
        out_val_file = os.path.join(os.environ['KITTI360_DATASET'], f'kitti360_val_fisheye_undistort_f{f_tgt}.txt')
        # Delete out_val_file if it exists
        if os.path.isfile(out_train_file):
            os.remove(out_train_file)
        # Delete out_val_file if it exists
        if os.path.isfile(out_val_file):
            os.remove(out_val_file)

        # prepare validation data for fisheye cameras
        for seq in val_seq:
            for cam_id in cam_ids:
                # visualize raw 3D velodyne scans in 2D
                SaveVeloToImage(seq=seq, cam_id=cam_id, f_tgt=f_tgt, out_file=out_val_file, vis=vis)
        
        # prepare training data for fisheye cameras
        for seq in train_seq:
            for cam_id in cam_ids:
                # visualize raw 3D velodyne scans in 2D
                SaveVeloToImage(seq=seq, cam_id=cam_id, f_tgt=f_tgt, out_file=out_train_file, vis=vis)
    




