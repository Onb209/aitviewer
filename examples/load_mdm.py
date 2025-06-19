# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import os
import re

import joblib
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import WeakPerspectiveCamera
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer

# Set to True for rendering in headless mode, no window will be created and
# a video will be exported to 'headless/test.mp4' in the export directory
HEADLESS = False


if __name__ == "__main__":
    # Load camera and SMPL data from the output of the VIBE demo from https://github.com/mkocabas/VIBE
    # data = joblib.load(open("resources/vibe/vibe_output.pkl", "rb"))
    # input_motion = joblib.load(open("resources/vibe/07_input.pkl", "rb"))

    # input_poses = input_motion["smpl_poses"]
    # input_trans = np.zeros_like(input_motion["smpl_trans"])
    
    output_motion1 = joblib.load(open("resources/aioz/sample02_rep00_smpl_params.npy.pkl", "rb"))
    output_motion2 = joblib.load(open("resources/aioz/sample07_rep02_smpl_params.npy.pkl", "rb"))
    # output_motion3 = joblib.load(open("resources/vibe/test2.pkl", "rb"))

    output_poses1 = output_motion1["smpl_poses"]
    output_trans1 = np.zeros_like(output_motion1["smpl_trans"])
    # breakpoint()
    output_poses2 = output_motion2["smpl_poses"]
    output_trans2 = np.zeros_like(output_motion2["smpl_trans"])
    # output_poses3 = output_motion3["smpl_poses"]
    # output_trans3 = np.zeros_like(output_motion3["smpl_trans"])
    

    # Create the viewer, set a size that has 16:9 aspect ratio to match the input data
    if HEADLESS:
        viewer = HeadlessRenderer(size=(1600, 900))
    else:
        viewer = Viewer(size=(1600, 900))

    # Instantiate an SMPL sequence using the parameters from the data file.
    # We rotate the sequence by 180 degrees around the x axis to flip the y and z axis
    # because VIBE outputs the pose in a different coordinate system.
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)

    # input_motion = SMPLSequence(
    #     poses_body=input_poses[:, 3 : 24 * 3],
    #     poses_root=input_poses[:, 0:3],
    #     trans=input_trans,
    #     smpl_layer=smpl_layer,
    #     rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi),
    # )

    output_motion1 = SMPLSequence(
        poses_body=output_poses1[:, 3 : 24 * 3],
        poses_root=output_poses1[:, 0:3],
        trans=output_trans1,
        smpl_layer=smpl_layer,
        rotation=aa2rot_numpy(np.array([0, 0, 0]) * np.pi),
    )

    output_motion2 = SMPLSequence(
        poses_body=output_poses2[:, 3 : 24 * 3],
        poses_root=output_poses2[:, 0:3],
        trans=output_trans2,
        smpl_layer=smpl_layer,
        rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi),
    )

    # output_motion3 = SMPLSequence(
    #     poses_body=output_poses3[:, 3 : 24 * 3],
    #     poses_root=output_poses3[:, 0:3],
    #     trans=output_trans3,
    #     smpl_layer=smpl_layer,
    #     rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi),
    # )

    # Add all the objects to the scene.
    # viewer.scene.add(input_motion)
    viewer.scene.add(output_motion1)
    viewer.scene.add(output_motion2)
    # viewer.scene.add(output_motion3)


    # Viewer settings.
    viewer.auto_set_floor = False
    viewer.playback_fps = 25
    viewer.scene.fps = 25
    viewer.scene.floor.position = np.array([0, -1.15, 0])
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False

    if HEADLESS:
        viewer.save_video(video_dir=os.path.join(C.export_dir, "headless/vibe.mp4"), output_fps=25)
    else:
        viewer.run()
