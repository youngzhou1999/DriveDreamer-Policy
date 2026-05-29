# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imageio
import numpy as np

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.visualize import visualize_depth

import pickle

vis_flag = True

def export_to_depth_vis(
    prediction: Prediction,
    export_dir: str,
    feat
):
    global vis_flag
    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    # os.makedirs(os.path.join(export_dir, "depth_vis"), exist_ok=True)
    dep_dict = {}
    export_dir, data_dir, cams, tok = export_dir
    for idx in range(prediction.depth.shape[0]):
        # depth_vis = visualize_depth(prediction.depth[idx])
        # image_vis = images_u8[idx]
        # depth_vis = depth_vis.astype(np.uint8)
        # image_vis = image_vis.astype(np.uint8)
        # vis_image = np.concatenate([image_vis, depth_vis], axis=1)
        # vis_image = depth_vis
        export_dir_this = export_dir[idx]
        # 0421: old version: current frame
        dep_dict[cams[idx]] = prediction.depth[idx]
        # if cams[idx] not in dep_dict:
        #     dep_dict[cams[idx]] = [prediction.depth[idx]]
        # else:
        #     dep_dict[cams[idx]].append(prediction.depth[idx])

    pkl_path = os.path.join(data_dir + "-depth.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(dep_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save_path = os.path.join(export_dir_this + "-depth.jpg")
        # imageio.imwrite(save_path, vis_image, quality=95)
        # feat_map = feat[idx].detach().cpu().numpy()  # 128, 280, 504
        # save_path = os.path.join(export_dir_this + "-depth_feat.npy")
        # np.save(save_path, feat_map)

    if vis_flag:
        vis_flag=False
        depth_vis = visualize_depth(prediction.depth[0])
        image_vis = images_u8[0]
        depth_vis = depth_vis.astype(np.uint8)
        image_vis = image_vis.astype(np.uint8)
        vis_image = np.concatenate([image_vis, depth_vis], axis=1)
        save_path = os.path.join(f'./vis/{tok}' + "-depth.jpg")
        imageio.imwrite(save_path, vis_image, quality=95)


