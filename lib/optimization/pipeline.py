"""
Main optimization module for shape from heat reconstruction.
"""

from typing import Dict, Any, List, Tuple
import torch
import numpy as np
from .wrapper import OptimizationWrapper
from ..config.optimization_config import get_optimization_config
import cv2
from lib.utils import increase_image_border
import matplotlib.pyplot as plt
import pickle as pkl
import os
from lib.utils import plt_plot_first_nimages
import skimage.filters

def runShapeOptimization(video_name: str, params: Dict[str, Any], R: torch.Tensor, T: torch.Tensor,
                    view_masks_forobj: np.ndarray, frames_temp: np.ndarray, 
                    Lopti_params: Dict[str, Any], Sopti_params: Dict[str, Any],
                    writer: Any, filename: str, exp_name: str, light_xyhalfspace_loc: List[List[float]]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Run shape from heat reconstruction optimization.
    
    Args:
        video_name: Name of the video to process
        params: General parameters dictionary
        R: Rotation matrices
        T: Translation vectors
        view_masks_forobj: View masks for object
        frames_temp: Temperature frames
        Lopti_params: Laplacian optimization parameters
        Sopti_params: Shape optimization parameters
        writer: Tensorboard writer
        filename: Output filename
        exp_name: Experiment name
        
    Returns:
        Tuple of (vertices, faces, losses)
    """
    # Initialize optimization wrapper
    opti_wrapper = OptimizationWrapper(video_name, params)
    
    # Get video-specific configuration
    config = get_optimization_config(params.workspace_path, video_name)
    
    # Initialize lists for results
    verts_numpy = []
    faces_numpy = []
    losses = []
    global_step = 0
    curr_img_size_opti = (params.start_img_size[0], params.start_img_size[1])
    
    depth_map = [None]
    shading_img = None
    normal_halfspace_img = None
    stage_num = 0
    pre_final_stage_size_idx = 0
    light_xhalfspace_loc = light_xyhalfspace_loc[0]
    light_yhalfspace_loc = light_xyhalfspace_loc[1]
    num_prefinal_stage = len(config.pre_final_stage_sizes)
    
    while curr_img_size_opti[0] <= params.max_img_size[0]:
        print(f"Current image size: {curr_img_size_opti}, stage num: {stage_num}")
        # Prepare renderers
        opti_wrapper.prepare_renderers(R, T, curr_img_size_opti)
        
        # Resize view masks
        view_masks_resized = []
        for i in range(view_masks_forobj.shape[0]):
            view_masks_resized.append(cv2.resize(view_masks_forobj[i].astype(float), 
                                               (curr_img_size_opti[1], curr_img_size_opti[0]), 
                                               interpolation=cv2.INTER_NEAREST))
        
        if params.use_symmetry_for_backside:
            back_mask = view_masks_resized[-1].copy()
            view_masks_resized.append(cv2.flip(back_mask, 1))
            
        view_masks_resized = (np.array(view_masks_resized) > 0).astype(int)
        if params.init_mesh_from_mask:
            view_masks_resized = np.any(view_masks_resized, axis=0).astype(int)[None,...]
        
        plt.subplot(1,2,1)
        plt.imshow(view_masks_resized[0])
        if depth_map[0] is not None:
            depth_map_resized = []
            for i in range(depth_map.shape[0]):
                depth_map_resized.append(cv2.resize(depth_map[i].astype(float), (curr_img_size_opti[1], curr_img_size_opti[0]), interpolation=cv2.INTER_CUBIC))
            depth_map = np.array(depth_map_resized)
            plt.subplot(1,2,2)
            plt.imshow(depth_map[0])
        plt.show()
        
        # Prepare candidate mesh
        opti_wrapper.prepare_candidate_mesh(view_masks_resized[0].copy(), depth_map[0])
        opti_wrapper.map_heatvalues_tomesh(frames_temp)
        
        # Run Laplacian optimization if needed
        if params.optimize_MinvL:
            opti_wrapper.initialize_Loptimization(Lopti_params)
            opti_wrapper.run_MinvLoptimization(writer=writer, filename=filename, exp_name=exp_name)
            optimization_results = opti_wrapper.get_optimization_results()
            
            fstr = "filter" if params.filter_data else "nofilter"
            opti_filename = video_name + f"_opti_results{curr_img_size_opti[0]}x{curr_img_size_opti[1]}-Delta{params.NUM_FRAME_DELTA}-{fstr}.pkl"
            opti_path = os.path.join(params.workspace_path, 'DATA', 'obj_study_proc',  opti_filename)
            with open(opti_path, 'wb') as f:
                pkl.dump(optimization_results, f)
        else:
            fstr = "filter" if params.filter_data else "nofilter"
            opti_filename = video_name + f"_opti_results{curr_img_size_opti[0]}x{curr_img_size_opti[1]}-Delta{params.NUM_FRAME_DELTA}-{fstr}.pkl"
            opti_path = os.path.join(params.workspace_path, 'DATA', 'obj_study_proc',  opti_filename)
            with open(opti_path, 'rb') as f:
                optimization_results = pkl.load(f)
            opti_wrapper.set_optimization_results(optimization_results)
            
        # Get heat source image
        if params.optim_heat_source:
            heat_source_img = opti_wrapper.get_heatsource_img()
            shading_img = heat_source_img
            plt_plot_first_nimages(heat_source_img.copy(), heat_source_img.shape[0])

                
        if params.use_normal_halfspace_constraint:
            normal_halfspace_img = np.zeros((2, heat_source_img.shape[1], heat_source_img.shape[2]))
            valid_locs = ~np.isnan(heat_source_img[0])
            max_heatvals = np.nanmax(heat_source_img, axis=(1, 2))
            min_heatvals = np.nanmin(heat_source_img, axis=(1, 2))
            # print(max_heatvals, min_heatvals)
            heat_img_norm = (heat_source_img - min_heatvals[:, None, None]) / (max_heatvals[:, None, None] - min_heatvals[:, None, None])

            diff_img = heat_img_norm[0] - heat_img_norm[1]
            normal_halfspace_img[0, valid_locs] = np.where((heat_img_norm[0][valid_locs] - heat_img_norm[1][valid_locs]) >= params.eps_for_halfspace_const, light_xhalfspace_loc[0], normal_halfspace_img[0, valid_locs])
            normal_halfspace_img[0, valid_locs] = np.where((heat_img_norm[0][valid_locs] - heat_img_norm[1][valid_locs]) <= -params.eps_for_halfspace_const, light_xhalfspace_loc[1], normal_halfspace_img[0, valid_locs])
            normal_halfspace_img[1, valid_locs] = np.where((heat_img_norm[0][valid_locs] - heat_img_norm[1][valid_locs]) >= params.eps_for_halfspace_const, light_yhalfspace_loc[0], normal_halfspace_img[1, valid_locs])
            normal_halfspace_img[1, valid_locs] = np.where((heat_img_norm[0][valid_locs] - heat_img_norm[1][valid_locs]) <= -params.eps_for_halfspace_const, light_yhalfspace_loc[1], normal_halfspace_img[1, valid_locs])
            plt_plot_first_nimages(normal_halfspace_img.copy(), normal_halfspace_img.shape[0], num_cols=normal_halfspace_img.shape[0], title="Halfspace constraint")
            
        # Initialize and run shape optimization
        opti_wrapper.initialize_Soptimization(Sopti_params, shading_img, normal_halfspace_img)
        verts_numpy, faces_numpy, losses = opti_wrapper.run_Soptimization(
            writer=writer, verts_numpy=verts_numpy, faces_numpy=faces_numpy, losses=losses, filename=filename, exp_name=exp_name
        )
        
        # Update depth map
        depth_images = opti_wrapper.get_depth_image()
        depth_images = opti_wrapper.get_depth_image()
    
        # plt.subplot(1,2,1)
        dimg = depth_images[0].copy()
        dimg[dimg == 100] = np.NaN
        plt.imshow(dimg)
        plt.show()
        depth_map = increase_image_border(depth_images, 5, bg_value=100.0)
        # Gaussian blur depth map
        dimg = depth_images[0].copy()
        dimg[dimg == 100] = params.obj_dist
        if stage_num - num_prefinal_stage < 1:
            depth_map[0] = skimage.filters.gaussian(dimg, sigma=0.3, preserve_range=True, truncate=2.0, cval=params.obj_dist)
        plt.imshow(depth_map[0])
        plt.show()
        
        if curr_img_size_opti[0] < params.max_img_size[0] and curr_img_size_opti[0]*2 > params.max_img_size[0]:
            curr_img_size_opti = params.max_img_size
        elif (stage_num - num_prefinal_stage == 0 and curr_img_size_opti[0] < params.max_img_size[0]) or (stage_num - num_prefinal_stage == 1 and params.split_final_stage_opti):
            curr_img_size_opti = params.max_img_size
        else:
            curr_img_size_opti = (curr_img_size_opti[0]*2, curr_img_size_opti[1]*2)    

        # Update stage parameters
        if len(config.pre_final_stage_sizes) == 0 or pre_final_stage_size_idx >= len(config.pre_final_stage_sizes):
            # Final stage parameters
            pass  # Parameters are already set in the configuration
        else:
            # Pre-final stage parameters
            curr_img_size_opti = opti_wrapper.get_pre_final_stage_size(pre_final_stage_size_idx)
            pre_final_stage_size_idx += 1
            
        stage_num += 1
        
    return verts_numpy, faces_numpy, losses 