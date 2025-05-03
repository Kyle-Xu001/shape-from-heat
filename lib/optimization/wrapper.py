"""
Optimization wrapper for shape from heat reconstruction.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import numpy as np
from ..config.optimization_config import VideoOptimizationConfig, get_optimization_config
from .shapeOpti import ShapeOptimizationWrapper
import types

class OptimizationWrapper:
    """Wrapper class for optimization process."""
    
    def __init__(self, video_name: str, params):
        """
        Initialize optimization wrapper.
        
        Args:
            video_name: Name of the video to process
            params: General parameters dictionary
        """
        self.video_name = video_name
        self.params = params
        self.config = get_optimization_config(params.workspace_path, video_name)
        self.optimizer = ShapeOptimizationWrapper(params)
        self.current_stage = 0
        
    def prepare_renderers(self, R: torch.Tensor, T: torch.Tensor, img_size: Tuple[int, int]) -> None:
        """Prepare renderers for optimization."""
        self.optimizer.prepare_renderers(R, T, img_size)
        
    def prepare_candidate_mesh(self, view_mask: np.ndarray, depth_map: Optional[np.ndarray] = None) -> None:
        """Prepare candidate mesh for optimization."""
        self.optimizer.prepare_candidate_mesh(view_mask, depth_map)
        
    def map_heatvalues_tomesh(self, frames_temp: np.ndarray) -> None:
        """Map heat values to mesh."""
        self.optimizer.map_heatvalues_tomesh(frames_temp)
        
    def initialize_Loptimization(self, Lopti_params: Dict[str, Any]) -> None:
        """Initialize Laplacian optimization."""
        self.optimizer.initialize_Loptimization(Lopti_params)
        
    def run_MinvLoptimization(self, writer: Any, filename: str, exp_name: str) -> None:
        """Run MinvL optimization."""
        self.optimizer.run_MinvLoptimization(writer=writer, filename=filename, exp_name=exp_name)
        
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get optimization results."""
        return self.optimizer.get_optimization_results()
        
    def set_optimization_results(self, results: Dict[str, Any]) -> None:
        """Set optimization results."""
        self.optimizer.set_optimization_results(results)
        
    def get_heatsource_img(self) -> np.ndarray:
        """Get heat source image."""
        return self.optimizer.get_heatsource_img()
        
    def initialize_Soptimization(self, Sopti_params: Dict[str, Any], shading_image: np.ndarray, 
                               normal_halfspace_img: Optional[np.ndarray] = None) -> None:
        """Initialize shape optimization."""
        # Update parameters from current stage
        current_stage = self.config.Sopti_stages[self.current_stage]
        Sopti_params_dict = Sopti_params.__dict__
        
        # Only update parameters that are not None
        if current_stage.num_iterations is not None:
            Sopti_params_dict['num_iterations'] = current_stage.num_iterations
        if current_stage.learning_rate is not None:
            Sopti_params_dict['learning_rate'] = current_stage.learning_rate
        if current_stage.lambda_ is not None:
            Sopti_params_dict['lambda_'] = current_stage.lambda_
        if current_stage.laplacian_reg is not None:
            Sopti_params_dict['laplacian_reg'] = current_stage.laplacian_reg
        if current_stage.normal_reg is not None:
            Sopti_params_dict['normal_reg'] = current_stage.normal_reg
        if current_stage.lambda_symmetric_verts is not None:
            Sopti_params_dict['lambda_symmetric_verts'] = current_stage.lambda_symmetric_verts
        if current_stage.lambda_normal_halfspace is not None:
            Sopti_params_dict['lambda_normal_halfspace'] = current_stage.lambda_normal_halfspace
        if current_stage.lambda_reg_init_mesh is not None:
            Sopti_params_dict['lambda_reg_init_mesh'] = current_stage.lambda_reg_init_mesh
        if current_stage.activate_laplacian_reg_after_step is not None:
            Sopti_params_dict['activate_laplacian_reg_after_step'] = current_stage.activate_laplacian_reg_after_step
        if current_stage.activate_normal_reg_after_step is not None:
            Sopti_params_dict['activate_normal_reg_after_step'] = current_stage.activate_normal_reg_after_step
        if current_stage.opti_close_to_init_mesh is not None:
            Sopti_params_dict['opti_close_to_init_mesh'] = current_stage.opti_close_to_init_mesh
        if current_stage.reduce_normal_halfspace_lambda is not None:
            Sopti_params_dict['reduce_normal_halfspace_lambda'] = current_stage.reduce_normal_halfspace_lambda
            
        Sopti_params = types.SimpleNamespace(**Sopti_params_dict)
        
        # Update global parameters
        params_dict = self.params.__dict__
        current_stage_params = self.config.params_stages[self.current_stage]
        
        # Only update parameters that are not None
        if self.config.split_final_stage is not None:
            params_dict['split_final_stage_opti'] = self.config.split_final_stage
        if current_stage_params.use_normal_halfspace_constraint is not None:
            params_dict['use_normal_halfspace_constraint'] = current_stage_params.use_normal_halfspace_constraint
        if current_stage_params.depth_onlyfor_interior is not None:
            params_dict['depth_onlyfor_interior'] = current_stage_params.depth_onlyfor_interior
        if current_stage_params.use_image_symmetry_constraint is not None:
            params_dict['use_image_symmetry_constraint'] = current_stage_params.use_image_symmetry_constraint
        if current_stage_params.eps_for_halfspace_const is not None:
            params_dict['eps_for_halfspace_const'] = current_stage_params.eps_for_halfspace_const
        if self.config.normal_const_heatmap_ids is not None:
            params_dict['normal_const_heatmap_ids'] = self.config.normal_const_heatmap_ids
        params = types.SimpleNamespace(**params_dict)
        self.params = params
        self.optimizer.initialize_Soptimization(opti_params=Sopti_params, shading_image=shading_image, normal_halfspace_img=normal_halfspace_img)
        
    def run_Soptimization(self, writer: Any, verts_numpy: List[np.ndarray], 
                         faces_numpy: List[np.ndarray], losses: List[float],
                         filename: str, exp_name: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Run shape optimization."""
        verts_numpy, faces_numpy, losses = self.optimizer.run_Soptimization(
            writer=writer, verts_numpy=verts_numpy, faces_numpy=faces_numpy, losses=losses, filename=filename, exp_name=exp_name,
        )
        self.current_stage += 1
        return verts_numpy, faces_numpy, losses
        
    def get_depth_image(self) -> np.ndarray:
        """Get depth image."""
        return self.optimizer.get_depth_image()
        
    def get_pre_final_stage_size(self, idx: int) -> Optional[Tuple[int, int]]:
        """Get pre-final stage size."""
        if self.config.pre_final_stage_sizes and idx < len(self.config.pre_final_stage_sizes):
            return self.config.pre_final_stage_sizes[idx]
        return None
        
    def get_pre_final_stage_iterations(self, idx: int) -> Optional[int]:
        """Get pre-final stage iterations."""
        if self.config.pre_final_stage_iterations and idx < len(self.config.pre_final_stage_iterations):
            return self.config.pre_final_stage_iterations[idx]
        return None 