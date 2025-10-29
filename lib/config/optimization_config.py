"""
Configuration module for optimization parameters.
"""

from dataclasses import dataclass
from typing import List, Optional
import yaml
from pathlib import Path

@dataclass
class Sopti_ParamsClass:
    """Configuration for a single optimization stage."""
    num_iterations: Optional[int] = None
    learning_rate: Optional[float] = None
    lambda_: Optional[float] = None
    laplacian_reg: Optional[float] = None
    normal_reg: Optional[float] = None
    lambda_symmetric_verts: Optional[float] = None
    lambda_normal_halfspace: Optional[float] = None
    lambda_reg_init_mesh: Optional[float] = None
    activate_laplacian_reg_after_step: Optional[int] = None
    activate_normal_reg_after_step: Optional[int] = None
    opti_close_to_init_mesh: Optional[bool] = None
    reduce_normal_halfspace_lambda: Optional[bool] = None

@dataclass
class GeneralParamsClass:
    use_normal_halfspace_constraint: Optional[bool] = None
    depth_onlyfor_interior: Optional[bool] = None
    use_image_symmetry_constraint: Optional[bool] = None
    eps_for_halfspace_const: Optional[float] = None
    optimize_MinvL: Optional[bool] = None

@dataclass
class VideoOptimizationConfig:
    """Configuration for video-specific optimization parameters."""
    video_name: str
    Sopti_stages: List[Sopti_ParamsClass]
    params_stages: List[GeneralParamsClass]
    split_final_stage: Optional[bool] = None
    normal_const_heatmap_ids: Optional[List[int]] = None
    pre_final_stage_sizes: Optional = None
    pre_final_stage_iterations: Optional[List[int]] = None

    def __post_init__(self):
        if self.normal_const_heatmap_ids is None:
            self.normal_const_heatmap_ids = [0, 1]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'VideoOptimizationConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        Sopti_stages = []
        for stage_data in config_data.get('stages', {}).get('Sopti_params', [{}]):
            Sopti_stages.append(Sopti_ParamsClass(**stage_data))
        
        params_stages = []
        for stage_data in config_data.get('stages', {}).get('params', [{}]):
            params_stages.append(GeneralParamsClass(**stage_data))

        # Ensure at least one stage exists
        if not Sopti_stages:
            Sopti_stages = [Sopti_ParamsClass()]
        if not params_stages:
            params_stages = [GeneralParamsClass()]
            
        return cls(
            video_name=config_data.get('video_name', Path(config_path).stem),
            Sopti_stages=Sopti_stages,
            params_stages=params_stages,
            split_final_stage=config_data.get('split_final_stage'),
            normal_const_heatmap_ids=config_data.get('normal_const_heatmap_ids'),
            pre_final_stage_sizes=config_data.get('pre_final_stage_sizes'),
            pre_final_stage_iterations=config_data.get('pre_final_stage_iterations')
        )

    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_data = {
            'video_name': self.video_name,
            'stages': {
                'Sopti_params': [vars(stage) for stage in self.Sopti_stages],
                'params': [vars(stage) for stage in self.params_stages]
            }
        }
        
        if self.split_final_stage is not None:
            config_data['split_final_stage'] = self.split_final_stage
        if self.normal_const_heatmap_ids is not None:
            config_data['normal_const_heatmap_ids'] = self.normal_const_heatmap_ids
        if self.pre_final_stage_sizes is not None:
            config_data['pre_final_stage_sizes'] = self.pre_final_stage_sizes
        if self.pre_final_stage_iterations is not None:
            config_data['pre_final_stage_iterations'] = self.pre_final_stage_iterations
            
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

def get_optimization_config(workspace_path: str, video_name: str) -> VideoOptimizationConfig:
    """Get optimization configuration for a video."""
    config_path = Path(f"{workspace_path}/config/optimization/{video_name}.yaml")
    if not config_path.exists():
        print(f"Warning: No configuration found for video: {video_name}. Using default configuration.")
        return VideoOptimizationConfig(
            video_name=video_name,
            Sopti_stages=[Sopti_ParamsClass()],
            params_stages=[GeneralParamsClass()]
        )
    return VideoOptimizationConfig.from_yaml(str(config_path))