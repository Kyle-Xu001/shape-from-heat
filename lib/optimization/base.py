import torch
import numpy as np
import gpytoolbox
import pymesh
import pytorch3d as p3d
import cv2
import lib.constants as constants

from pytorch3d.renderer import (
    FoVOrthographicCameras, look_at_view_transform,
    RasterizationSettings, MeshRendererWithFragments, MeshRasterizer, MeshRenderer
)

from lib.utils import (
    get_perspective_camera,
    compute_vertex_normals_numpy,
    render_vertices_as_image
)

from matplotlib import pyplot as plt
import meshplot as mp
import matplotlib as mpl


class BaseOptimizationClass():

    def __init__(self, params, plot=False):
        self.params = params
        self.plot = plot
        self.device = params.device
        
        # Just initializing some optimizable variables to None
        self.K_opti = None
        self.optimized_heat_vals = None
        self.heat_source_opti = None
        self.heat_capacity_opti = None
        self.convection_coeff_opti = None
        self.alpha_opti = None
        self.HFOV = params.HFOV if hasattr(params, 'HFOV') else constants.HFOV
        self.VFOV = params.VFOV if hasattr(params, 'VFOV') else constants.VFOV
        pass

    def prepare_renderers(self, R, T, img_size):
        self.R, self.T, self.img_size = R, T, img_size
        if self.params.use_orthographic_cam_opti:
            self.cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, znear=self.params.znear, zfar=self.params.zfar)
        else:
            self.cameras = get_perspective_camera(img_size, self.HFOV, self.VFOV, R, T, self.device)
        
        raster_settings = RasterizationSettings(
            image_size=(img_size[0], img_size[1]),
            blur_radius=0,
            faces_per_pixel=1,
            clip_barycentric_coords=True,
        )

        sigma = 1e-4
        raster_settings_sh = RasterizationSettings(
            image_size=(img_size[0], img_size[1]),
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
            faces_per_pixel=50,
            clip_barycentric_coords=True,
        )

        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=p3d.renderer.mesh.shader.HardDepthShader(
                device=self.device,
                cameras=self.cameras,
            )
        )

        self.renderer_sh = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings_sh
            ),
            shader=p3d.renderer.mesh.shader.SoftSilhouetteShader()
        )
        
    def prepare_candidate_mesh(self, mask=None, depth=None):
        if self.params.init_mesh_from_mask:
            self.get_mesh_from_mask(mask, depth)
            boundary_vertices = gpytoolbox.boundary_vertices(self.faces_sim_np)
            self.boundary_vertices = torch.from_numpy(boundary_vertices).to(self.device).long()
            self.interior_vertices = torch.from_numpy(np.setdiff1d(np.arange(self.verts_sim_pt.shape[0]), boundary_vertices)).to(self.device).long()
            
    def get_mesh_from_mask(self, mask, depth = None):
        valid_pixel_locs = np.stack(np.where(mask), axis=-1)
        
        self.cat_corres_pixel_coords_init = torch.tensor(valid_pixel_locs.copy(), device=self.device, dtype=torch.long)
        self.cat_points_image_num_init = torch.full([valid_pixel_locs.shape[0], self.params.num_files], 0, device=self.device, dtype=torch.long)
        for i in range(self.cat_points_image_num_init.shape[1]):
            self.cat_points_image_num_init[:, i] = i

        # Do a simple triangulation of a 2D mask to get face indices
        triangulation = mpl.tri.Triangulation(valid_pixel_locs[:,0], valid_pixel_locs[:,1])
        triangles = triangulation.triangles
        xtri = valid_pixel_locs[:, 0][triangles] - np.roll(valid_pixel_locs[:, 0][triangles], 1, axis=1)
        ytri = valid_pixel_locs[:, 1][triangles] - np.roll(valid_pixel_locs[:, 1][triangles], 1, axis=1)
        # maxi = np.amax(np.sqrt(xtri**2 + ytri**2), axis=1)
        # trimask = maxi < 2.0
        edge_lengths = np.sqrt(xtri**2 + ytri**2)
        mean_edge_length = np.mean(edge_lengths)
        std = np.std(edge_lengths)
        trimask = edge_lengths < (mean_edge_length + 2*std)
        trimask = np.all(trimask, axis=-1)
        faces_sim_np = triangulation.triangles[trimask.astype(bool)]
    
        
        xy_vals_np = valid_pixel_locs[:, ::-1].copy() + 0.5 # Values related to Width first and then Height
        xy_vals_np[:, 1] = mask.shape[0] - xy_vals_np[:, 1]
        xy_vals_np[:, 0] = mask.shape[1] - xy_vals_np[:, 0]
        init_xy_vals = torch.from_numpy(xy_vals_np).to(device=self.device)

        if depth is not None:
            depth_vals = depth[valid_pixel_locs[:, 0], valid_pixel_locs[:, 1]]
            depth_vals_pt = torch.from_numpy(depth_vals[:, np.newaxis]).to(device=self.device)
        else:
            depth_vals_pt = torch.full([valid_pixel_locs.shape[0], 1], self.params.obj_dist, device=self.device)
        
        init_xy_depth = torch.cat([init_xy_vals, depth_vals_pt], dim=-1).float()
        init_xyz = self.cameras.unproject_points(init_xy_depth.float(), world_coordinates=True)
        
        verts_sim_np = init_xyz.cpu().numpy()

        pmesh = pymesh.form_mesh(verts_sim_np.copy(), faces_sim_np.copy())
        pmesh, info = pymesh.remove_isolated_vertices(pmesh)
        faces_sim_np = pmesh.faces
        verts_sim_np = pmesh.vertices
        valid_submesh_verts_init = info['ori_vertex_index'].astype(int)

        self.cat_corres_pixel_coords_init = self.cat_corres_pixel_coords_init[valid_submesh_verts_init]
        self.cat_points_image_num_init = self.cat_points_image_num_init[valid_submesh_verts_init]
        valid_pixel_locs = valid_pixel_locs[valid_submesh_verts_init]
        init_xy_depth = init_xy_depth[valid_submesh_verts_init]
        init_xyz = init_xyz[valid_submesh_verts_init]

        pmesh = pymesh.form_mesh(verts_sim_np.copy(), faces_sim_np.copy())
        pmeshes = pymesh.separate_mesh(pmesh)
        verts_sim_np = pmeshes[0].vertices
        faces_sim_np = pmeshes[0].faces
        valid_submesh_verts_init = pmeshes[0].get_attribute('ori_vertex_index').astype(int)

        self.cat_corres_pixel_coords_init = self.cat_corres_pixel_coords_init[valid_submesh_verts_init]
        self.cat_points_image_num_init = self.cat_points_image_num_init[valid_submesh_verts_init]
        valid_pixel_locs = valid_pixel_locs[valid_submesh_verts_init]
        init_xy_depth = init_xy_depth[valid_submesh_verts_init]
        init_xyz = init_xyz[valid_submesh_verts_init]
        
        
        self.verts_sim_pt = torch.tensor(verts_sim_np, device=self.device, dtype=torch.float32)
        self.faces_sim_pt = torch.tensor(faces_sim_np, device=self.device, dtype=torch.long)

        self.verts_sim_np = verts_sim_np
        self.faces_sim_np = faces_sim_np
        self.init_xy_depth = init_xy_depth
        self.obj_masks_pt = torch.from_numpy(mask).float().to(self.device)


        if self.plot:
            mp.plot(self.verts_sim_np, self.faces_sim_np, shading={"wireframe": True})
            # mp.plot(init_xyz.cpu().numpy(), faces_sim_np, shading={"wireframe": True})

    def set_mesh_params(self, verts_np, faces_np, cat_points_image_num_np, cat_corres_pixel_coords_np):

        self.verts_sim_np = verts_np
        self.faces_sim_np = faces_np
        self.cat_points_image_num_init = torch.from_numpy(cat_points_image_num_np).to(self.device)
        if self.cat_points_image_num_init.shape[1] != self.params.num_files:
            self.cat_points_image_num_init = torch.tile(self.cat_points_image_num_init, [1, self.params.num_files])
            for i in range(self.cat_points_image_num_init.shape[1]):
                self.cat_points_image_num_init[:, i] = i
        self.cat_corres_pixel_coords_init = torch.from_numpy(cat_corres_pixel_coords_np).to(self.device)
        self.verts_sim_pt = torch.tensor(verts_np, device=self.device, dtype=torch.float32)
        self.faces_sim_pt = torch.tensor(faces_np, device=self.device, dtype=torch.long)

        
    def map_heatvalues_tomesh(self, frames_temp):
        with torch.no_grad():
            self.u_real_atverts_pt = torch.full((frames_temp.shape[0], frames_temp.shape[1],self.verts_sim_pt.shape[0]),constants.OBJECT_TEMP, dtype=torch.float32, device=self.device)
            verts_wprojection_forview = torch.zeros([frames_temp.shape[0], self.verts_sim_pt.shape[0]], dtype=torch.bool, device=self.device)
            verts_wprojection = torch.zeros([self.verts_sim_pt.shape[0]], dtype=torch.bool, device=self.device)
            print(verts_wprojection.shape)
            
            for i in range(frames_temp.shape[0]):
                # Changing this to have separate output along the column dimension.
                valid_indx_init = torch.where(self.cat_points_image_num_init[:, i] == i)[0]
                for j in range(frames_temp.shape[1]):
                    if frames_temp[i][j].shape[0] != self.img_size[0] or frames_temp[i][j].shape[1] != self.img_size[1]:
                        tmp_frame = cv2.resize(frames_temp[i][j], (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_CUBIC)
                    else:
                        tmp_frame = frames_temp[i][j]
                    tmp_frame_pt = torch.from_numpy(tmp_frame).float().to(self.device)
                    self.u_real_atverts_pt[i, j, valid_indx_init] = tmp_frame_pt[self.cat_corres_pixel_coords_init[valid_indx_init, 0], self.cat_corres_pixel_coords_init[valid_indx_init, 1]]
                
                verts_wprojection[valid_indx_init] = True
                verts_wprojection_forview[i, valid_indx_init] = True
            
            self.u_real_atverts_np = self.u_real_atverts_pt.detach().cpu().numpy()
            
            self.valid_projected_locs = torch.where(verts_wprojection)[0].cpu().numpy()
            self.invalid_projected_locs = torch.where(~verts_wprojection)[0].cpu().numpy()
            self.valid_projected_locs_forview = torch.where(verts_wprojection_forview)

            print(self.valid_projected_locs.shape, self.invalid_projected_locs.shape, verts_wprojection.shape, self.cat_points_image_num_init.shape)

    def get_vertex_quantity_as_img(self, vertex_quantity, renderNviews=None):
        """
        vertex_quantity: (N, V) numpy array
        """
        vq_img = render_vertices_as_image(vertex_quantity, self.cat_corres_pixel_coords_init, self.cat_points_image_num_init, self.img_size, fill_value=0.0, renderNviews=renderNviews)
        return vq_img


    @property
    def num_vertices(self):
        return self.verts_sim_pt.shape[0]
    
    @property
    def num_faces(self):
        return self.faces_sim_pt.shape[0]

    @property
    def num_timesteps(self):
        return self.u_real_atverts_pt.shape[1]

    @property
    def dt(self):
        return self.params.NUM_FRAME_DELTA/60.0
