import torch
import numpy as np
import lib.constants as constants
from tqdm import tqdm
from .laplacianOpti import LaplacianOptimization
from largesteps.geometry import compute_matrix, massmatrix_fast
from largesteps.parameterize import to_differential, from_differential
from largesteps.optimize import AdamUniform
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
import largesteps
from lib.utils import (
    render_vertices_as_image,
    compute_vertex_normals_torch,
    save_meshes_pkl,
)
from typing import Dict, Any


class ShapeOptimizationWrapper(LaplacianOptimization):

        
    def initialize_Sopti_variables(self, opti_params):
        with torch.no_grad():
            
            self.M_lsteps = compute_matrix(self.verts_sim_pt, self.faces_sim_pt, opti_params.lambda_, cotan=True)
            
            ### Optimize Thermal Diffusivity ###
            if self.params.optim_thermal_diff:
                if self.alpha_opti is None:
                    # self.alpha_opti = torch.tensor(np.sqrt(TConductivity[self.params.obj_material]), dtype=torch.float32, device=self.device)
                    self.alpha_opti = torch.tensor(np.sqrt(constants.TDiff[self.params.obj_material]), dtype=torch.float32, device=self.device)
                self.alpha_opti.requires_grad = True
            else:
                self.alpha_opti = torch.tensor(np.sqrt(constants.TDiff[self.params.obj_material]), dtype=torch.float32, device=self.device)

            
            ### Optimize Heat values at Unknown Locations ###
            if self.params.optim_heat_vals:
                if self.optimized_heat_vals is None:
                    self.optimized_heat_vals = self.u_real_atverts_pt.clone().detach()
                
                if opti_params.heat_value_opti:
                    self.optimized_heat_vals.requires_grad = True
                else:
                    self.optimized_heat_vals.requires_grad = False
            else:
                self.optimized_heat_vals = self.u_real_atverts_pt.clone().detach().requires_grad_(False)

            # Sampling probability for the time dimension
            dT_dt = (self.optimized_heat_vals[:, 1:] - self.optimized_heat_vals[:, :-1]).detach().abs()
            dT_dt = dT_dt.permute(0, 2, 1)
            self.dT_dt_weights = dT_dt / torch.sum(dT_dt, dim=-1, keepdim=True)
                
            self.unknown_heatverts_mask = torch.zeros_like(self.u_real_atverts_pt, dtype=torch.bool, device=self.device)
            self.unknown_heatverts_mask[:, :, self.invalid_projected_locs] = True

            ### Optimize Heat Capacity ###
            if self.params.optim_heat_capacity:
                if self.heat_capacity_opti  is None:
                    self.heat_capacity_opti = torch.tensor(np.sqrt(self.params.SURFACE_THICKNESS*self.params.RHO*self.params.C), dtype=torch.float32, device=self.device)
                if opti_params.heat_capacity_opti:
                    self.heat_capacity_opti.requires_grad = True
                else:
                    self.heat_capacity_opti.requires_grad = False
            else:
                self.heat_capacity_opti = torch.tensor(np.sqrt(self.params.SURFACE_THICKNESS*self.params.RHO*self.params.C), dtype=torch.float32, device=self.device)
            
            ### Optimize Convection Coefficient ###
            if self.params.optim_convection_coeff:
                if self.convection_coeff_opti is None:
                    self.convection_coeff_opti = torch.tensor(np.sqrt(constants.CONVECTION_COEFF), dtype=torch.float32, device=self.device)
                if opti_params.convection_coeff_opti:
                    self.convection_coeff_opti.requires_grad = True
                else:
                    self.convection_coeff_opti.requires_grad = False
            else:
                self.convection_coeff_opti = torch.tensor(np.sqrt(constants.CONVECTION_COEFF), dtype=torch.float32, device=self.device)

            ### Optimize Heat Source ###
            if self.params.optim_heat_source and self.heat_source_opti is None:
                # heat_source_opti = torch.rand(self.u_real_atverts_pt[:, :-1].shape, device=device, requires_grad=True)
                self.heat_source_opti = torch.rand([self.u_real_atverts_pt.shape[0], 1, self.verts_sim_pt.shape[0]], device=self.device, requires_grad=True)
                # self.heat_lsteps = to_differential(self.M_lsteps, self.heat_source_opti.permute(2,0,1).view(self.verts_sim_pt.shape[0], -1))
                self.heat_source_shape = self.heat_source_opti.shape
            elif self.params.optim_heat_source and not opti_params.heat_source_opti:
                self.heat_source_opti.requires_grad = False
                self.heat_source_shape = self.heat_source_opti.shape

            ### Optimize Mesh Scale Factor ###
            if self.params.optim_scale_factor:
                self.scale_factor = torch.tensor(1.0, device=self.device, requires_grad=True)
            else:
                self.scale_factor = torch.tensor(1.0, device=self.device, requires_grad=False)

            ### Optimize Shap e Offsets ###
            if self.params.optim_shape:
                if self.params.depth_onlyfor_interior:
                    self.update_dir_offset = torch.zeros([self.interior_vertices.shape[0]], device=self.device)
                    self.update_dir_offset += torch.rand([self.interior_vertices.shape[0]], device=self.device)*0.0001
                    self.update_dir_offset.requires_grad = True
                else:
                    self.update_dir_offset = torch.zeros([self.verts_sim_pt.shape[0]], device=self.device, requires_grad=True)
            else:
                self.update_dir_offset = torch.zeros([self.verts_sim_pt.shape[0]], device=self.device, requires_grad=False)

            if not self.params.unproject_and_update:
                cam_center = self.cameras.get_camera_center()
                cam_center_unit = cam_center / torch.norm(cam_center, dim=1, keepdim=True)
                # print(cam_center_unit.shape, cam_center_unit)
                verts_imagenum = self.cat_points_image_num_init.clone()
                verts_imagenum[self.invalid_projected_locs] = 0
                self.update_direction = cam_center_unit[verts_imagenum].squeeze()
            
            if not self.params.unproject_and_update:
                self.u_lsteps = to_differential(self.M_lsteps, self.verts_sim_pt)
            else:
                self.depth_lsteps = to_differential(self.M_lsteps, self.init_xy_depth[:, -1])
            
            if self.params.use_image_symmetry_constraint:
                meanxy = torch.ceil(torch.mean(self.cat_corres_pixel_coords_init.to(dtype=torch.float32), axis=0)).to(dtype=torch.long)
                vertex_num_img = render_vertices_as_image(torch.arange(self.verts_sim_pt.shape[0])[None, :], \
                                                          self.cat_corres_pixel_coords_init, self.cat_points_image_num_init, self.img_size, fill_value=-1)[0][0]
                
                # print(vertex_num_img)
                if self.params.symmetric_about_axis == 0 or self.params.symmetric_about_axis == -1:
                    vleft_img = vertex_num_img[:meanxy[0], :]
                    vright_img = vertex_num_img[meanxy[0]:, :]
                    vright_img_flipped = torch.flip(vright_img, dims=[0])
                    vleft_img_flipped = torch.flip(vleft_img, dims=[0])
                    vsymmetric_vert_img = torch.full([self.img_size[0], self.img_size[1]], -1, device=self.device, dtype=torch.long)
                    min_himg_size = min(vleft_img.shape[0], vright_img.shape[0])
                    vsymmetric_vert_img[meanxy[0] - min_himg_size:meanxy[0], :] = vright_img_flipped[-min_himg_size:, :]
                    vsymmetric_vert_img[meanxy[0]:meanxy[0] + min_himg_size, :] = vleft_img_flipped[:min_himg_size, :]
                    # Get symmetric index for vertices
                    self.vsymmetric_vert_indx = torch.arange(self.verts_sim_pt.shape[0], device=self.device, dtype=torch.long)
                    vsymmetric_vert_indx = (vsymmetric_vert_img[self.cat_corres_pixel_coords_init[:, 0], self.cat_corres_pixel_coords_init[:, 1]]).to(dtype=torch.long)
                    self.vsymmetric_vert_indx[vsymmetric_vert_indx != -1] = vsymmetric_vert_indx[vsymmetric_vert_indx != -1]

                if self.params.symmetric_about_axis == 1 or self.params.symmetric_about_axis == -1:
                    hleft_img = vertex_num_img[:, :meanxy[1]]
                    hright_img = vertex_num_img[:, meanxy[1]:]
                    hright_img_flipped = torch.flip(hright_img, dims=[1])
                    hleft_img_flipped = torch.flip(hleft_img, dims=[1])
                    hsymmetric_vert_img = torch.full([self.img_size[0], self.img_size[1]], -1, device=self.device, dtype=torch.long)
                    min_wimg_size = min(hleft_img.shape[1], hright_img.shape[1])
                    hsymmetric_vert_img[:, meanxy[1] - min_wimg_size:meanxy[1]] = hright_img_flipped[:, -min_wimg_size:]
                    hsymmetric_vert_img[:, meanxy[1]:meanxy[1] + min_wimg_size] = hleft_img_flipped[:, :min_wimg_size]
                    
                    # Get symmetric index for vertices
                    self.hsymmetric_vert_indx = torch.arange(self.verts_sim_pt.shape[0], device=self.device, dtype=torch.long)
                    hsymmetric_vert_indx = (hsymmetric_vert_img[self.cat_corres_pixel_coords_init[:, 0], self.cat_corres_pixel_coords_init[:, 1]]).to(dtype=torch.long)
                    self.hsymmetric_vert_indx[hsymmetric_vert_indx != -1] = hsymmetric_vert_indx[hsymmetric_vert_indx != -1]
                



    def initialize_Soptimizers(self, opti_params):
        # u_lsteps.requires_grad = True
        optim_variables = []
        if self.params.optim_shape:
            optim_variables += [self.update_dir_offset]
        if self.params.optim_heat_vals and opti_params.heat_value_opti:
            optim_variables += [self.optimized_heat_vals]
        if self.params.optim_scale_factor:
            optim_variables += [self.scale_factor]
        if self.params.optim_heat_source and opti_params.heat_source_opti:
            optim_variables += [self.heat_source_opti]
        if self.params.optim_heat_capacity and opti_params.heat_capacity_opti:
            optim_variables += [self.heat_capacity_opti]
        if self.params.optim_convection_coeff and opti_params.convection_coeff_opti:
            optim_variables += [self.convection_coeff_opti]
        if self.params.optim_thermal_cond:
            optim_variables += [self.alpha_opti]
        if self.params.optim_thermal_diff:
            optim_variables += [self.K_opti]
        self.optimizer = AdamUniform(optim_variables, opti_params.learning_rate)
        # self.optimizer = optim.Adam(optim_variables, opti_params.learning_rate)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opti_params.lr_step_size, gamma=opti_params.lr_gamma)

    def initialize_Soptimization(self, opti_params, shading_image = None, normal_halfspace_img = None):

        self.initialize_Sopti_variables(opti_params)
        self.initialize_Soptimizers(opti_params)
        self.opti_params = opti_params

        if self.params.use_shading_constraint and shading_image is not None:
            shading_image[np.isnan(shading_image)] = 0.0
            # Normalize shading image
            for i in range(shading_image.shape[0]):
                shading_image[i] = shading_image[i]/np.amax(shading_image[i])
            shading_image_pt = torch.from_numpy(shading_image).float().to(self.device)
            self.sdotn_forverts = torch.zeros([shading_image.shape[0], self.verts_sim_pt.shape[0]], device=self.device)
            for i in range(shading_image.shape[0]):
                self.sdotn_forverts[i, self.valid_projected_locs] = \
                    shading_image_pt[i, self.cat_corres_pixel_coords_init[self.valid_projected_locs, 0], self.cat_corres_pixel_coords_init[self.valid_projected_locs, 1]]
            self.light_direction = torch.from_numpy(np.array(self.params.light_directions)).to(self.device)
            self.light_direction = self.light_direction/torch.norm(self.light_direction, dim=1, keepdim=True)
            self.lit_verts = self.sdotn_forverts > 0.0
            self.unlit_verts = self.sdotn_forverts <= 0.0
            
            # Compute opposite light direction
            self.opp_light_direction = torch.stack([-self.light_direction[:, 0], self.light_direction[:, 1], self.light_direction[:, 2]], dim=1)
            self.sdotn_forverts[self.unlit_verts] = -1.0

        if self.params.use_normal_halfspace_constraint and normal_halfspace_img is not None:
            normal_halfspace_img_pt = torch.from_numpy(normal_halfspace_img).float().to(self.device)
            self.normal_halfspace_forverts = torch.zeros([normal_halfspace_img.shape[0], self.verts_sim_pt.shape[0]], device=self.device)
            for i in range(normal_halfspace_img.shape[0]):
                self.normal_halfspace_forverts[i, self.valid_projected_locs] = \
                    normal_halfspace_img_pt[i, self.cat_corres_pixel_coords_init[self.valid_projected_locs, 0], self.cat_corres_pixel_coords_init[self.valid_projected_locs, 1]]
    
    def run_Soptimization(self, writer=None, verts_numpy=[], faces_numpy=[], losses=None, filename='dummy', exp_name='v1', global_step=0):

        store_mesh_every = 20
        save_mesh_pkl_every = 50
        t = self.params.NUM_FRAME_DELTA/60.0

        if losses is None:
            losses = []

        V = self.verts_sim_pt.shape[0]
        faces_numpy.append(self.faces_sim_np)
        halfspace_itr = np.inf
        for step in tqdm(range(self.opti_params.num_iterations)):

            if not self.params.unproject_and_update:
                new_ulsteps = self.u_lsteps.clone()
                if self.params.optim_shape:
                    new_ulsteps[:, -1] = new_ulsteps[:, -1] + (self.update_dir_offset * self.update_direction[:,-1])

                verts_sim_pt = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')

                verts_sim_pt_act = (verts_sim_pt / 2.0) * constants.IMAGE_WIDTH_METRIC * (self.scale_factor**2)
                
            else:
                new_ulsteps = self.depth_lsteps.clone()
                if self.params.depth_onlyfor_interior:
                    if self.params.limit_zvals:
                        new_ulsteps[self.interior_vertices] = new_ulsteps[self.interior_vertices] + ((torch.tanh(self.update_dir_offset)) * (self.params.zfar - self.params.znear))
                    else:
                        new_ulsteps[self.interior_vertices] = new_ulsteps[self.interior_vertices] + self.update_dir_offset
                else:
                    if self.params.limit_zvals:
                        new_ulsteps = new_ulsteps + ((torch.tanh(self.update_dir_offset)) * (self.params.zfar - self.params.znear))
                    else:
                        new_ulsteps = new_ulsteps + self.update_dir_offset

                # new_ulsteps = new_ulsteps + torch.tanh(self.update_dir_offset)
                # new_ulsteps = new_ulsteps + ((torch.sigmoid(update_dir_offset) - 0.5) * (zfar - znear)/2)
                if self.M_lsteps.dtype == torch.float64:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'CholeskyD')
                else:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')
                
                # # Clamp depth values to be between zfar and znear
                # update_depth_vals = torch.clamp(update_depth_vals, min=self.params.znear, max=self.params.zfar)
                
                curr_xy_depth = self.init_xy_depth.clone()
                # curr_xy_depth[:, -1] = curr_xy_depth[:, -1] + update_depth_vals
                if self.params.depth_onlyfor_interior:
                    curr_xy_depth[self.interior_vertices, -1] = update_depth_vals[self.interior_vertices]
                else:
                    curr_xy_depth[:, -1] = update_depth_vals
                # curr_xy_depth[self.interior_vertices, -1] = curr_xy_depth[self.interior_vertices, -1] + update_depth_vals
                
                # Unproject and multiply by 1000 to convert things to mm from m
                verts_sim_pt = (self.cameras.unproject_points(curr_xy_depth, world_coordinates=True, from_ndc=False))
                verts_sim_pt_act = verts_sim_pt * (self.scale_factor**2) * 1000.0
            
            if not self.params.optim_heat_vals_all_vertices:
                new_u_real_atverts_pt = (self.u_real_atverts_pt.float() * ~self.unknown_heatverts_mask) + \
                    (self.optimized_heat_vals.float() * self.unknown_heatverts_mask)
            else:
                new_u_real_atverts_pt = self.optimized_heat_vals.float()

            # new_u_real_atverts_pt = (u_real_atverts_pt * ~self.unknown_heatverts_mask) + (new_heat_atall_verts * self.unknown_heatverts_mask)
            
            ############################################################
            curr_mesh = Meshes(verts=[verts_sim_pt], faces=[self.faces_sim_pt])

            # Simulate heat diffusion
            L = largesteps.geometry.laplacian_cot2(verts_sim_pt_act, self.faces_sim_pt)
            
            # Compute mass matrix
            Ma = massmatrix_fast(verts_sim_pt_act, self.faces_sim_pt)
            # Invert M
            Macoalesced = Ma.coalesce()
            M_indx = Macoalesced.indices()
            M_val = Macoalesced.values()
            Mv = torch.sparse_coo_tensor(M_indx, M_val*self.params.SURFACE_THICKNESS, size=Ma.shape)

            # Simulated Heat Error
            ii = torch.arange(L.shape[0], device=self.device)        
            I = torch.sparse_coo_tensor(torch.stack([ii, ii]), torch.ones(L.shape[0], device=self.device), dtype=torch.float32, size=L.shape)
            ZERO_MAT = torch.sparse_coo_tensor(torch.stack([ii, ii]), torch.zeros(L.shape[0], device=self.device), dtype=torch.float32, size=L.shape)
            lrad = Mv * 4*constants.SIGMA * constants.EPSILON * (constants.AMBIENT_TEMP**3) if self.params.ADD_RADIATION else ZERO_MAT
            rrad = Mv * 4*constants.SIGMA * constants.EPSILON * (constants.AMBIENT_TEMP**4) if self.params.ADD_RADIATION else ZERO_MAT

            convection_coeff = (self.convection_coeff_opti)**2
            lconv = torch.sparse.mm(Mv, convection_coeff*I) if self.params.ADD_CONVECTION else ZERO_MAT
            rconv = torch.sparse.mm(Mv, convection_coeff*constants.AMBIENT_TEMP*I) if self.params.ADD_CONVECTION else ZERO_MAT

            tL = t*L
            if self.params.optim_thermal_diff:
                tL_indx = tL._indices()
                tL_val = tL._values()
                alphatL = torch.sparse_coo_tensor(tL_indx, (self.alpha_opti**2)*tL_val, size=L.shape)
            else:
                alphatL = (self.alpha_opti**2) * tL
                # tL_indx = tL._indices()
                # tL_val = tL._values()
                # alphatL = torch.sparse_coo_tensor(tL_indx, self.K_opti*tL_val, size=L.shape)
            
            heat_capacity = (self.heat_capacity_opti)**2

            ldiff = Mv - alphatL

            # TODO: If you optimize for heat capacity here it might throw an error! Fix it later
            lhs_val = ldiff + t*(lrad/heat_capacity + lconv)
            u_tplus1 = new_u_real_atverts_pt.permute(2,0,1)[:, :, 1:]
            ushape = u_tplus1.shape
            u_tplus1 = u_tplus1.reshape(ushape[0], ushape[1]*ushape[2])
            lhs_val = torch.sparse.mm(lhs_val, u_tplus1).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)

            # rhs_rad_conv = torch.sparse.mm(t*(rrad + rconv)/(self.params.RHO*self.params.C), torch.ones_like(new_u_real_atverts_pt.permute(2,0,1)[:, :, :-1]))
            rhs_rad_conv = torch.sparse.mm(t*(rrad/heat_capacity + rconv), torch.ones([ushape[0], ushape[1]*ushape[2]]))
            u_t = new_u_real_atverts_pt.permute(2,0,1)[:, :, :-1]
            u_t = u_t.reshape(ushape[0], ushape[1]*ushape[2])
            rhs_val = (torch.sparse.mm(Mv, u_t) + rhs_rad_conv).reshape(ushape[0], ushape[1], ushape[2]).permute(1, 2, 0)
            
            if self.params.optim_heat_source:
                # hrelu = torch.nn.ReLU()
                # heat_lsteps = self.heat_lsteps.clone()
                # heat_lsteps = heat_lsteps + self.heat_source_opti.permute(2,0,1).reshape(self.heat_source_shape[2], -1)
                # heat_source_opti = from_differential(self.M_lsteps, heat_lsteps, 'Cholesky').permute(1,0).reshape(self.heat_source_shape)
                heat_source_opti = self.heat_source_opti
                heat_source_vals = (heat_source_opti)**2
                # heat_source_vals = hrelu(heat_source_opti)
                heat_source_rhs = t*heat_source_vals
                heat_source_rhs = torch.sparse.mm(Mv, heat_source_rhs.permute(2,0,1).reshape(self.heat_source_shape[2], -1))
                heat_source_rhs = heat_source_rhs.reshape(self.heat_source_shape[2], self.heat_source_shape[0], self.heat_source_shape[1]).permute(1,2,0)
                rhs_val += heat_source_rhs
            

            
            # pde_diff = torch.norm(lhs_val - rhs_val, p=2, dim=1)
            # pde_diff = (lhs_val - rhs_val).abs()
            # lhs val - Num videos x Num frames x Num vertices
            pde_diff = (lhs_val - rhs_val).permute(0,2,1)
            # sampled_time_dim = torch.multinomial(self.dT_dt_weights.reshape(-1, self.dT_dt_weights.shape[-1]), 10, replacement=False)
            # sampled_time_dim = sampled_time_dim.reshape(self.dT_dt_weights.shape[0], self.dT_dt_weights.shape[1], -1)
            # # Use take along dim
            # pde_diff = torch.take_along_dim(pde_diff, sampled_time_dim, dim=-1)

            pde_diff = pde_diff[self.valid_projected_locs_forview].abs()

            # Normalize pde-diff based on max value along dim 1
            if self.params.normalize_pde_diff:
                pde_diff = pde_diff/pde_diff.max(dim=1, keepdim=True)[0]

            pde_diff = pde_diff.mean(dim=-1)
            # sim_heat_loss = pde_diff.sum()
            sim_heat_loss = pde_diff.mean()
            
            # sim_heat_loss = torch.sum(pde_diff)
            
            loss = 0.0
            if self.params.optim_shape:
                edge_loss = mesh_edge_loss(curr_mesh)
                normal_reg_term = mesh_normal_consistency(curr_mesh)
                laplacian_loss = mesh_laplacian_smoothing(curr_mesh, method='uniform')
                loss += self.opti_params.mesh_edge_reg*edge_loss

                if step > self.opti_params.activate_normal_reg_after_step:
                    loss += (self.opti_params.normal_reg*normal_reg_term)
                if step > self.opti_params.activate_laplacian_reg_after_step:
                    loss += (self.opti_params.laplacian_reg*laplacian_loss)
                # willmore_energy = get_willmore_energy(verts_sim_pt_act, L, M_val)
                # loss += lambda_reg * willmore_energy

            if self.params.optim_heat_vals and self.opti_params.heat_value_opti:
                heat_reg_loss = torch.linalg.norm(self.optimized_heat_vals - self.u_real_atverts_pt, dim=1).mean()
                loss += self.opti_params.lambda_heat_reg * heat_reg_loss

            # print(heat_texture_loss.item())
            if self.params.optim_shape or self.params.optim_thermal_diff:
                loss += self.opti_params.lambda_pde * sim_heat_loss

            add_normal_halfspace_loss = False
            if self.params.use_normal_halfspace_constraint:
                if step < self.opti_params.num_steps_for_halfspace_const or self.opti_params.num_steps_for_halfspace_const == -1:
                    add_normal_halfspace_loss = True
                elif (self.opti_params.toggle_halfspace_const and ((step+1)//self.opti_params.toggle_halfspace_const_steps) == 0) or halfspace_itr < 100:
                    add_normal_halfspace_loss = True
                    if ((step+1)//self.opti_params.toggle_halfspace_const_steps) == 0:
                        halfspace_itr = 0
                    else:
                        halfspace_itr += 1
                else:
                    add_normal_halfspace_loss = False
                    
                if add_normal_halfspace_loss:
                    vertex_normals = compute_vertex_normals_torch(verts_sim_pt, self.faces_sim_pt)
                    # limit x direction of vertex normals to satisfy the halfspace constraint
                    normal_halfspace_loss = 0.0
                    for i in range(self.normal_halfspace_forverts.shape[0]):
                        # normal_halfspace_loss += torch.maximum(-(vertex_normals[self.interior_vertices, i] * self.normal_halfspace_forverts[i][self.interior_vertices]) + 0.1, torch.tensor(0.0)).sum()
                        normal_halfspace_loss += torch.maximum(-((vertex_normals[self.interior_vertices, i] - 0.0) * self.normal_halfspace_forverts[i][self.interior_vertices]), torch.tensor(0.0)).mean()
                        # halfspace_loss = torch.maximum(-(vertex_normals[self.interior_vertices, i] * self.normal_halfspace_forverts[i][self.interior_vertices]), torch.tensor(-0.01))
                    loss += self.opti_params.lambda_normal_halfspace * normal_halfspace_loss
            
            if self.opti_params.boundary_normal_const or self.opti_params.add_willmore_loss:
                vertex_normals = compute_vertex_normals_torch(verts_sim_pt, self.faces_sim_pt)
                boundary_normals = vertex_normals[self.boundary_vertices]
                boundary_normals = boundary_normals/torch.norm(boundary_normals, dim=1, keepdim=True)
                exp_boundary_normals = self.boundary_normals.clone()
                if self.opti_params.invert_boundary_normal_dir:
                    exp_boundary_normals[:, :2] = -exp_boundary_normals[:, :2]
                boundary_normal_loss = torch.norm((boundary_normals - exp_boundary_normals), dim=1).mean()
                loss += self.opti_params.lambda_boundary_normal * boundary_normal_loss
                
            if self.opti_params.add_willmore_loss and not add_normal_halfspace_loss:
                willmore_energy = self.get_willmore_energy(verts_sim_pt_act, L, M_val)
                loss += self.opti_params.willmore_reg * willmore_energy

            if self.opti_params.opti_close_to_init_mesh:
                reg_loss = torch.mean(torch.norm(verts_sim_pt - self.verts_sim_pt, dim=-1))
                loss += self.opti_params.lambda_reg_init_mesh * reg_loss

            symmetric_verts_loss_np = 0.0
            if self.params.use_image_symmetry_constraint and (step < self.opti_params.num_steps_for_symmetric_verts or self.opti_params.num_steps_for_symmetric_verts == -1):
                # Get symmetric vertices
                if self.params.symmetric_about_axis == 0 or self.params.symmetric_about_axis == -1:
                    vsymmetric_verts_depth = curr_xy_depth[self.vsymmetric_vert_indx, -1]
                    vsymmetric_verts_loss = torch.mean((curr_xy_depth[:, -1] - vsymmetric_verts_depth)**2)
                    loss += self.opti_params.lambda_symmetric_verts * vsymmetric_verts_loss
                    symmetric_verts_loss_np += vsymmetric_verts_loss.item()
                if self.params.symmetric_about_axis == 1 or self.params.symmetric_about_axis == -1:
                    hsymmetric_verts_depth = curr_xy_depth[self.hsymmetric_vert_indx, -1]
                    hsymmetric_verts_loss = torch.mean((curr_xy_depth[:, -1] - hsymmetric_verts_depth)**2)
                    loss += self.opti_params.lambda_symmetric_verts * hsymmetric_verts_loss
                    symmetric_verts_loss_np += hsymmetric_verts_loss.item()


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            with torch.no_grad():
                losses.append(loss.item())
                if self.params.optim_shape:
                    if step % store_mesh_every == 0:
                        verts_numpy.append(verts_sim_pt.detach().cpu().numpy())
                        # vis_pde_loss.append(pde_diff.detach().cpu().numpy())
                    # if (step+1) % 1000 == 0 and step < 10000:
                    #     store_mesh_every += 10
                    
                    # if self.opti_params.reduce_willmore_lambda and (step+1) % self.opti_params.reduce_willmore_steps == 0:
                    #     self.opti_params.willmore_reg *= 0.7
                    #     self.opti_params.willmore_reg = max(self.opti_params.willmore_reg, 1.0)
                    if self.opti_params.reduce_normal_halfspace_lambda and (step+1) % self.opti_params.reduce_normal_halfspace_steps == 0:
                        if step < self.opti_params.num_steps_for_halfspace_const or self.opti_params.num_steps_for_halfspace_const == -1:
                            self.opti_params.lambda_normal_halfspace *= 0.7
                    
                    if self.opti_params.reduce_diffusivity and (step +1) % 1000 == 0 and step > 2000:
                        # self.alpha_opti = torch.maximum(torch.sqrt((self.alpha_opti**2) * 0.5), torch.sqrt(torch.tensor(5.0)))
                        self.alpha_opti = torch.minimum(torch.sqrt((self.alpha_opti**2) * 2.0), torch.sqrt(torch.tensor(1.0)))
                        
                    if (step+1) % save_mesh_pkl_every == 0:
                        # Use self.params.workspace_path or a global variable if defined
                        workspace_path = getattr(self.params, 'workspace_path', '')
                        save_meshes_pkl(verts_numpy, faces_numpy, workspace_path, filename, exp_name)

            if writer is not None:
                if self.params.optim_shape:
                    # writer.add_scalar('silhouette_loss', silhouette_loss.item(), global_step=step)
                    writer.add_scalar(f'laplacian_smoothing/{self.img_size[0]}x{self.img_size[1]}', laplacian_loss.item(), global_step=global_step)
                    writer.add_scalar(f'normal_consistency/{self.img_size[0]}x{self.img_size[1]}', normal_reg_term.item(), global_step=global_step)
                    writer.add_scalar(f'mesh_edge_loss/{self.img_size[0]}x{self.img_size[1]}', edge_loss.item(), global_step=global_step)
                    
                
                # writer.add_scalar('mesh_volume', mesh_vol.item(), global_step=step)
                if self.params.optim_heat_vals and self.opti_params.heat_value_opti:
                    writer.add_scalar(f'HeatOptim/ShapeOpti/heat_reg_loss/{self.img_size[0]}x{self.img_size[1]}', heat_reg_loss.item(), global_step=global_step)

                if self.params.optim_shape or self.params.optim_thermal_diff:
                    cost = sim_heat_loss.detach().cpu().numpy()
                    writer.add_scalar(f'cost_function/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', cost, global_step=global_step)
                
                if self.params.optim_scale_factor:
                    writer.add_scalar(f'scale_factor/{self.img_size[0]}x{self.img_size[1]}', self.scale_factor.item(), global_step=global_step)

                if self.params.optim_heat_source:
                    writer.add_scalar(f'HeatSourceOptim/ShapeOpti/Min(Watts)/{self.img_size[0]}x{self.img_size[1]}', heat_source_vals.min().item(), global_step=global_step)
                    writer.add_scalar(f'HeatSourceOptim/ShapeOpti/Max(Watts)/{self.img_size[0]}x{self.img_size[1]}', heat_source_vals.max().item(), global_step=global_step)
                
                if self.params.optim_heat_capacity and self.opti_params.heat_capacity_opti:
                    writer.add_scalar(f'HeatCapacityOptim/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', heat_capacity.item(), global_step=global_step)

                if self.params.optim_convection_coeff and self.opti_params.convection_coeff_opti:
                    writer.add_scalar(f'ConvectionCoeffOptim/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', convection_coeff.item(), global_step=global_step)

                if self.params.optim_thermal_diff:
                    writer.add_scalar(f'AlphaOptim/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', (self.alpha_opti**2).item(), global_step=global_step)
                
                if self.opti_params.opti_close_to_init_mesh:
                    writer.add_scalar(f'InitMeshReg/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', reg_loss.item(), global_step=global_step)

                # if self.params.use_gauss_curv_constraint:
                #     writer.add_scalar(f'CurvSignLoss/ShapeOpti/Min/{self.img_size[0]}x{self.img_size[1]}', curv_sign_loss.item(), global_step=global_step)

                # if self.params.use_shading_constraint:
                #     writer.add_scalar(f'ShadingLoss/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', sdotn_loss.item(), global_step=global_step)

                if self.params.use_normal_halfspace_constraint:
                    writer.add_scalar(f'NormalHalfspaceLoss/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', normal_halfspace_loss.item(), global_step=global_step)
                
                if self.params.use_image_symmetry_constraint:
                    writer.add_scalar(f'SymmetricVertsLoss/ShapeOpti/{self.img_size[0]}x{self.img_size[1]}', symmetric_verts_loss_np, global_step=global_step)

                global_step += 1
        
        with torch.no_grad():
            self.curr_vertex_normals = compute_vertex_normals_torch(verts_sim_pt, self.faces_sim_pt)
        return verts_numpy, faces_numpy, losses
    

    
    def get_depth_image(self):
        """
        Rendering depth image doesn't seem to work. Will fix later
        Now only works for perspective cameras. Orthographic cameras are not supported
        """
        with torch.no_grad():
            if not self.params.unproject_and_update:
                new_ulsteps = self.u_lsteps.clone()
                if self.params.optim_shape:
                    new_ulsteps[:, -1] = new_ulsteps[:, -1] + (self.update_dir_offset * self.update_direction[:,-1])

                verts_sim_pt = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')

                verts_sim_pt_act = (verts_sim_pt / 2.0) * constants.IMAGE_WIDTH_METRIC * (self.scale_factor**2)
                return None
            else:
                new_ulsteps = self.depth_lsteps.clone()
                if self.params.depth_onlyfor_interior:
                    if self.params.limit_zvals:
                        new_ulsteps[self.interior_vertices] = new_ulsteps[self.interior_vertices] + ((torch.tanh(self.update_dir_offset)) * (self.params.zfar - self.params.znear))
                    else:
                        new_ulsteps[self.interior_vertices] = new_ulsteps[self.interior_vertices] + self.update_dir_offset
                else:
                    if self.params.limit_zvals:
                        new_ulsteps = new_ulsteps + ((torch.tanh(self.update_dir_offset)) * (self.params.zfar - self.params.znear))
                    else:
                        new_ulsteps = new_ulsteps + self.update_dir_offset

                # new_ulsteps = new_ulsteps + torch.tanh(self.update_dir_offset)
                # new_ulsteps = new_ulsteps + ((torch.sigmoid(update_dir_offset) - 0.5) * (zfar - znear)/2)
                if self.M_lsteps.dtype == torch.float64:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'CholeskyD')
                else:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')
                
                # # Clamp depth values to be between zfar and znear
                # update_depth_vals = torch.clamp(update_depth_vals, min=self.params.znear, max=self.params.zfar)
                
                curr_xy_depth = self.init_xy_depth.clone()
                if self.params.depth_onlyfor_interior:
                    curr_xy_depth[self.interior_vertices, -1] = update_depth_vals[self.interior_vertices]
                else:
                    curr_xy_depth[:, -1] = update_depth_vals
                
                depth_images = np.full([self.params.num_views, self.img_size[0], self.img_size[1]], 100.0)
                cat_points_image_num_init_np = self.cat_points_image_num_init.cpu().numpy()
                cat_corres_pixel_coords_init_np = self.cat_corres_pixel_coords_init.cpu().numpy()
                for i in range(self.params.num_views):
                    valid_indx_init = np.where(cat_points_image_num_init_np[:, i] == i)[0]
                    depth_images[i, cat_corres_pixel_coords_init_np[valid_indx_init, 0], cat_corres_pixel_coords_init_np[valid_indx_init, 1]] \
                        = curr_xy_depth[valid_indx_init, -1].detach().cpu().numpy()
                
                return depth_images.copy()
            

    def get_latest_outputs(self):
        """
        Not useful at the moment. Will delete later
        """
        with torch.no_grad():
            if not self.params.unproject_and_update:
                new_ulsteps = self.u_lsteps.clone()
                if self.params.optim_shape:
                    new_ulsteps[:, -1] = new_ulsteps[:, -1] + (self.update_dir_offset * self.update_direction[:,-1])

                verts_sim_pt = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')

                length_xyz = torch.max(verts_sim_pt, dim=0)[0] - torch.min(verts_sim_pt, dim=0)[0]
                verts_sim_pt_act = (verts_sim_pt / 2.0) * constants.IMAGE_WIDTH_METRIC * (self.scale_factor**2)
                self.verts_sim_np = verts_sim_pt_act.detach().cpu().numpy()
                    
            else:
                new_ulsteps = self.depth_lsteps.clone()
                new_ulsteps = new_ulsteps + ((torch.tanh(self.update_dir_offset)) * (self.params.zfar - self.params.znear)/2)
                # new_ulsteps = new_ulsteps + ((torch.sigmoid(update_dir_offset) - 0.5) * (zfar - znear)/2)
                if self.M_lsteps.dtype == torch.float64:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'CholeskyD')
                else:
                    update_depth_vals = from_differential(self.M_lsteps, new_ulsteps, 'Cholesky')
                curr_xy_depth = self.init_xy_depth.clone()
                curr_xy_depth[:, -1] = curr_xy_depth[:, -1] + update_depth_vals
                # Unproject and multiply by 1000 to convert things to mm from m
                verts_sim_pt = (self.cameras.unproject_points(curr_xy_depth, world_coordinates=True, from_ndc=False))
                verts_sim_pt_act = verts_sim_pt * (self.scale_factor**2) * 1000.0
                
                self.verts_sim_np = verts_sim_pt_act.detach().cpu().numpy()
                self.curr_xy_depth_np = curr_xy_depth.detach().cpu().numpy()
