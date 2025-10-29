from .base import BaseOptimizationClass
import torch
import numpy as np
import lib.constants as constants
import torch.optim as optim
from tqdm import tqdm

class LaplacianOptimization(BaseOptimizationClass):

    def __init__(self, params, plot=False):
        super().__init__(params, plot)
    

    def initialize_Lopti_variables(self, opti_params):
        with torch.no_grad():
            
            ### Optimize Heat values at Unknown Locations ###
            if self.params.optim_heat_vals:
                self.optimized_heat_vals = self.u_real_atverts_pt.clone().detach().requires_grad_(True)
            else:
                self.optimized_heat_vals = self.u_real_atverts_pt.clone().detach().requires_grad_(False)
            
            # Sampling probability for the time dimension
            dT_dt = (self.optimized_heat_vals[:, 1:] - self.optimized_heat_vals[:, :-1]).detach().abs()
            dT_dt = dT_dt.permute(0, 2, 1)
            self.dT_dt_weights = dT_dt / torch.sum(dT_dt, dim=-1, keepdim=True)

            self.unknown_heatverts_mask = torch.zeros_like(self.u_real_atverts_pt, dtype=torch.bool, device=self.device)
            self.unknown_heatverts_mask[:, :, self.invalid_projected_locs] = True

            ### Optimize Heat Capacity ###
            self.heat_capacity_opti = torch.tensor(np.sqrt(self.params.SURFACE_THICKNESS*self.params.RHO*self.params.C), dtype=torch.float32, device=self.device)
            if self.params.optim_heat_capacity:
                self.heat_capacity_opti.requires_grad = True
            else:
                self.heat_capacity_opti.requires_grad = False

            ### Optimize Heat Source ###
            if self.params.optim_heat_source:
                # heat_source_opti = torch.rand(self.u_real_atverts_pt[:, :-1].shape, device=device, requires_grad=True)
                self.heat_source_opti = torch.randn([self.u_real_atverts_pt.shape[0], 1, self.verts_sim_pt.shape[0]], device=self.device) * 0.001
                self.heat_source_opti.requires_grad = True
                self.heat_source_shape = self.heat_source_opti.shape
                # self.heatM_lsteps = compute_matrix(self.verts_sim_pt, self.faces_sim_pt, 1, cotan=True)
                # self.heat_lsteps = to_differential(self.heatM_lsteps, self.heat_source_opti.permute(2,0,1).view(self.verts_sim_pt.shape[0], -1))

            ### Optimize Convection Coefficient ###
            self.convection_coeff_opti = torch.tensor(np.sqrt(constants.CONVECTION_COEFF), dtype=torch.float32, device=self.device)
            if self.params.optim_convection_coeff:
                self.convection_coeff_opti.requires_grad = True
            else:
                self.convection_coeff_opti.requires_grad = False

            if self.params.optim_thermal_cond:
                # self.thermal_cond = torch.tensor(np.sqrt(constants.TDiff[self.params.obj_material]*self.params.SURFACE_THICKNESS*self.params.RHO*self.params.C), dtype=torch.float32, device=self.device)
                self.thermal_cond = torch.tensor(1.0, dtype=torch.float32, device=self.device)
                self.thermal_cond.requires_grad = True
            
            if self.params.optim_thermal_diff:
                self.thermal_diff = torch.tensor(1.0, dtype=torch.float32, device=self.device)
                self.thermal_diff.requires_grad = True

            ### Optimize Shape Offsets ###
            if self.params.optim_MinvL:
                
                L_locs = np.zeros([self.verts_sim_pt.shape[0], self.verts_sim_pt.shape[0]], dtype=bool)
                for k in range(self.faces_sim_np.shape[0]):
                    L_locs[self.faces_sim_np[k, 0], self.faces_sim_np[k, 1]] = True
                    L_locs[self.faces_sim_np[k, 1], self.faces_sim_np[k, 2]] = True
                    L_locs[self.faces_sim_np[k, 2], self.faces_sim_np[k, 0]] = True
                    L_locs[self.faces_sim_np[k, 1], self.faces_sim_np[k, 0]] = True
                    L_locs[self.faces_sim_np[k, 2], self.faces_sim_np[k, 1]] = True
                    L_locs[self.faces_sim_np[k, 0], self.faces_sim_np[k, 2]] = True

                # Make L_locs upper triangular
                L_locs = np.triu(L_locs)

                self.MinvL_locs = torch.from_numpy(np.array(np.where(L_locs))).to(device=self.device, dtype=torch.long)
                self.MinvLopti = torch.rand([self.MinvL_locs[0].shape[0]], device=self.device) + 0.001
                # self.MinvLopti = torch.ones([self.MinvL_locs[0].shape[0]], device=self.device)
                self.MinvLopti.requires_grad = True
                self.mass_opti = torch.ones([self.verts_sim_pt.shape[0]], device=self.device)
                self.mass_opti.requires_grad = True

    def initialize_Loptimizers(self, Lopti_params):
        # u_lsteps.requires_grad = True
        optim_variables = []
        if self.params.optim_MinvL:
            optim_variables += [self.MinvLopti]
            optim_variables += [self.mass_opti]
        if self.params.optim_heat_vals:
            optim_variables += [self.optimized_heat_vals]
        if self.params.optim_thermal_cond:
            optim_variables += [self.thermal_cond]
        if self.params.optim_thermal_diff:
            optim_variables += [self.thermal_diff]
        if self.params.optim_heat_source:
            optim_variables += [self.heat_source_opti]
        if self.params.optim_heat_capacity:
            optim_variables += [self.heat_capacity_opti]
        if self.params.optim_convection_coeff:
            optim_variables += [self.convection_coeff_opti]
        
        self.Loptimizer = optim.Adam(optim_variables, lr=Lopti_params.learning_rate)

        self.Llr_scheduler = torch.optim.lr_scheduler.StepLR(self.Loptimizer, step_size=Lopti_params.lr_step_size, gamma=Lopti_params.lr_gamma)

    def initialize_Loptimization(self, Lopti_params):

        self.initialize_Lopti_variables(Lopti_params)
        self.initialize_Loptimizers(Lopti_params)
        self.Lopti_params = Lopti_params

    def run_MinvLoptimization(self, writer=None, verts_numpy=[], faces_numpy=[], losses=None, filename='dummy', exp_name='v1', global_step=0):

        t = self.params.NUM_FRAME_DELTA/60.0

        if losses is None:
            losses = []

        V = self.verts_sim_pt.shape[0]
        faces_numpy.append(self.faces_sim_np)
        for step in tqdm(range(self.Lopti_params.num_iterations)):

            self.Loptimizer.zero_grad()
            if not self.params.optim_heat_vals_all_vertices:
                new_u_real_atverts_pt = (self.u_real_atverts_pt.float() * ~self.unknown_heatverts_mask) + \
                    (self.optimized_heat_vals.float() * self.unknown_heatverts_mask)
            else:
                new_u_real_atverts_pt = self.optimized_heat_vals.float()
            
            ############################################################
            L = torch.sparse_coo_tensor(self.MinvL_locs, (self.MinvLopti)**1, size=(V, V))
            L = L + L.t()
            vals = torch.sparse.sum(L, dim=-1).to_dense()
            indices = torch.arange(V, device='cuda')
            idx = torch.stack([indices, indices], dim=0)
            L = L - torch.sparse_coo_tensor(idx, vals, (V, V))
            midx = torch.stack([torch.arange(V), torch.arange(V)], dim=0)
            Minv = torch.sparse_coo_tensor(midx, (1.0/(self.mass_opti**2)), size=(V, V))
            MinvL = torch.sparse.mm(Minv, L)

            
            u_tplus1 = new_u_real_atverts_pt.permute(2,0,1)[:, :, 1:]
            ushape = u_tplus1.shape
            u_tplus1 = u_tplus1.reshape(ushape[0], ushape[1]*ushape[2])
            
            u_t = new_u_real_atverts_pt.permute(2,0,1)[:, :, :-1]
            u_t = u_t.reshape(ushape[0], ushape[1]*ushape[2])
                
            du = (u_tplus1 - u_t)
            du = (du).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)

            tMinvL = t*MinvL
            tMinvLT_t1 = torch.sparse.mm(tMinvL, u_tplus1).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)

            lhs_terms = du - tMinvLT_t1
            if self.params.ADD_CONVECTION or self.params.optim_convection_coeff:
                convection_coeff = (self.convection_coeff_opti)**2
                self.convection_coeff_np = convection_coeff.detach().cpu().numpy()
                u_surr_diff = (u_tplus1 - constants.AMBIENT_TEMP).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)
                convection_term = t*convection_coeff*u_surr_diff
                lhs_terms += convection_term

            if self.params.optim_heat_source:
                heat_source_opti = self.heat_source_opti
                heat_source_vals = (heat_source_opti)**2
                self.heat_source_vals_np = heat_source_vals.detach().cpu().numpy()
                heat_source_term = t*heat_source_vals
                lhs_terms -= heat_source_term
            
            if self.params.ADD_RADIATION:
                heat_capacity = (self.heat_capacity_opti)**2
                self.heat_capacity_np = heat_capacity.detach().cpu().numpy()
                rad_coeff = t*4*constants.SIGMA*constants.EPSILON/(heat_capacity)
                u_tplus1_pow4 = u_tplus1*(constants.AMBIENT_TEMP**3)
                surr_diff = (u_tplus1_pow4 - (constants.AMBIENT_TEMP**4))
                rad_term = (rad_coeff*surr_diff).reshape(ushape[0], ushape[1], ushape[2]).permute(1,2,0)
                lhs_terms += rad_term

            
            pde_diff = lhs_terms.permute(0,2,1)
            pde_diff = pde_diff[self.valid_projected_locs_forview]

            sim_heat_loss = (pde_diff**2).mean()
            self.sim_heat_loss_np = sim_heat_loss.item()
            # sim_heat_loss = torch.sum(pde_diff)
            
            loss = 0.0
            
            if self.params.optim_heat_vals:
                heat_reg_loss = torch.linalg.norm(self.optimized_heat_vals - self.u_real_atverts_pt, dim=1).mean()
                self.heat_reg_loss_np = heat_reg_loss.item()
                loss += self.Lopti_params.lambda_heat_reg * heat_reg_loss


            loss += self.Lopti_params.lambda_pde * sim_heat_loss
            
            loss.backward()
            self.Loptimizer.step()
            self.Llr_scheduler.step()

            if writer is not None:
                # writer.add_scalar('mesh_volume', mesh_vol.item(), global_step=step)
                if self.params.optim_heat_vals:
                    writer.add_scalar(f'HeatOptim/LaplacianOpti/heat_reg_loss/{self.img_size[0]}x{self.img_size[1]}', self.heat_reg_loss_np, global_step=global_step)

                if self.params.optim_thermal_cond:
                    writer.add_scalar(f'ThermalCondOptim/LaplacianOpti/{self.img_size[0]}x{self.img_size[1]}', (self.thermal_cond**2).item(), global_step=global_step)

                cost = self.sim_heat_loss_np
                writer.add_scalar(f'cost_function/LaplacianOpti/{self.img_size[0]}x{self.img_size[1]}', cost, global_step=global_step)
                    
                if self.params.optim_heat_source:
                    writer.add_scalar(f'HeatSourceOptim/LaplacianOpti/Min(Watts)/{self.img_size[0]}x{self.img_size[1]}', self.heat_source_vals_np.min(), global_step=global_step)
                    writer.add_scalar(f'HeatSourceOptim/LaplacianOpti/Max(Watts)/{self.img_size[0]}x{self.img_size[1]}', self.heat_source_vals_np.max(), global_step=global_step)

                if self.params.optim_heat_capacity:
                    writer.add_scalar(f'HeatCapacityOptim/LaplacianOpti/{self.img_size[0]}x{self.img_size[1]}', self.heat_capacity_np, global_step=global_step)
                
                if self.params.optim_convection_coeff:
                    writer.add_scalar(f'ConvectionCoeffOptim/LaplacianOpti/{self.img_size[0]}x{self.img_size[1]}', self.convection_coeff_np, global_step=global_step)
                
                global_step += 1
        return None
    
    def get_heatsource_img(self):
        
        with torch.no_grad():
            heat_source_opti = self.heat_source_opti
            heat_source_vals = (heat_source_opti)**2
            # heat_source_vals = hrelu(heat_source_opti)
            heat_source_vals_np = heat_source_vals.detach().cpu().numpy()

        cat_points_image_num_init_np = self.cat_points_image_num_init.cpu().numpy()
        valid_indx_init = np.where(cat_points_image_num_init_np[:, 0] == 0)[0]
        heat_source_optimized_img = np.full((heat_source_vals_np.shape[0], self.img_size[0], self.img_size[1]), np.NaN)
        cat_corres_pixel_coords_init_np = self.cat_corres_pixel_coords_init.detach().cpu().numpy()
        for it in range(heat_source_vals_np.shape[0]):
            heat_source_optimized_img[it, cat_corres_pixel_coords_init_np[valid_indx_init, 0], cat_corres_pixel_coords_init_np[valid_indx_init, 1]] \
                = heat_source_vals_np[it, 0, valid_indx_init]
        return heat_source_optimized_img
    
    def get_optimization_results(self):

        optimization_results = {}
        if self.params.optim_MinvL:
            # optim_variables += [self.MinvLopti]
            optimization_results['MinvLopti'] = self.MinvLopti.detach().cpu()
            optimization_results['MinvL_locs'] = self.MinvL_locs
            optimization_results['mass_opti'] = self.mass_opti.detach().cpu()
            V = self.verts_sim_pt.shape[0]
            L = torch.sparse_coo_tensor(self.MinvL_locs, (self.MinvLopti)**1, size=(V, V))
            L += L.t()
            vals = torch.sparse.sum(L, dim=-1).to_dense()
            indices = torch.arange(V, device='cuda')
            idx = torch.stack([indices, indices], dim=0)
            L = L - torch.sparse_coo_tensor(idx, vals, (V, V))
            optimization_results['L'] = L.detach().cpu()
                # optimization_results['symm_weight_indices'] = self.symm_weight_indices
        if self.params.optim_heat_vals:
            # optim_variables += [self.optimized_heat_vals]
            optimization_results['optimized_heat_vals'] = self.optimized_heat_vals.detach().cpu()
        if self.params.optim_thermal_diff:
            # optim_variables += [self.K_opti]
            optimization_results['thermal_diff'] = self.thermal_diff.detach().cpu()
        if self.params.optim_thermal_cond:
            optimization_results['thermal_cond'] = self.thermal_cond.detach().cpu()
        if self.params.optim_heat_source:
            # optim_variables += [self.heat_source_opti]
            optimization_results['heat_source_opti'] = self.heat_source_opti.detach().cpu()
        if self.params.optim_heat_capacity:
            # optim_variables += [self.heat_capacity_opti]
            optimization_results['heat_capacity_opti'] = self.heat_capacity_opti.detach().cpu()
        if self.params.optim_convection_coeff:
            # optim_variables += [self.convection_coeff_opti]
            optimization_results['convection_coeff_opti'] = self.convection_coeff_opti.detach().cpu()
        
        optimization_results['verts_sim_pt'] = self.verts_sim_pt.detach().cpu()
        optimization_results['faces_sim_pt'] = self.faces_sim_pt.detach().cpu()
        optimization_results['cat_points_image_num_init'] = self.cat_points_image_num_init.detach().cpu()
        optimization_results['cat_corres_pixel_coords_init'] = self.cat_corres_pixel_coords_init.detach().cpu()
        return optimization_results
    
    def get_preoptimization_dict(self):

        pkl_dict = {}

        pkl_dict['verts_sim_pt'] = self.verts_sim_pt.detach().cpu()
        pkl_dict['faces_sim_pt'] = self.faces_sim_pt.detach().cpu()
        pkl_dict['cat_points_image_num_init'] = self.cat_points_image_num_init.detach().cpu()
        pkl_dict['cat_corres_pixel_coords_init'] = self.cat_corres_pixel_coords_init.detach().cpu()
        pkl_dict['heat_values_atverts'] = self.u_real_atverts_pt.clone().detach().cpu()
        pkl_dict['params'] = self.params

        return pkl_dict

    def set_optimization_results(self, optimization_results):

        if self.params.optim_MinvL:
            self.MinvLopti = optimization_results['MinvLopti'].to(self.device)
            self.MinvLopti.requires_grad = True
        if self.params.optim_heat_vals:
            self.optimized_heat_vals = optimization_results['optimized_heat_vals'].to(self.device)
            self.optimized_heat_vals.requires_grad = True
        if self.params.optim_thermal_diff:
            self.thermal_diff = optimization_results['thermal_diff'].to(self.device)
            self.thermal_diff.requires_grad = True
        if self.params.optim_heat_source:
            self.heat_source_opti = optimization_results['heat_source_opti'].to(self.device)
            self.heat_source_opti.requires_grad = True
        if self.params.optim_heat_capacity:
            self.heat_capacity_opti = optimization_results['heat_capacity_opti'].to(self.device)
            self.heat_capacity_opti.requires_grad = True
        if self.params.optim_convection_coeff:
            self.convection_coeff_opti = optimization_results['convection_coeff_opti'].to(self.device)
            self.convection_coeff_opti.requires_grad = True

