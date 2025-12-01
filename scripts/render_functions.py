import torch
import torch.nn as nn

from utils import quaternion_to_rotation_matrix, convert_input_to_pose, alpha_blending, regulate

class TransferModelOrigin(nn.Module):
    def __init__(self, model, rot, trans, transform_hom, scale, domain_type):
        super(TransferModelOrigin, self).__init__()

        self.model = model
        self.rot = rot
        self.trans = trans
        self.transform_hom = transform_hom
        self.scale = scale
        self.domain_type = domain_type

    def preprocess(self, input):
        pose = convert_input_to_pose(input, self.rot, self.trans, self.transform_hom, self.scale, self.domain_type)
        return pose
    
    def call_model(self, method_name, *args, **kwargs):
        method = getattr(self.model, method_name)
        return method(*args, **kwargs)

    def call_model_preprocess(self, method_name, input, *args, **kwargs):
  
        pose = self.preprocess(input)
        method = getattr(self.model, method_name)
        
        return method(pose, *args, **kwargs)
    
    def forward(self, input):
        pose = self.preprocess(input)
        res = self.model.forward(pose)
        return res

class GsplatRGBOrigin(nn.Module):
    def __init__(self, camera_dict, scene_dict_all, bg_color=None):
        super(GsplatRGBOrigin, self).__init__()

        self.camera_dict = camera_dict
        self.preprocess(scene_dict_all, bg_color)

    def preprocess(self, scene_dict_all, bg_color):

        means, quats, opacities, scales, colors  = (scene_dict_all[key] for key in ["means", "quats", "opacities", "scales", "colors"])
        DEVCIE = means.device

        # Process Means and Convariance Matrices in World Coordinates
        N = means.size(0)
        ones = torch.ones(N, 1, device=DEVCIE)
        means_hom_world = torch.cat([means, ones], dim=1)  # [N, 4]

        Rs = quaternion_to_rotation_matrix(quats)
        Ss = torch.diag_embed(scales)
        Ms = Rs@Ss # [N, 3, 3]
        Ms_world = Ms # [N, 3, 3]

        self.scene_dict_all = {
            "means_hom_world": means_hom_world,
            "Ms_world": Ms_world,
            "opacities": opacities,
            "colors": colors,
        }

        if bg_color == None:
            fx, fy, width, height = (self.camera_dict[key] for key in ["fx", "fy", "width", "height"])
            bg_color = torch.zeros((height, width, 3), device=DEVICE)
        self.bg_color = bg_color

    def sort_gauss(self, pose):

        # Extract Parameters
        fx, fy, width, height = (self.camera_dict[key] for key in ["fx", "fy", "width", "height"])
        means_hom_world = self.scene_dict_all["means_hom_world"]
        
        N = means_hom_world.size(0)
        DEVICE = means_hom_world.device
        pose = pose.to(DEVICE)

        # Step 1: Convert from World Coordinates to Camera Coordinates
        means_hom_cam = torch.matmul(pose, means_hom_world[None, :, :].transpose(-1,-2)).transpose(-1,-2)    # [1, N, 4]
        means_cam = means_hom_cam[:, :, :3] # [1, N, 3]
        depth = means_cam[0, :, 2] # [N, ]
        # print(depth[:10])
        # print(self.scene_dict_all["opacities"][6],self.scene_dict_all["colors"][6])

        # Step 2: Filter Gaussians based on depth 
        mask = (depth >= 0.7)
        sorted_indices = torch.argsort(depth[mask])

        self.scene_dict_sorted = {
            name: attr[mask][sorted_indices]
            for name, attr in self.scene_dict_all.items()
        }


        N_masked = mask.sum().item()
        #print(f"Contains {N_masked} Gaussians.")


    def get_num(self):
        if self.scene_dict == None:
            return 0
        else:
            colors = self.scene_dict_sorted["colors"]
            return colors.size(0)

    def update_tile(self, tile_dict):
        self.tile_dict = tile_dict

    def render_alpha(self, pose, scene_dict, eps_max=1.0):

        # Extract Parameters
        fx, fy, width, height = (self.camera_dict[key] for key in ["fx", "fy", "width", "height"])
        means_hom_world, Ms_world, opacities  = (scene_dict[key] for key in ["means_hom_world", "Ms_world", "opacities"])
        hl,wl,hu,wu = (self.tile_dict[key] for key in ["hl", "wl", "hu", "wu"])

        N = opacities.size(0)
        DEVICE = opacities.device
        pose = pose.to(DEVICE)

        # Generate Mesh Grid
        pix_coord = torch.stack(torch.meshgrid(torch.arange(wl,wu), torch.arange(hl,hu), indexing='xy'), dim=-1)
        pix_coord = pix_coord.unsqueeze(0).to(DEVICE)  # [1, TH, TW, 2]

        # Step 1: Convert from World Coordinates to Camera Coordinates
        means_hom_cam = torch.matmul(pose, means_hom_world[None, :, :].transpose(-1,-2)).transpose(-1,-2)    # [1, N, 4]
        means_cam = means_hom_cam[:, :, :3] # [1, N, 3]

        us = means_cam[:, :, 0]
        vs = means_cam[:, :, 1]
        depth = means_cam[:, :,2] # [1, N]

        R_pose = pose[:, :3, :3]  # [1, 3, 3]
        Ms_cam = R_pose[:, None, :, :]@Ms_world[None, :, :, :] # [1, N, 3, 3]

        # Step 2: Prepare Matrix K and J for Coordinate Transformation
        Ks = torch.Tensor([[
            [fx, 0, width/2],
            [0, fy, height/2],
            [0,0,1]
        ]]).to(DEVICE) # [1, 3, 3]

        # tu = torch.min(depth*lim_u, torch.max(-depth*lim_u, us))
        # tv = torch.min(depth*lim_v, torch.max(-depth*lim_v, vs))

        J_00 = fx * depth # [1, N]
        J_02 = -fx * us 
        J_11 = fy * depth
        J_12 = -fy * vs

        J_00 = fx * depth # [1, N]
        J_02 = -fx * us #tu
        J_11 = fy * depth
        J_12 = -fy * vs#tv

        J_row0 = torch.stack([J_00, torch.zeros_like(J_00), J_02], dim=-1)  # [1, N, 3]
        J_row1 = torch.stack([torch.zeros_like(J_00), J_11, J_12], dim=-1)  # [1, N, 3]
        Js = torch.stack([J_row0, J_row1], dim=-2) # [1, N, 2, 3]

        # Step 3: Convert from Camera Coodinates to Pixel Coordinates
        means_hom_pix = means_cam @ Ks.transpose(1, 2) # [1, N, 3]
        means_pix = means_hom_pix[:, :, :2] # [1, N, 2]

        Ms_pix = Js@Ms_cam # [1, N, 2, 3]

        # covs_pix = Ms_pix@Ms_pix.transpose(-1,-2) # [1, N, 2, 2]
        # covs_pix_00 = covs_pix[:, :, 0, 0] # [1, N]
        # covs_pix_01 = covs_pix[:, :, 0, 1] # [1, N]
        # covs_pix_11 = covs_pix[:, :, 1, 1] # [1, N]

        # covs_pix_det = (covs_pix_00*covs_pix_11)-covs_pix_01*covs_pix_01

        Ms_pix_00 = Ms_pix[:, :, 0, 0] # [1, N]
        Ms_pix_01 = Ms_pix[:, :, 0, 1]
        Ms_pix_02 = Ms_pix[:, :, 0, 2]
        Ms_pix_10 = Ms_pix[:, :, 1, 0]
        Ms_pix_11 = Ms_pix[:, :, 1, 1]
        Ms_pix_12 = Ms_pix[:, :, 1, 2]

        covs_pix_det = (Ms_pix_00*Ms_pix_11-Ms_pix_01*Ms_pix_10)**2+(Ms_pix_00*Ms_pix_12-Ms_pix_02*Ms_pix_10)**2+(Ms_pix_01*Ms_pix_12-Ms_pix_02*Ms_pix_11)**2
        # covs_pix_det += 1e-12 # May cause error

        covs_pix_00 = Ms_pix_00**2+Ms_pix_01**2+Ms_pix_02**2
        covs_pix_01 = Ms_pix_00*Ms_pix_10+Ms_pix_01*Ms_pix_11+Ms_pix_02*Ms_pix_12
        covs_pix_11 = Ms_pix_10**2+Ms_pix_11**2+Ms_pix_12**2

        # conics_pix_00 = covs_pix_11/covs_pix_det # [1, N]
        # conics_pix_01 = -covs_pix_01/covs_pix_det
        # conics_pix_11 = covs_pix_00/covs_pix_det

        # conics_pix_0 = torch.stack([conics_pix_00, conics_pix_01], dim=-1) # [1, N, 2]
        # conics_pix_1 = torch.stack([conics_pix_01, conics_pix_11], dim=-1)
        
        # conics_pix = torch.stack([conics_pix_0, conics_pix_1], dim=-2) # [1, N, 2, 2]

        # Step 4: Compute Probability Density and Alpha at Pixel Coordinates
        pix_diff = (pix_coord[:, :, :, None, :]*depth[:, None, None, :, None]-means_pix[:, None, None, :, :])*depth[:, None, None, :, None] #[1, TH, TW, N, 2]
        
        pix_diff_0 = pix_diff[:, :, :, :, 0] #[1, TH, TW, N]
        pix_diff_1 = pix_diff[:, :, :, :, 1]

        # prob_density = pix_diff_0**2*conics_pix_00[:, None, None, :]+2*pix_diff_0*pix_diff_1*conics_pix_01[:, None, None, :]+pix_diff_1**2*conics_pix_11[:, None, None, :] #[1, TH, TW, N]
        # prob_density = 1/covs_pix_det[:, None, None, :]*(pix_diff_0**2*covs_pix_11[:, None, None, :]-2*pix_diff_0*pix_diff_1*covs_pix_01[:, None, None, :]+pix_diff_1**2*covs_pix_00[:, None, None, :]) #[1, TH, TW, N]
        
        prob_density = 1/covs_pix_det[:, None, None, :]*(\
        (pix_diff_0*Ms_pix_10[:, None, None, :]-pix_diff_1*Ms_pix_00[:, None, None, :])**2+\
        (pix_diff_0*Ms_pix_11[:, None, None, :]-pix_diff_1*Ms_pix_01[:, None, None, :])**2+\
        (pix_diff_0*Ms_pix_12[:, None, None, :]-pix_diff_1*Ms_pix_02[:, None, None, :])**2) #[1, TH, TW, N]

        prob_density = prob_density.unsqueeze(-1) #[1, TH, TW, N, 1]

        # return prob_density.unsqueeze(-1)

        alpha = opacities[None, None, None, :, :]*torch.exp(-1/2*prob_density) # [1, TH, TW, N, 1]
        alpha = -torch.nn.functional.relu(-alpha+eps_max)+eps_max 

        return alpha # [1, TH, TW, N, 1]

    def alpha_blending(self, alpha, colors):
        alpha_shifted = torch.cat([torch.zeros_like(alpha[:,:,:,0:1,:], dtype=alpha.dtype), alpha[:,:,:,:-1,:]], dim=-2)
        transmittance = regulate(torch.cumprod((1-alpha_shifted), dim=-2))

        alpha_combined= regulate((alpha*transmittance).sum(dim=-2, keepdim=True)) # [1, TH, TW, 1, 1]
        colors_combined = regulate((alpha*transmittance*colors).sum(dim=-2, keepdim=True)) # [1, TH, TW, 1, 3]

        return colors_combined, alpha_combined
    
    def render_color(self, pose):
        hl,wl,hu,wu = (self.tile_dict[key] for key in ["hl", "wl", "hu", "wu"])
        bg_color = self.bg_color[hl:hu, wl:wu, :].view(1, hu-hl, wu-wl, 1, 3)

        if self.scene_dict_sorted is None:
            print('Warning: No Gaussians to render, return background color only!')
            return bg_color.squeeze(-2)
        
        else:
            N = self.scene_dict_sorted["opacities"].size(0)
            DEVICE = self.scene_dict_sorted["opacities"].device

            alpha = self.render_alpha(pose, self.scene_dict_sorted) # [1, TH, TW, N, 1]

            # print(alpha[0,:2,:2,:5])
            colors = self.scene_dict_sorted["colors"].view(1, 1, 1, N, 3).repeat(1, hu-hl, wu-wl, 1, 1)
            # print(colors[0,:2,:2,:5])
            ones = torch.ones((1, hu-hl, wu-wl, 1, 1), device=DEVICE)
            alpha = torch.cat([alpha, ones], dim=-2) # [1, TH, TW, N+1, 1]
            #print(colors.shape, bg_color.shape)
            colors = torch.cat([colors, bg_color], dim=-2) # [1, TH, TW, N+1, 1]

            colors_combined, alpha_combined = self.alpha_blending(alpha, colors)

            return colors_combined.squeeze(-2)
        
    def forward(self, input):
        pose = input
        return self.render_color(pose)

    # def render_color(self, pose):

    #     # Batch 
    #     hl,wl,hu,wu = (self.tile_dict[key] for key in ["hl", "wl", "hu", "wu"])
    #     if self.scene_dict is None:
    #         bg_color = self.bg_color[hl:hu, wl:wu, :].view(1, hu-hl, wu-wl, 3)

    #         return bg_color
    #     else:
    #         N = self.scene_dict["opacities"].size(0)
    #         DEVICE = self.scene_dict["opacities"].device
    #         gs_batch = self.gs_batch

    #         alpha_list = []
    #         colors_list = []

    #         ####
    #         #print(self.alpha_remainder.shape)
    #         colors_batch = self.scene_dict["colors"].view(1, 1, 1, N, 3).repeat(1, hu-hl, wu-wl, 1, 1)
    #         alpha_batch = self.render_alpha(pose, self.scene_dict) # [1, TH, TW, B, 1]

    #         ones = torch.ones((1, hu-hl, wu-wl, 1, 1), device=DEVICE)
    #         bg_color = self.bg_color[hl:hu, wl:wu, :].view(1, hu-hl, wu-wl, 1, 3)
    #         # print(alpha_batch.shape, self.alpha_remainder.shape, ones.shape)
    #         # alpha_batch = torch.cat([alpha_batch, ones], dim=-2)
    #         # colors_batch = torch.cat([colors_batch, bg_color], dim=-2)
    #         if self.alpha_remainder is None:
    #             alpha_batch = torch.cat([alpha_batch, ones], dim=-2)
    #             colors_batch = torch.cat([colors_batch, bg_color], dim=-2)
    #         else:
    #             alpha_batch = torch.cat([alpha_batch, self.alpha_remainder, ones], dim=-2)
    #             print(colors_batch.shape, self.colors_remainder.shape, bg_color.shape)
    #             colors_batch = torch.cat([colors_batch, self.colors_remainder, bg_color], dim=-2)

    #         colors_combined, alpha_combined = self.alpha_blending(alpha_batch, colors_batch, "fast").split([3,1], dim=-1) 
    #         return colors_combined.squeeze(-2)

    #     ####

    #     if N==0:
    #         print("Warning None Gaussian is selected!")

    #     else:
    #         for epcoh, idx_start in enumerate(range(0, N, gs_batch)):
    #             idx_end = min(idx_start + gs_batch, N)

    #             scene_dict_batch = {
    #                 name: attr[idx_start:idx_end]
    #                 for name, attr in self.scene_dict.items()
    #             }

    #             colors_batch = scene_dict_batch["colors"].view(1, 1, 1, idx_end-idx_start, 3).repeat(1, hu-hl, wu-wl, 1, 1)
    #             alpha_batch = self.render_alpha(pose, scene_dict_batch) # [1, TH, TW, B, 1]

    #             if epcoh <=1:
    #                 colors_combined, alpha_combined = self.alpha_blending(alpha_batch, colors_batch, "slow").split([3,1], dim=-1) # [1, TH, TW, 1, 1], [1, TH, TW, 1, 3]
    #             else:
    #                 colors_combined, alpha_combined = self.alpha_blending(alpha_batch, colors_batch, "fast").split([3,1], dim=-1) 

    #             alpha_list.append(alpha_combined)
    #             colors_list.append(colors_combined)


    #     # Add Background
    #     alpha_list.append(torch.ones((1, hu-hl, wu-wl, 1, 1), device=DEVICE))
    #     colors_list.append(self.bg_color[hl:hu, wl:wu, :].view(1, hu-hl, wu-wl, 1, 3)) 
            
    #     alpha_epoch = torch.cat(alpha_list, dim=-2) # [1, TH, TW, E, 1]
    #     colors_epoch = torch.cat(colors_list, dim=-2) # [1, TH, TW, E, 1]

    #     # Alpha-Blending
    #     colors_out, alpha_out = self.alpha_blending(alpha_epoch, colors_epoch, "slow").split([3,1], dim=-1) # [1, TH, TW, 1, 1], [1, TH, TW, 1, 3]

    #     # Output
    #     res = colors_out.squeeze(-2) # [1, TH, TW, 3]
    #     return res