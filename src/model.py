import torch
import torch.nn as nn
import torchsparse
from torchsparse import nn as spnn
from torchsparse import SparseTensor
import kit.op as op
from kit.nn import ResNet, FOG, FCG, TargetEmbedding
from kit.op import z_order_sort, calc_morton_code

"""
UCM_Context_Model and UCM_Context_Model_1stage are the version of Paper: https://arxiv.org/abs/2510.20331 <AnyPcc: Compressing Any Point Cloud with a Single Universal Model>
"""

class UCM_Context_Model(nn.Module):
    """
    UCM for lossless compression for all point clouds
    """
    def __init__(self, channels, kernel_size, device='cpu'):
        super(UCM_Context_Model, self).__init__()
        self.device = device

        # Base components
        self.prior_embedding = nn.Embedding(256, channels)
        self.prior_resnet = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            ResNet(channels, k=kernel_size),
            ResNet(channels, k=kernel_size),
        )
        
        self.target_embedding = TargetEmbedding(channels)
        self.target_resnet = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            ResNet(channels, k=kernel_size),
            ResNet(channels, k=kernel_size),
        )

        # Spatial convolution networks
        self.group1_spatial_conv_s0 = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
        )
        
        self.group1_spatial_conv_s1 = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
        )

        # Enhanced group 2 spatial convolution (utilizes group 1 context)
        self.group2_spatial_conv_s0 = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
        )
        
        self.group2_spatial_conv_s1 = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
        )

        # Efficient neighbor feature aggregation
        self.neighbor_conv = spnn.Conv3d(channels, channels, kernel_size)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(channels * 2, channels), 
            nn.ReLU(True),
        )

        # Group predictors
        self.group1_pred_head_s0 = nn.Sequential(
            nn.Linear(channels, 32), nn.ReLU(True),
            nn.Linear(32, 32), nn.ReLU(True),
            nn.Linear(32, 16), nn.Softmax(dim=-1),
        )

        self.group1_pred_head_s1_emb = nn.Embedding(16, channels)
        self.group1_pred_head_s1 = nn.Sequential(
            nn.Linear(channels, 32), nn.ReLU(True),
            nn.Linear(32, 32), nn.ReLU(True),
            nn.Linear(32, 16), nn.Softmax(dim=-1),
        )

        self.group2_pred_head_s0 = nn.Sequential(
            nn.Linear(channels, 32), nn.ReLU(True),
            nn.Linear(32, 32), nn.ReLU(True),
            nn.Linear(32, 16), nn.Softmax(dim=-1),
        )

        self.group2_pred_head_s1_emb = nn.Embedding(16, channels)
        self.group2_pred_head_s1 = nn.Sequential(
            nn.Linear(channels, 32), nn.ReLU(True),
            nn.Linear(32, 32), nn.ReLU(True),
            nn.Linear(32, 16), nn.Softmax(dim=-1),
        )

        self.channels = channels
        self.fog = FOG()
        self.fcg = FCG()
        
        self.to(device)

    
    def get_3d_checkerboard_groups(self, coords):
        """
        3D checkerboard grouping based on (x+y+z) parity:
        Group 1 (White): (x+y+z) % 2 == 0
        Group 2 (Black): (x+y+z) % 2 == 1
        """
        coord_sum = coords[:, 1] + coords[:, 2] + coords[:, 3]
        group1_mask = (coord_sum % 2 == 0)
        group2_mask = (coord_sum % 2 == 1)
        return group1_mask, group2_mask

    def aggregate_neighbor_features_efficient(self, coords, features, group1_mask, group2_mask, occupancy):
        """
        Args:
            coords: Coordinates of all points [N, 4]
            features: Features of all points [N, channels]
            group1_mask: Mask for group 1 [N]
            group2_mask: Mask for group 2 [N]
            occupancy: Occupancy values [N]
        Returns:
            group2_enhanced_features: Enhanced features for group 2 [N_group2, channels]
        """
        masked_features = features.clone()
        masked_features[group2_mask] = 0
        
        # Integrate encoded group 1 occupancy into features
        if group1_mask.sum() > 0:
            group1_occupancy_emb = self.prior_embedding(occupancy[group1_mask].int())
            masked_features[group1_mask] += group1_occupancy_emb.squeeze()
        
        masked_sparse = SparseTensor(coords=coords, feats=masked_features)
        
        # Aggregate 26-neighborhood features via 3x3x3 convolution
        aggregated_sparse = self.neighbor_conv(masked_sparse)
        group2_enhanced_features = aggregated_sparse.feats[group2_mask]
        
        group2_original_features = features[group2_mask]
        combined_features = torch.cat([group2_original_features, group2_enhanced_features], dim=-1)
        
        final_features = self.feature_fusion(combined_features)
        return final_features

    def forward(self, x):
        N = x.coords.shape[0]

        if x.coords.device != self.device:
            x.coords = x.coords.to(self.device)
        if x.feats.device != self.device:
            x.feats = x.feats.to(self.device)

        # Multi-scale occupancy code generation
        data_ls = []
        while True:
            x = self.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone()))
            if x.coords.shape[0] < 64:
                break
        data_ls = data_ls[::-1]

        total_bits = 0
        group1_bits = 0
        group2_bits = 0

        # Layer-wise encoding
        for depth in range(len(data_ls) - 1):
            x_C, x_O = data_ls[depth]
            gt_x_up_C, gt_x_up_O = data_ls[depth + 1]
            gt_x_up_C, gt_x_up_O = z_order_sort(gt_x_up_C, gt_x_up_O)

            x_F = self.prior_embedding(x_O.int()).view(-1, self.channels)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = self.prior_resnet(x)

            x_up_C, x_up_F = self.fcg(x_C, x_O, x.feats)
            x_up_C, x_up_F = z_order_sort(x_up_C, x_up_F)

            x_up_F = self.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = self.target_resnet(x_up)

            group1_mask, group2_mask = self.get_3d_checkerboard_groups(gt_x_up_C)

            # Group 1 encoding
            if group1_mask.sum() > 0:
                group1_coords = gt_x_up_C[group1_mask]
                group1_features = x_up.feats[group1_mask]
                group1_occupancy = gt_x_up_O[group1_mask]

                group1_sparse = SparseTensor(coords=group1_coords, feats=group1_features)

                gt_s0 = torch.remainder(group1_occupancy, 16)
                gt_s1 = torch.div(group1_occupancy, 16, rounding_mode='floor')

                # Stage 0 prediction
                group1_s0_sparse = self.group1_spatial_conv_s0(group1_sparse)
                prob_s0 = self.group1_pred_head_s0(group1_s0_sparse.feats)
                selected_prob_s0 = prob_s0.gather(1, gt_s0.long())
                bits_s0 = torch.sum(torch.clamp(-torch.log2(selected_prob_s0 + 1e-10), 0, 50))

                # Stage 1 prediction
                s1_emb = self.group1_pred_head_s1_emb(gt_s0[:, 0].long())
                group1_s1_F = group1_sparse.feats + s1_emb
                group1_s1_sparse = SparseTensor(coords=group1_coords, feats=group1_s1_F)
                group1_s1_sparse = self.group1_spatial_conv_s1(group1_s1_sparse)
                prob_s1 = self.group1_pred_head_s1(group1_s1_sparse.feats)
                selected_prob_s1 = prob_s1.gather(1, gt_s1.long())
                bits_s1 = torch.sum(torch.clamp(-torch.log2(selected_prob_s1 + 1e-10), 0, 50))

                group1_bits += (bits_s0 + bits_s1)

            # Group 2 encoding
            if group2_mask.sum() > 0 and group1_mask.sum() > 0:
                group2_coords = gt_x_up_C[group2_mask]
                group2_occupancy = gt_x_up_O[group2_mask]

                enhanced_features = self.aggregate_neighbor_features_efficient(
                    gt_x_up_C, x_up.feats, group1_mask, group2_mask, gt_x_up_O
                )

                group2_sparse = SparseTensor(coords=group2_coords, feats=enhanced_features)

                gt_s0 = torch.remainder(group2_occupancy, 16)
                gt_s1 = torch.div(group2_occupancy, 16, rounding_mode='floor')

                group2_s0_sparse = self.group2_spatial_conv_s0(group2_sparse)
                prob_s0 = self.group2_pred_head_s0(group2_s0_sparse.feats)
                selected_prob_s0 = prob_s0.gather(1, gt_s0.long())
                bits_s0 = torch.sum(torch.clamp(-torch.log2(selected_prob_s0 + 1e-10), 0, 50))

                s1_emb = self.group2_pred_head_s1_emb(gt_s0[:, 0].long())
                group2_s1_F = group2_sparse.feats + s1_emb
                group2_s1_sparse = SparseTensor(coords=group2_coords, feats=group2_s1_F)
                group2_s1_sparse = self.group2_spatial_conv_s1(group2_s1_sparse)
                prob_s1 = self.group2_pred_head_s1(group2_s1_sparse.feats)
                selected_prob_s1 = prob_s1.gather(1, gt_s1.long())
                bits_s1 = torch.sum(torch.clamp(-torch.log2(selected_prob_s1 + 1e-10), 0, 50))

                group2_bits += (bits_s0 + bits_s1)

            elif group2_mask.sum() > 0 and group1_mask.sum() == 0:
                # Fallback to standard encoding if group 1 is empty
                group2_coords = gt_x_up_C[group2_mask]
                group2_features = x_up.feats[group2_mask]
                group2_occupancy = gt_x_up_O[group2_mask]

                group2_sparse = SparseTensor(coords=group2_coords, feats=group2_features)

                gt_s0 = torch.remainder(group2_occupancy, 16)
                gt_s1 = torch.div(group2_occupancy, 16, rounding_mode='floor')

                group2_s0_sparse = self.group2_spatial_conv_s0(group2_sparse)
                prob_s0 = self.group2_pred_head_s0(group2_s0_sparse.feats)
                selected_prob_s0 = prob_s0.gather(1, gt_s0.long())
                bits_s0 = torch.sum(torch.clamp(-torch.log2(selected_prob_s0 + 1e-10), 0, 50))

                s1_emb = self.group2_pred_head_s1_emb(gt_s0[:, 0].long())
                group2_s1_F = group2_sparse.feats + s1_emb
                group2_s1_sparse = SparseTensor(coords=group2_coords, feats=group2_s1_F)
                group2_s1_sparse = self.group2_spatial_conv_s1(group2_s1_sparse)
                prob_s1 = self.group2_pred_head_s1(group2_s1_sparse.feats)
                selected_prob_s1 = prob_s1.gather(1, gt_s1.long())
                bits_s1 = torch.sum(torch.clamp(-torch.log2(selected_prob_s1 + 1e-10), 0, 50))

                group2_bits += (bits_s0 + bits_s1)

        total_bits = group1_bits + group2_bits
        bpp = total_bits / N

        return bpp


class UCM_Context_Model_1stage(UCM_Context_Model):
    """
    1-stage model for lossy compression in dense point clouds
    """
    def __init__(self, channels, kernel_size, device='cpu'):
        super().__init__(channels, kernel_size, device)

        del self.neighbor_conv
        del self.feature_fusion
        
        del self.group1_spatial_conv_s0
        del self.group1_spatial_conv_s1
        del self.group2_spatial_conv_s0
        del self.group2_spatial_conv_s1
        
        del self.group1_pred_head_s0, self.group1_pred_head_s1, self.group1_pred_head_s1_emb
        del self.group2_pred_head_s0, self.group2_pred_head_s1, self.group2_pred_head_s1_emb
        
        self.pred_head = nn.Sequential(
            nn.Linear(channels, 32), nn.ReLU(True),
            nn.Linear(32, 32), nn.ReLU(True),
            nn.Linear(32, 256), nn.Softmax(dim=-1)
        )
        
        print("Initialized UCM_1STAGE: This is a 1-stage model for lossy compression.")

    def forward(self, x):
        N = x.coords.shape[0]
        if x.coords.device != self.device: x.coords = x.coords.to(self.device)
        if x.feats.device != self.device: x.feats = x.feats.to(self.device)

        data_ls = []
        while True:
            x = self.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone()))
            if x.coords.shape[0] < 64: break
        data_ls = data_ls[::-1]

        total_bits = 0

        for depth in range(len(data_ls) - 1):
            x_C, x_O = data_ls[depth]
            gt_x_up_C, gt_x_up_O = data_ls[depth + 1]
            gt_x_up_C, gt_x_up_O = z_order_sort(gt_x_up_C, gt_x_up_O)

            x_F = self.prior_embedding(x_O.int()).view(-1, self.channels)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = self.prior_resnet(x)

            x_up_C, x_up_F = self.fcg(x_C, x_O, x.feats)
            x_up_C, x_up_F = z_order_sort(x_up_C, x_up_F)

            x_up_F = self.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = self.target_resnet(x_up)

            if gt_x_up_C.shape[0] > 0:
                all_features = x_up.feats
                all_occupancy = gt_x_up_O
                
                # Single-stage 256-class prediction using features directly
                prob = self.pred_head(all_features)
                selected_prob = prob.gather(1, all_occupancy.long())
                bits = torch.sum(torch.clamp(-torch.log2(selected_prob + 1e-10), 0, 50))
                
                total_bits += bits

        bpp = total_bits / N
        return bpp


if __name__ == "__main__":
    print("1")