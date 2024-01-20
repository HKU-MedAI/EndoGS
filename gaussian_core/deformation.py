import torch
import torch.nn as nn
import torch.nn.init as init

from gaussian_core.hexplane import HexPlaneField


class Deformation(nn.Module):
    def __init__(self, D=8, input_ch=27, input_ch_time=9, W=256, skips=[]): #, args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips

        kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 75]
        }

        self.grid = HexPlaneField(1.6, kplanes_config, [1,2,4,8]).float()
        self.onenet = self.create_net()

    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        self.feature_out = [nn.Linear(self.grid.feat_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        return nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 11+16*3))

    def query_time(self, rays_pts_emb, time_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
        h = grid_feature
        h = self.feature_out(h)
        return h

    def forward(self, rays_pts, scales=None, rotations=None, opacity=None, shs=None, time=None):
        return self.forward_dynamic(rays_pts, scales, rotations, opacity, shs, time)
    
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb):

        hidden = self.query_time(rays_pts_emb, time_emb).float()
        
        d = self.onenet(hidden)
        dx = d[:,:3]
        ds = d[:,3:6]
        dr = d[:,6:10]
        do = d[:,10:11]
        dshs = d[:,11:]

        pts = rays_pts_emb[:, :3] + dx
        scales = scales_emb[:,:3] + ds
        rotations = rotations_emb[:,:4] + dr
        opacity = opacity_emb[:,:1] + do
        shs = shs_emb + dshs.reshape([shs_emb.shape[0],16,3])

        return pts, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        return list(self.grid.parameters()) 


class deform_network(nn.Module):
    def __init__(self) :
        super(deform_network, self).__init__()
        net_width = 256
        defor_depth= 2
        posbase_pe= 10
        timebase_pe = 4
        timenet_width = 64
        timenet_output = 32
        times_ch = 2*timebase_pe+1
        scale_rotation_pe = 2
        opacity_pe = 2
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output)#, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2)#, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
        else:
            return self.forward_static(point)

    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):

        means3D, scales, rotations, opacity, shs = self.deformation_net( point,
                                                  scales,
                                                  rotations,
                                                  opacity,
                                                  shs,
                                                  times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
