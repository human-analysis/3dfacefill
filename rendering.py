import os
import torch
import torch.nn.functional as F
import numpy as np
import zbuffertri_batch
from _3dmm_utils import *

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def get_pixel_value(img, u, v):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, C, H, W)
    - x: flattened tensor of shape (B, 53215)
    - y: flattened tensor of shape (B, 53215)
    Returns
    -------
    - output: tensor of shape (B, C, H, W)
    """
    B, C, H, W = img.shape
    u = u.squeeze(-1)
    v = v.squeeze(-1)
    if u.dim() == 1:
        points = img[:,:,u,v].transpose(2,1)
    elif u.dim() == 2:
        points = img[torch.arange(B)[:, None], :, u, v]
    else:
        points = img[torch.arange(B)[:, None, None], :, u, v]
    return points

def shading(L, normal):
    shape = normal.shape
    normal_x, normal_y, normal_z = normal[...,0], normal[...,1], normal[...,2]
    pi = math.pi

    sh = [0]*9
    sh[0] = 1/math.sqrt(4*pi) * torch.ones_like(normal_x)
    sh[1] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_z
    sh[2] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_y
    sh[3] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_x
    sh[4] = (pi/4)*(1/2)*(math.sqrt(5/(4*pi)))*(2*torch.pow(normal_z, 2)-torch.pow(normal_x, 2)-torch.pow(normal_y, 2))
    sh[5] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_y*normal_z)
    sh[6] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_x*normal_z)
    sh[7] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_x*normal_y)
    sh[8] = (pi/4)*(3/2)*(math.sqrt(5/(12*pi)))*( torch.pow(normal_x, 2)-torch.pow(normal_y, 2))

    sh = torch.stack(sh, dim=2).unsqueeze(-2)
    L1, L2, L3 = torch.chunk(L, chunks=3, dim=1)
    L1 = L1.unsqueeze(1).expand(-1, shape[1], -1).unsqueeze(-1)
    L2 = L2.unsqueeze(1).expand(-1, shape[1], -1).unsqueeze(-1)
    L3 = L3.unsqueeze(1).expand(-1, shape[1], -1).unsqueeze(-1)

    B1 = torch.matmul(sh, L1)
    B2 = torch.matmul(sh, L2)
    B3 = torch.matmul(sh, L3)

    B = torch.stack((B1, B2, B3), dim=2).squeeze(-1).squeeze(-1)
    return B

def flatten(x):
    return x.contiguous().view(-1)

def median(x, perctile=0.36, kernel_size=3, stride=1):
    pad_px = kernel_size - stride
    pad_l = pad_px // 2
    pad_r = pad_px - pad_l
    padding = (pad_l, pad_r, pad_l, pad_r)
    idx = math.floor(perctile * kernel_size**2)

    if x.dim() == 3:
        x = x.unsqueeze(1)
    x = F.pad(x, padding, mode='reflect')
    x = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    x, i = x.contiguous().view(x.size()[:4] + (-1,)).sort(dim=-1)
    return x[...,-idx], i[...,-idx]

class Rendering:
    """docstring for Rendering"""
    def __init__(self, args):
        super(Rendering, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
        self.vertex_num = args.vertex_num
        self.tri_num = args.tri_num
        self.texture_size = args.texture_size
        self.output_size = args.resolution[0]

        self.tri = torch.tensor(load_3DMM_tri(), dtype=torch.long, device=self.device)
        # self.tri[self.tri == 53215] = 0
        self.vertex_tri = torch.tensor(load_3DMM_vertex_tri(), dtype=torch.long, device=self.device)
        v_append = torch.zeros(self.vertex_tri.shape[0], 1).to(self.device).long()
        self.vertex_tri = torch.cat((self.vertex_tri, v_append), 1)
        self.tri_2d, self.tri_2d_mask = load_3DMM_tri_2d(with_mask = True)
        padding = (self.texture_size[1] - self.tri_2d.shape[1]) // 2
        self.tri_2d = F.pad(torch.tensor(self.tri_2d, dtype=torch.long, device=self.device), pad=(padding, padding), mode='constant', value=args.tri_num).view(-1)
        self.tri_2d_mask = F.pad(torch.tensor(self.tri_2d_mask, device=self.device), pad=(padding, padding), mode='constant', value=False)
        self.tri_2d_barycoord = torch.tensor(load_3DMM_tri_2d_barycoord(), dtype=torch.float32, device=self.device)
        self.kpts = torch.tensor(load_3DMM_kpts(), dtype=torch.long, device=self.device).squeeze()
        self.vt2pixel_u, self.vt2pixel_v = load_3DMM_vt2pixel()
        self.vt2pixel_u = torch.tensor(self.vt2pixel_u, dtype=torch.float32, device=self.device)
        self.vt2pixel_v = torch.tensor(self.vt2pixel_v, dtype=torch.float32, device=self.device) + padding

        meshgrid = torch.meshgrid(torch.linspace(0.0, self.output_size-1.0, self.output_size), torch.linspace(0.0, self.output_size-1.0, self.output_size))
        self.img_u = flatten(meshgrid[0]).to(self.device)
        self.img_v = flatten(meshgrid[1]).to(self.device)

    def barycoords(self, uv_vt_map):
        u1 = uv_vt_map[:,:,:,0,0:1]
        v1 = uv_vt_map[:,:,:,0,1:2]

        u2 = uv_vt_map[:,:,:,1,0:1]
        v2 = uv_vt_map[:,:,:,1,1:2]

        u3 = uv_vt_map[:,:,:,2,0:1]
        v3 = uv_vt_map[:,:,:,2,1:2]

        u_13 = u1 - u3; u_23 = u2 - u3;
        v_13 = v1 - v3; v_23 = v2 - v3;

        u_u3 = self.img_u.view(-1,self.output_size).unsqueeze(0).unsqueeze(-1) - u3
        v_v3 = self.img_v.view(-1,self.output_size).unsqueeze(0).unsqueeze(-1) - v3

        inv_deno = 1 / (u_13 * v_23 - u_23 * v_13 + 1e-8)
        c1 = (v_23 * u_u3 - u_23 * v_v3) * inv_deno
        c2 = (u_13 * v_v3 - v_13 * u_u3) * inv_deno
        c3 = 1 - c1 - c2
        barycoords = torch.cat((c1, c2, c3), dim=-1)

        return barycoords


    # Convert camera projection vector to 4x3 matrix
    def get_camera_matrix(self, m):
        m = m.view(-1, 4, 2)

        m_row1 = F.normalize(m[:,0:3,0], dim=1)
        m_row2 = F.normalize(m[:,0:3,1], dim=1)
        m_row3 = F.pad(torch.cross(m_row1, m_row2), pad=(0, 1), mode='constant')
        m_row3 = m_row3.unsqueeze(dim=2)

        m = torch.cat([m, m_row3], dim=2)
        return m

    def get_projection(self, mshape, m):
        batch_size = mshape.shape[0]
        vertex3d = mshape.view(batch_size, -1, 3)
        vertex4d = torch.cat([vertex3d, torch.ones(vertex3d.shape[0:2], device=mshape.device).unsqueeze(-1).type(torch.float32)], dim=2)
        vertex2d = torch.matmul(vertex4d, m)
        v_append = torch.zeros(batch_size, 1, 3).to(mshape.device)
        vertex3d = torch.cat((vertex3d, v_append), 1)
        vertex2d = torch.cat((vertex2d, v_append), 1)
        return vertex3d, vertex2d

    def bilinear_sampler(self, img, u=None, v=None, mask=None):
        B, C, H, W = img.shape

        u_max = H-1
        v_max = W-1

        if u is None:
            u = self.vt2pixel_u[:-1]
        if v is None:
            v = self.vt2pixel_v[:-1]

        unmapped = (u <= -1) + (u >= u_max + 1) + (v <= -1) + (v >= v_max + 1)
        u = u.clamp(0, u_max)
        v = v.clamp(0, v_max)

        u_floor = u.floor().long()
        u_ceil = u.ceil().long()
        v_floor = v.floor().long()
        v_ceil = v.ceil().long()
        same_u = (u == u_floor)
        same_v = (v == v_floor)
        if same_u.dim() == 1:
            same_u = same_u.unsqueeze(0).expand(B,-1)
            same_v = same_v.unsqueeze(0).expand(B,-1)

        img_a = get_pixel_value(img, u_floor, v_floor)
        img_b = get_pixel_value(img, u_floor, v_ceil)
        img_c = get_pixel_value(img, u_ceil, v_floor)
        img_d = get_pixel_value(img, u_ceil, v_ceil)

        if mask is not None:
            valid_a = get_pixel_value(mask.unsqueeze(1), u_floor, v_floor).squeeze(2)
            valid_b = get_pixel_value(mask.unsqueeze(1), u_floor, v_ceil).squeeze(2)
            valid_c = get_pixel_value(mask.unsqueeze(1), u_ceil, v_floor).squeeze(2)
            valid_d = get_pixel_value(mask.unsqueeze(1), u_ceil, v_ceil).squeeze(2)
        else:
            valid_a = valid_b = valid_c = valid_d = torch.ones_like(u_floor).bool()

        wa = ((u_ceil - u) * (v_ceil - v) * valid_a).unsqueeze(-1)
        wb = ((u_ceil - u) * (v - v_floor) * valid_b).unsqueeze(-1)
        wc = ((u - u_floor) * (v_ceil - v) * valid_c).unsqueeze(-1)
        wd = ((u - u_floor) * (v - v_floor) * valid_d).unsqueeze(-1)

        # perform bilinear sampling
        img_vt = img_a * wa + img_b * wb + img_c * wc + img_d * wd
        img_vt = img_vt / (wa + wb + wc + wd + 1e-8)

        # when point lies on a row (same u)
        same_u_wa = ((v_ceil - v) * valid_a).unsqueeze(-1)
        same_u_wb = ((v - v_floor) * valid_b).unsqueeze(-1)
        img_vt[same_u] = ((img_a * same_u_wa + img_b * same_u_wb) / (same_u_wa + same_u_wb + 1e-8))[same_u]

        # when point lies on a column (same v)
        same_v_wa = ((u_ceil - u) * valid_a).unsqueeze(-1)
        same_v_wc = ((u - u_floor) * valid_c).unsqueeze(-1)
        img_vt[same_v] = ((img_a * same_v_wa + img_c * same_v_wc) / (same_v_wa + same_v_wc + 1e-8))[same_v]

        # when point lies on an exact pixel
        same = same_u * same_v
        img_vt[same] = img_a[same]

        img_vt[torch.isnan(img_vt)] = img.min()

        valid = (valid_a + valid_b + valid_c + valid_d).bool().float()
        img_vt_mask = valid * ~unmapped

        return img_vt, img_vt_mask

    def compute_normal(self, x, rot_mat):
        batch_size = x.shape[0]
        T = self.vertex_tri.shape[0]

        # Compute triangle normal using its vertices 3dlocation
        vt1 = torch.index_select(x, dim=1, index=self.tri[0])
        vt2 = torch.index_select(x, dim=1, index=self.tri[1])
        vt3 = torch.index_select(x, dim=1, index=self.tri[2])
        normalf = torch.cross(vt2-vt1, vt3-vt1)
        normalf = F.normalize(normalf, dim=2)

        # Compute vertices normal
        mask = (self.vertex_tri != self.tri.shape[1]-1).unsqueeze(2).expand(-1, -1, 3).unsqueeze(0).type(x.dtype)
        vertex_tri = self.vertex_tri.view(-1)
        normal = torch.index_select(normalf, dim=1, index=vertex_tri).view(batch_size, T, -1, 3)

        normal = (normal * mask).sum(dim=1)
        normal = F.normalize(normal, dim=2)

        # Enforce that the normal are outward
        v = x - x.mean(dim=1, keepdim=True)
        s = (v * normal).sum(dim=1, keepdim=True)
        count_s_greater_0 = (s > 0).sum(dim=0, keepdim=True)
        count_s_less_0 = (s < 0).sum(dim=0, keepdim=True)

        sign = 2 * (count_s_greater_0 > count_s_less_0).type(torch.float32) - 1
        normal *= sign
        normalf *= sign

        m_normal = torch.matmul(rot_mat, normal.transpose(2, 1)).transpose(2, 1)
        m_normalf = torch.matmul(rot_mat, normalf.transpose(2, 1)).transpose(2, 1)

        return normal, normalf, m_normal, m_normalf

    def generate_shade(self, il, m_normal):
        batch_size = il.shape[0]

        normal_uv, normal_uv_mask = self.get_uv_from_vt(m_normal[:,:-1])
        normal_uv = normal_uv.reshape(batch_size, 3, -1).transpose(2,1)

        shade = shading(il, normal_uv)
        shade = shade.reshape(batch_size, *self.texture_size, -1).permute(0,3,1,2)
        shade = shade * normal_uv_mask.unsqueeze(1)
        return shade

    def get_vt_from_image(self, image, vertex2d, m_normalf, m_normal):
        batch_size = image.shape[0]

        m_normal_z = m_normal[:,:-1,2]
        visible_vt = (m_normal_z > 0)

        vertex2d_u = image.shape[-2] - vertex2d[:,:-1,1] - 1
        vertex2d_v = vertex2d[:,:-1,0]    #- 2

        texture_vt, mapped = self.bilinear_sampler(image, vertex2d_u, vertex2d_v)
        mask = visible_vt*mapped
        texture_vt[~mask.bool()] = texture_vt.min()
        return texture_vt, mask

    def get_uv_from_vt(self, texture_vt, texture_vt_mask=None):
        # point of symmetricity is 127
        batch_size = texture_vt.shape[0]

        if texture_vt_mask is None:
            texture_vt_mask = torch.ones_like(texture_vt)[:,:,0].bool()

        tex_min = texture_vt.min()
        tex_max = texture_vt.max()
        texture_vt = (texture_vt - tex_min) / (tex_max - tex_min + 1e-8)

        # commented code for vt2pixel based mapping
        # texture_uv = torch.zeros([texture_vt.shape[0], *self.texture_size, texture_vt.shape[-1]], device=texture_vt.device)
        # texture_uv_mask = torch.zeros([texture_vt.shape[0], *self.texture_size], device=texture_vt.device, dtype=torch.bool)
        # u_round = torch.round(self.vt2pixel_u[:-1]).clamp(0, self.texture_size[0]-1).long()
        # v_round = torch.round(self.vt2pixel_v[:-1]).clamp(0, self.texture_size[1]-1).long()

        # texture_uv[:,u_round,v_round] = texture_vt
        # texture_uv = texture_uv * (tex_max - tex_min) + tex_min
        # texture_uv = texture_uv.permute(0,3,1,2)
        # texture_uv_mask[:,u_round,v_round] = texture_vt_mask

        uv_vt_map = self.tri[:,self.tri_2d]
        mapped = self.tri_2d_mask.view(-1)

        # assign random valid triangle to unmapped pixels for computation
        # uv_vt_map[:,~mapped] = 0

        # get rgb values at each mapped triangle vertex and find their avg
        vt_rgb1 = texture_vt[:,uv_vt_map[0]]
        valid1 = texture_vt_mask[:,uv_vt_map[0]]
        vt_rgb2 = texture_vt[:,uv_vt_map[1]]
        valid2 = texture_vt_mask[:,uv_vt_map[1]]
        vt_rgb3 = texture_vt[:,uv_vt_map[2]]
        valid3 = texture_vt_mask[:,uv_vt_map[2]]
        count = valid1.float() + valid2.float() + valid3.float()
        texture_uv = (vt_rgb1 + vt_rgb2 + vt_rgb3) / (count.unsqueeze(-1) + 1e-8)
        invalid = (count == 0)

        texture_uv = texture_uv * mapped.unsqueeze(-1)
        texture_uv = texture_uv.permute(0,2,1).reshape(batch_size,-1,*self.texture_size)
        texture_uv = texture_uv * (tex_max - tex_min) + tex_min

        # get the uv mask
        texture_uv_mask = (mapped * ~invalid).reshape(batch_size, *self.texture_size)

        return texture_uv, texture_uv_mask

    def get_vt_from_uv(self, texture, texture_mask=None):
        return self.bilinear_sampler(texture, self.vt2pixel_u[:-1], self.vt2pixel_v[:-1], texture_mask)

    def get_image_from_vt(self, texture_vt, texture_vt_mask, vertex2d, m_normalf, m_normal, viz=None):
        batch_size = texture_vt.shape[0]
        nc = texture_vt.shape[-1]

        tex_min = texture_vt.min()
        tex_max = texture_vt.max()
        texture_vt = (texture_vt - tex_min) / (tex_max - tex_min)

        # a triangle is visible is it's normal's z is +ve
        # another way would be if any of its vertex is visible, but we are not adapting that approach right now
        m_normalf_z = m_normalf[:,:,2]
        visible_tri = (m_normalf_z > 0)

        vertex2d_u = self.output_size - vertex2d[:,:,1] - 1
        vertex2d_v = vertex2d[:,:,0]
        vertex2d_z = vertex2d[:,:,2]
        img_uvz = torch.stack((vertex2d_u, vertex2d_v, vertex2d_z), dim=1)
        img_uvz[:,:,-1] = 0  # really not needed

        # Get triangle-map, barycoords and visiblity for each pixel
        tri_map, depth, barycoords_cpp, trimap_mask = zbuffertri_batch.forward(img_uvz, self.tri.float(), visible_tri, self.output_size)
        depth[~trimap_mask.bool()] = 0

        # correction for faulty zbuffer mappings
        depth_med = median(depth, kernel_size=5)[0].squeeze() * trimap_mask
        depth_diff = (depth_med - depth > 5000)
        trimap_med = median(tri_map * trimap_mask, kernel_size=5)[0].squeeze(1)
        tri_map[depth_diff] = trimap_med[depth_diff]

        verts_map = self.tri.transpose(1,0)[tri_map.long()]
        verts_map[~trimap_mask.bool()] = 0
        verts_map_index = verts_map.reshape(batch_size, -1).unsqueeze(-1).expand(-1,-1,3)
        verts_map_uvz = torch.gather(img_uvz[:,:,:-1].transpose(2,1), dim=1, index=verts_map_index).reshape(*verts_map.shape, 3)

        # barycoords with gradients
        barycoords = self.barycoords(verts_map_uvz)
        barycoords[depth_diff] = barycoords_cpp[depth_diff]

        # Valid vertices are vertices with true input mask label
        verts_valid = torch.gather(texture_vt_mask, dim=1, index=verts_map.view(batch_size, -1)).view(verts_map.shape)      # mapped vertices that are visible in texture_vt too
        barycoords_valid = barycoords * verts_valid                                         # barycoords for invalid vertices is made 0

        # Image RGB computation from Vertices
        verts_map_rgb = torch.gather(texture_vt, dim=1, index=verts_map_index[...,:nc]).reshape(*verts_map.shape, nc) # batch_size * 224 * 224 * 3 vertices * rgb_per_vertex
        pixel_rgb_sum = (verts_map_rgb * barycoords_valid.unsqueeze(-1)).sum(-2)
        pixel_rgb = pixel_rgb_sum / (barycoords_valid.sum(-1, keepdim=True) + 1e-8)
        pixel_rgb = pixel_rgb.permute(0,3,1,2)
        pixel_rgb = pixel_rgb * (tex_max - tex_min) + tex_min
        pixel_rgb = pixel_rgb * trimap_mask.unsqueeze(1)

        # Mask
        pixel_valid = verts_valid.sum(-1)
        pixel_mask = (pixel_valid * trimap_mask).bool()

        return pixel_rgb, pixel_mask, depth

    def get_image_mapping_params_from_vt(self, vertex2d, m_normalf):
        batch_size = vertex2d.shape[0]

        # a triangle is visible is it's normal's z is +ve
        # another way would be if any of its vertex is visible, but we are not adapting that approach right now
        m_normalf_z = m_normalf[:,:,2]
        visible_tri = (m_normalf_z > 0)

        vertex2d_u = self.output_size - vertex2d[:,:,1] - 1
        vertex2d_v = vertex2d[:,:,0]
        vertex2d_z = vertex2d[:,:,2]
        img_uvz = torch.stack((vertex2d_u, vertex2d_v, vertex2d_z), dim=1)
        img_uvz[:,:,-1] = 0  # really not needed

        # Get triangle-map, barycoords and visiblity for each pixel
        tri_map, depth, barycoords_cpp, trimap_mask = zbuffertri_batch.forward(img_uvz, self.tri.float(), visible_tri, self.output_size)
        depth[~trimap_mask.bool()] = 0

        # correction for faulty zbuffer mappings
        depth_med = median(depth, kernel_size=5)[0].squeeze(1) * trimap_mask
        depth_diff = (depth_med - depth > 5000)
        trimap_med = median(tri_map * trimap_mask, kernel_size=5)[0].squeeze(1)
        tri_map[depth_diff] = trimap_med[depth_diff]

        verts_map = self.tri.transpose(1,0)[tri_map.long()]
        verts_map[~trimap_mask.bool()] = 0
        verts_map_index = verts_map.reshape(batch_size, -1).unsqueeze(-1).expand(-1,-1,3)

        # barycoords with gradients
        verts_map_uvz = torch.gather(img_uvz[:,:,:-1].transpose(2,1), dim=1, index=verts_map_index).reshape(*verts_map.shape, 3)
        barycoords = self.barycoords(verts_map_uvz)
        barycoords[depth_diff] = barycoords_cpp[depth_diff]

        return depth, verts_map, barycoords, trimap_mask

    def get_image_from_mapping_params(self, texture_vt, texture_vt_mask, verts_map, barycoords, trimap_mask):
        batch_size = texture_vt.shape[0]
        nc = texture_vt.shape[-1]

        tex_min = texture_vt.min()
        tex_max = texture_vt.max()
        texture_vt = (texture_vt - tex_min) / (tex_max - tex_min + 1e-8)

        # Valid vertices are vertices with true input mask label
        verts_valid = torch.gather(texture_vt_mask, dim=1, index=verts_map.view(batch_size, -1)).view(verts_map.shape)      # mapped vertices that are visible in texture_vt too
        barycoords_valid = barycoords * verts_valid

        verts_map_index = verts_map.reshape(batch_size, -1).unsqueeze(-1).expand(-1,-1,3)
        verts_map_rgb = torch.gather(texture_vt, dim=1, index=verts_map_index[...,:nc]).reshape(*verts_map.shape, nc) # batch_size * 224 * 224 * 3 vertices * rgb_per_vertex
        pixel_rgb_sum = (verts_map_rgb * barycoords_valid.unsqueeze(-1)).sum(-2)
        pixel_rgb = pixel_rgb_sum / (barycoords_valid.sum(-1, keepdim=True) + 1e-8)
        pixel_rgb = pixel_rgb.permute(0,3,1,2)
        pixel_rgb = pixel_rgb * (tex_max - tex_min) + tex_min
        pixel_rgb = pixel_rgb * trimap_mask.unsqueeze(1)

        # Mask
        pixel_valid = verts_valid.sum(-1)
        pixel_mask = (pixel_valid * trimap_mask).bool()

        return pixel_rgb, pixel_mask


    def compute_landmarks(self, vertex2d):
        landmarks_xy = vertex2d[:,self.kpts,:2]

        landmarks_u = self.output_size - landmarks_xy[:,:,1] - 1
        landmarks_v = landmarks_xy[:,:,0]

        return landmarks_u, landmarks_v

