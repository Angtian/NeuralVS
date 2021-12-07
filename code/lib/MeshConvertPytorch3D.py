import sys
sys.path.append('./lib')


import numpy as np
import torch
import os
from ProcessCameraParameters import get_anno, Projector3Dto2D
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams
)
from pytorch3d.structures import Meshes
from MeshUtils import camera_position_from_spherical_angles, campos_to_R_T, pre_process_mesh_pascal
from PointProjecter import PointProjectPytorch
import BboxTools as bbt

# mesh_path = '../PASCAL3D/PASCAL3D+_release1.1/CAD_d4/car/'
mesh_path = './CAD_d4/car/'


def convert_vert_to_face(verts, faces):
    out = np.zeros((faces.shape[0], verts.shape[1]), dtype=verts.dtype)
    for i, face in enumerate(faces):
        out[i] = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3
    return out


def box_include_2d(self_box, other):
    return torch.logical_and(torch.logical_and(self_box.bbox[0][0] <= other[:, 0], other[:, 0] < self_box.bbox[0][1]),
                          torch.logical_and(self_box.bbox[1][0] <= other[:, 1], other[:, 1] < self_box.bbox[1][1]))


def load_off(off_file_name):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2:2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

    all_strings = ''.join(file_list[2 + n_points:])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

    array_ = array_.reshape((-1, 3))
    return array_, array_int.reshape((-1, 4))[:, 1::]


def if_visiable_pytorch(verts, faces, anno, rasterizer, device='cpu', binary=True):
    mesh_ = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
    )
    azimuth_, distance, elevation_ = get_anno(anno, 'azimuth', 'distance', 'elevation')
    azimuth = (azimuth_ * 180 / np.pi + 180) % 360
    elevation = elevation_ * 180 / np.pi

    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

    silhouete_ = rasterizer(meshes_world=mesh_, R=R, T=T)
    all_pixels = silhouete_.pix_to_face.squeeze().view(-1)
    all_pixels = torch.unique(all_pixels[all_pixels >= 0])
    all_visiable_verts = torch.unique(faces[all_pixels.type(torch.long)].view(-1)).type(torch.long)

    if not binary:
        return all_visiable_verts
    mask_ = torch.zeros(verts.shape[0], dtype=torch.bool).to(device)
    mask_.scatter_(0, all_visiable_verts, True)
    return mask_


def if_visiable_pytorch_given_pix_to_face(pix_to_face, verts, faces, binary=True):
    all_faces = pix_to_face.view(-1)
    all_pixels = torch.unique(all_faces[all_faces >= 0])
    all_visiable_verts = torch.unique(faces[all_pixels.type(torch.long)].view(-1)).type(torch.long)

    if not binary:
        return all_visiable_verts
    mask_ = torch.zeros(verts.shape[0], dtype=torch.bool).to(verts.device)
    mask_.scatter_(0, all_visiable_verts, True)
    return mask_


class MeshLoader(object):
    def __init__(self, path=mesh_path, use_torch=False, device='cpu'):
        file_list = os.listdir(path)

        l = len(file_list)
        file_list = ['%02d.off' % (i + 1) for i in range(l)]

        self.mesh_points_3d = []
        self.mesh_triangles = []

        for fname in file_list:
            points_3d, triangles = load_off(os.path.join(path, fname))
            if use_torch:
                self.mesh_points_3d.append(torch.from_numpy(points_3d).to(device))
                self.mesh_triangles.append(torch.from_numpy(triangles).to(device))
            else:
                self.mesh_points_3d.append(points_3d)
                self.mesh_triangles.append(triangles)

    def __getitem__(self, item):
        return self.mesh_points_3d[item], self.mesh_triangles[item]

    def __len__(self):
        return len(self.mesh_points_3d)


class MeshConverter3D(object):
    def __init__(self, path=mesh_path, device='cpu', rasterizer=None):
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        cameras = OpenGLPerspectiveCameras(device=device, fov=20)

        self.loader = MeshLoader(path=path, use_torch=True, device=device)
        self.viewpoint_para = 'azimuth, elevation, distance, focal, theta, principal, viewport'.split(', ')
        if rasterizer is None:
            self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        else:
            self.rasterizer = rasterizer
        self.device = device

        self.rasterizer.to(device)

    def get_one(self, azimuth, elevation, theta, distance, principal, focal=1, viewport=3000, off_idx=1):
        # principal -> image_size[0] // 2, image_size[1] // 2
        annos = {'azimuth': azimuth, 'elevation': elevation, 'theta': theta, 'distance': distance,
                 'principal': principal, 'focal': focal, 'viewport': viewport}
        points_3d, triangles = self.loader[off_idx - 1]

        points_2d = PointProjectPytorch(init_val=annos, training_parameters=[], do_registration=True)(points_3d)
        points_2d = torch.flip(points_2d, dims=(1, ))

        pixels_2d = points_2d

        points_3d_ = torch.cat([points_3d[:, 0:1], points_3d[:, 2:3], -points_3d[:, 1:2]], dim=1).to(self.device)
        if_visible = if_visiable_pytorch(points_3d_, triangles, annos, self.rasterizer, device=self.device)

        return pixels_2d, if_visible


def verts_proj(verts, azimuth, elevation, theta, distance, principal, M=3000, device='cpu'):
    C = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=False, device=device)
    R, T = campos_to_R_T(C, theta, device=device)

    return verts_proj_matrix(verts, R, T, principal=principal, M=M)
    #
    # get = verts @ R + T.unsqueeze(1)
    # return principal - torch.cat([get[:, :, 1:2] / get[:, :, 2:3], get[:, :, 0:1] / get[:, :, 2:3]], dim=2) * M


def verts_proj_matrix(verts, R, T, principal, M=3000):
    if len(verts.shape) == 2:
        verts = verts.unsqueeze(0)

    if not isinstance(principal, torch.Tensor):
        principal = torch.Tensor([principal[0], principal[1]]).view(1, 1, 2).type(torch.float32).to(verts.device)

    get = verts @ R + T.unsqueeze(1)
    return principal - torch.cat([get[:, :, 1:2] / get[:, :, 2:3], get[:, :, 0:1] / get[:, :, 2:3]], dim=2) * M


def limit_to_img_size(verts, img_size, vis=None):
    if vis is None:
        verts = torch.max(torch.zeros_like(verts), verts)
        verts = torch.min(torch.Tensor([img_size]).to(verts.device) - 1, verts)
        return verts
    else:
        ill1_ = torch.any(torch.zeros_like(verts) > verts, dim=2)
        ill2_ = torch.any(torch.Tensor([img_size]).to(verts.device) - 1 < verts, dim=2)
        vis_out = torch.logical_and(vis, torch.logical_not(torch.logical_or(ill1_, ill2_)))

        verts = torch.max(torch.zeros_like(verts), verts)
        verts = torch.min(torch.Tensor([img_size]).to(verts.device) - 1, verts)
        return verts, vis_out


def limit_to_obj_box(verts, box, vis=None):
    if vis is not None:
        vis = torch.logical_and(vis, box_include_2d(box, verts))
    return limit_to_img_size(verts, box.boundary, vis=vis)


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b * e * f
    return out


def limit_to_obj_mask(verts, mask, vis=None):
    # mask: [n, h, w]
    # verts: [n, k, 2]
    verts, vis_out = limit_to_img_size(verts, mask.shape[1::], vis=vis)
    inds = verts[:, :, 0].type(torch.long) * mask.shape[2] + verts[:, :, 1].type(torch.long)
    selected = batched_index_select(mask.view(mask.shape[0], -1, 1), dim=1, inds=inds).squeeze(2)
    vis_out = torch.logical_and(vis_out, selected)
    return verts, vis_out


class MeshConverter(object):
    def __init__(self, path=mesh_path, device='cpu'):
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        cameras = OpenGLPerspectiveCameras(device=device, fov=20)

        self.loader = MeshLoader(path=path, use_torch=True, device=device)
        self.viewpoint_para = 'azimuth, elevation, distance, focal, theta, principal, viewport'.split(', ')
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.device = device

        self.rasterizer.to(device)

    def get_one(self, annos):
        off_idx = get_anno(annos, 'cad_index')

        points_3d, triangles = self.loader[off_idx - 1]

        points_2d = Projector3Dto2D(annos)(points_3d).astype(np.int32)
        points_2d = np.flip(points_2d, axis=1)

        box_ori = bbt.from_numpy(get_anno(annos, 'box_ori'))
        box_cropped = bbt.from_numpy(get_anno(annos, 'box_obj'))

        projection_foo = bbt.projection_function_by_boxes(box_ori, box_cropped)

        pixels_2d = projection_foo(points_2d)

        # handle the case that points are out of boundary of the image
        pixels_2d = np.max([np.zeros_like(pixels_2d), pixels_2d], axis=0)
        pixels_2d = np.min([np.ones_like(pixels_2d) * (np.array([box_cropped.boundary]) - 1), pixels_2d], axis=0)

        points_3d_ = torch.cat([points_3d[:, 0:1], points_3d[:, 2:3], points_3d[:, 1:2]], dim=1).to(self.device)
        if_visible = if_visiable_pytorch(points_3d_, triangles, annos, self.rasterizer, device=self.device)

        return pixels_2d, if_visible


if __name__ == '__main__':
    name_ = 'n02814533_31493.JPEG'
    anno_path = '../PASCAL3D/PASCAL3D/annotations/car/'
    converter = MeshConverter()
    pixels, visibile = converter.get_one(np.load(os.path.join(anno_path, name_.split('.')[0] + '.npz'), allow_pickle=True))








