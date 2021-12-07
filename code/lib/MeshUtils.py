import torch
import torch.nn as nn
import numpy as np
import BboxTools as bbt

from pytorch3d.renderer.mesh.rasterizer import Fragments
import pytorch3d.renderer.mesh.utils as utils
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    camera_position_from_spherical_angles, HardPhongShader, PointLights,
)
try:
    from pytorch3d.structures import Meshes, Textures
    use_textures = True
except:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.renderer import TexturesVertex as Textures

    use_textures = False


def load_off(off_file_name, to_torch=False):
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

    if not to_torch:
        return array_, array_int.reshape((-1, 4))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(array_int.reshape((-1, 4))[:, 1::])


def save_off(off_file_name, vertices, faces):
    out_string = 'OFF\n'
    out_string += '%d %d 0\n' % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += '%.16f %.16f %.16f\n' % (v[0], v[1], v[2])
    for f in faces:
        out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
    with open(off_file_name, 'w') as fl:
        fl.write(out_string)
    return


def rotation_theta(theta, device_=None):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    if type(theta) == float:
        if device_ is None:
            device_ = 'cpu'
        theta = torch.ones((1, 1, 1)).to(device_) * theta
    else:
        if device_ is None:
            device_ = theta.device
        theta = theta.view(-1, 1, 1)

    mul_ = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]]).view(1, 2, 9).to(device_)
    bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

    # [n, 1, 2]
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

    # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
    trans = torch.matmul(cos_sin, mul_) + bia_
    trans = trans.view(-1, 3, 3)

    return trans


def rasterize(R, T, meshes, rasterizer, blur_radius=0):
    # It will automatically update the camera settings -> R, T in rasterizer.camera
    fragments = rasterizer(meshes, R=R, T=T)

    # Copy from pytorch3D source code, try if this is necessary to do gradient decent
    if blur_radius > 0.0:
        clipped_bary_coords = utils._clip_barycentric_coordinates(
            fragments.bary_coords
        )
        clipped_zbuf = utils._interpolate_zbuf(
            fragments.pix_to_face, clipped_bary_coords, meshes
        )
        fragments = Fragments(
            bary_coords=clipped_bary_coords,
            zbuf=clipped_zbuf,
            dists=fragments.dists,
            pix_to_face=fragments.pix_to_face,
        )
    return fragments


def campos_to_R_T(campos, theta, device='cpu', at=((0, 0, 0),), up=((0, 1, 0), )):
    R = look_at_rotation(campos, at=at, device=device, up=up)  # (n, 3, 3)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]  # (1, 3)
    return R, T


# For meshes in PASCAL3D+
def pre_process_mesh_pascal(verts):
    if isinstance(verts, torch.Tensor):
        verts = torch.cat((verts[:, 0:1], verts[:, 2:3], -verts[:, 1:2]), dim=1)
    else:
        verts = np.concatenate((verts[:, 0:1], verts[:, 2:3], -verts[:, 1:2]), axis=1)

    return verts


# Calculate interpolated maps -> [n, c, h, w]
# face_memory.shape: [n_face, 3, c]
def forward_interpolate(R, T, meshes, face_memory, rasterizer, blur_radius=0, mode='bilinear'):
    fragments = rasterize(R, T, meshes, rasterizer, blur_radius=blur_radius)

    # [n, h, w, 1, d]
    if mode == 'nearest':
        out_map = utils.interpolate_face_attributes(fragments.pix_to_face, set_bary_coords_to_nearest(fragments.bary_coords), face_memory)
    else:
        out_map = utils.interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_memory)

    out_map = out_map.squeeze(dim=3)
    out_map = out_map.transpose(3, 2).transpose(2, 1)
    return out_map, fragments


def set_bary_coords_to_nearest(bary_coords_):
    ori_shape = bary_coords_.shape
    exr = bary_coords_ * (bary_coords_ < 0)
    bary_coords_ = bary_coords_.view(-1, bary_coords_.shape[-1])
    arg_max_idx = bary_coords_.argmax(1)
    return torch.zeros_like(bary_coords_).scatter(1, arg_max_idx.unsqueeze(1), 1.0).view(*ori_shape) + exr


def vertex_memory_to_face_memory(memory_bank, faces):
    return memory_bank[faces.type(torch.long)]


def center_crop_fun(out_shape, max_shape):
    box = bbt.box_by_shape(out_shape, (max_shape[0] // 2, max_shape[1] // 2), image_boundary=max_shape)
    return lambda x: box.apply(x)


class MeshInterpolateModule(nn.Module):
    def __init__(self, vertices, faces, memory_bank, rasterizer, post_process=None, off_set_mesh=False):
        super(MeshInterpolateModule, self).__init__()

        # Convert memory feature of vertices to face
        self.face_memory = None
        self.update_memory(memory_bank=memory_bank, faces=faces)
        
        # Support multiple mesh at same time
        if type(vertices) == list:
            self.n_mesh = len(vertices)
            # Preprocess convert mesh in PASCAL3d+ standard to Pytorch3D
            verts = [pre_process_mesh_pascal(t) for t in vertices]

            # Create Pytorch3D mesh
            self.meshes = Meshes(verts=verts, faces=faces, textures=None)

        else:
            self.n_mesh = 1
            # Preprocess convert mesh in PASCAL3d+ standard to Pytorch3D
            verts = pre_process_mesh_pascal(vertices)

            # Create Pytorch3D mesh
            self.meshes = Meshes(verts=[verts], faces=[faces], textures=None)

        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh

    def update_memory(self, memory_bank, faces=None):
        with torch.no_grad():
            if type(memory_bank) == list:
                if faces is None:
                    faces = self.meshes.faces_list()
                # Convert memory feature of vertices to face
                self.face_memory = torch.cat([vertex_memory_to_face_memory(m, f).to(m.device) for m, f in zip(memory_bank, faces)], dim=0)
            else:
                if faces is None:
                    faces = self.meshes.faces_list()[0]
                # Convert memory feature of vertices to face
                self.face_memory = vertex_memory_to_face_memory(memory_bank, faces).to(memory_bank.device)

    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super(MeshInterpolateModule, self).to(device)
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        self.face_memory = self.face_memory.to(device)
        self.meshes = self.meshes.to(device)
        return self

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def forward(self, campos, theta, blur_radius=0, deform_verts=None, return_fragments=False, mode='bilinear', **kwargs):
        R, T = campos_to_R_T(campos, theta, device=campos.device, **kwargs)

        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)
        else:
            meshes = self.meshes

        n_cam = campos.shape[0]
        if n_cam > 1 and self.n_mesh > 1:
            get, fragments = forward_interpolate(R, T, meshes, self.face_memory, rasterizer=self.rasterizer, blur_radius=blur_radius, mode=mode)
        elif n_cam > 1 and self.n_mesh == 1:
            # get, fragments = forward_interpolate(R, T, meshes.extend(campos.shape[0]), self.face_memory.repeat(campos.shape[0], 1, 1).view(-1, *self.face_memory.shape[1:]), rasterizer=self.rasterizer, blur_radius=blur_radius, mode=mode)
            get, fragments = forward_interpolate(R, T, meshes.extend(campos.shape[0]), self.face_memory.repeat(campos.shape[0], 1, 1), rasterizer=self.rasterizer, blur_radius=blur_radius, mode=mode)
        else:
            get, fragments = forward_interpolate(R, T, meshes, self.face_memory, rasterizer=self.rasterizer, blur_radius=blur_radius, mode=mode)

        if self.post_process is not None:
            get = self.post_process(get)
        if return_fragments:
            return get, fragments
        return get


def camera_position_to_spherical_angle(camera_pose):
    distance_o = torch.sum(camera_pose ** 2, dim=1) ** .5
    azimuth_o = torch.atan(camera_pose[:, 0] / camera_pose[:, 2]) % np.pi + np.pi * (camera_pose[:, 0] <= 0).type(camera_pose.dtype).to(camera_pose.device)
    elevation_o = torch.asin(camera_pose[:, 1] / distance_o)
    return distance_o, elevation_o, azimuth_o


def angel_gradient_modifier(base, grad_=None, alpha=(1.0, 1.0), center_=None):
    # alpha[0]: normal
    # alpha[1]: tangential
    if grad_ is None:
        grad_ = base.grad
        apply_to = True
    else:
        apply_to = False

    if center_ is not None:
        base_ = base.clone() - center_
    else:
        base_ = base

    with torch.no_grad():
        direction = base_ / torch.sum(base_ ** 2, dim=1) ** .5
        normal_vector = torch.sum(direction * grad_, dim=1, keepdim=True) * direction

        tangential_vector = grad_ - normal_vector
        out = normal_vector * alpha[0] + tangential_vector * alpha[1]

    if apply_to:
        base.grad = out

    return out


def decompose_pose(pose, sorts=('distance', 'elevation', 'azimuth', 'theta')):
    return pose[:, sorts.index('distance')], pose[:, sorts.index('elevation')], \
           pose[:, sorts.index('azimuth')], pose[:, sorts.index('theta')]


def normalize(x, dim=0):
    return x / torch.sum(x ** 2, dim=dim, keepdim=True)[0] ** .5


def standard_loss_func_with_clutter(obj_s: torch.Tensor, clu_s: torch.Tensor):
    clu_s = torch.max(clu_s, dim=1)[0]
    return torch.ones(1, device=obj_s.device) - (torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s))


class MeshTrainingForwardModule(nn.Module):
    def __init__(self, path_mesh_file, render_size, feature_bank, n_points, clutter_merge_func=lambda x: normalize(torch.mean(x, dim=0), dim=0).unsqueeze(0), gradient_to_bank=False, train_mesh=False):
        super(MeshTrainingForwardModule, self).__init__()
        render_image_size = max(render_size)
        cameras = OpenGLPerspectiveCameras(fov=12.0)
        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        xvert, xface = load_off(path_mesh_file, to_torch=True)

        self.inter_module = MeshInterpolateModule(xvert, xface, feature_bank.memory[0:n_points], rasterizer,
                                             post_process=center_crop_fun(render_size, (max(render_size),) * 2), off_set_mesh=train_mesh)
        self.feature_bank = feature_bank
        self.n_points = n_points
        self.grad_to_bank = gradient_to_bank
        self.clutter_merge_func = clutter_merge_func

        if train_mesh:
            self.deform_verts = torch.nn.parameter.Parameter(torch.Tensor(*self.inter_module.meshes.verts_packed().shape))
            with torch.no_grad():
                self.deform_verts.fill_(0)
        else:
            self.register_parameter('deform_verts', None)

    def cuda(self, device=None):
        super().cuda(device)
        self.device = torch.device("cuda")
        self.inter_module.cuda(device)
        return self.inter_module.cuda(device)

    def to(self, device):
        super().to(device)
        self.inter_module.to(device)
        self.device = device
        return self.inter_module.to(device)

    def get_final_verts(self):
        if self.deform_verts is None:
            return None
        return self.inter_module.meshes.offset_verts(self.deform_verts).get_mesh_verts_faces(0)

    def save_mesh(self, mesh_file_path):
        final_verts, final_faces = self.get_final_verts()
        save_off(mesh_file_path, final_verts.detach().cpu().numpy(), final_faces.detach().cpu().numpy())

    def forward(self, forward_feature, pose, ):
        with torch.set_grad_enabled(self.grad_to_bank):
            self.inter_module.update_memory(self.feature_bank.memory[0:self.n_points])
            clutter_features = self.clutter_merge_func(self.feature_bank.memory[self.n_points::])

        pose_ = decompose_pose(pose)
        C = camera_position_from_spherical_angles(*pose_[0:3], device=forward_feature.device)
        theta = pose_[3]
        projected_feature = self.inter_module(C, theta, deform_verts=self.deform_verts)

        # [n, w, h]
        sim_fg = torch.sum(projected_feature * forward_feature, dim=1)

        # [n, clutter_num, w, h]
        sim_bg = torch.nn.functional.conv2d(forward_feature, clutter_features.unsqueeze(2).unsqueeze(3))

        return sim_fg, sim_bg


if __name__ == '__main__':
    from PIL import Image
    import io
    import os
    import matplotlib.pyplot as plt

    cate = 'car'
    mesh_d = 'buildn'
    # occ_level = 'FGL1_BGL1'
    occ_level = ''

    def plot_fun(values, para_scans, colors, figsize=(10.5, 4)):
        plt.figure(num=None, figsize=figsize)
        ax = plt.axes()

        for v, p, c in zip(values, para_scans, colors):
            ax.plot(v, p, c)
        plt.axvline(x=0, c='black')
        return ax


    def get_one_image_from_plt(plot_functions, plot_args=tuple(), plot_kwargs=dict()):
        plt.cla()
        plt.clf()
        ax = plot_functions(*plot_args, **plot_kwargs)
        positions = ax.get_position()
        pos = [positions.y0, positions.y1, positions.x0, positions.x1]
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        img = np.array(im)
        h, w = img.shape[0:2]
        box = bbt.from_numpy([np.array([int(t[0] * h), int(t[1] * h), int(t[2] * w), int(t[3] * w)]) for t in [pos]][0])
        box = box.pad(1)
        box = box.shift((2, 1))
        img = box.apply(img)
        bbt.draw_bbox(img, bbt.full(img.shape).pad(-2), boundary=(0, 0, 0), boundary_width=11)
        # img = np.transpose(img, (1, 0, 2))
        return img

    if len(occ_level) == 0:
        # mesh_path = '../PASCAL3D/CAD_d4/car/%02d.off'
        mesh_path = '../PASCAL3D/CAD_' + mesh_d + '/' + cate + '/%02d.off'
        img_path = '../PASCAL3D/PASCAL3D_distcrop/images/' + cate + '/%s.JPEG'
        annos_path = '../PASCAL3D/PASCAL3D_distcrop/annotations/' + cate + '/%s.npz'
        record_file_path = './saved_features/car/resunetpre_3D512_points1saved_model_car_39_' + mesh_d + '.npz'
        # record_file_path = './saved_features/' + cate + '/resunetpre_3D512_points1saved_model_' + cate + '_799_' + mesh_d + '.npz'

        save_dir = './loss_landscape/unsupervised/' + cate + '_' + mesh_d + '/'

        names = os.listdir('../PASCAL3D/PASCAL3D_distcrop/images/' + cate)
    else:
        mesh_path = '../PASCAL3D/CAD_' + mesh_d + '/' + cate + '/%02d.off'
        img_path = '../PASCAL3D/PASCAL3D_OCC_distcrop/images/' + cate + occ_level + '/%s.JPEG'
        annos_path = '../PASCAL3D/PASCAL3D_distcrop/annotations/' + cate + '/%s.npz'
        record_file_path = './saved_features/' + cate + '_occ/' + occ_level + '_resunetpre_3D512_points1saved_model_' + cate + '_799_' + mesh_d + '.npz'

        save_dir = '../junks/aligns_final/' + cate + occ_level + '_' + mesh_d + '/'
        names = os.listdir('../PASCAL3D/PASCAL3D_distcrop/images/' + cate)

    names = [t.split('.')[0] for t in names]

    image_name = 'n03498781_613'
    device = 'cuda:0'

    down_smaple_rate = 8
    image_sizes = {'car': (256, 672), 'bus': (384, 896), 'motorbike': (512, 512), 'boat': (512, 1216),
                   'bicycle': (608, 608), 'aeroplane': (320, 1024), 'sofa': (352, 736), 'tvmonitor': (480, 480),
                   'chair': (544, 384), 'diningtable': (320, 800), 'bottle': (512, 736), 'train': (256, 608)}

    distance_render = {'car': 5, 'bus': 6, 'motorbike': 4.5, 'bottle': 5, 'boat': 8, 'bicycle': 5.2, 'aeroplane': 7,
                       'sofa': 5, 'tvmonitor': 5.5, 'chair': 4, 'diningtable': 7, 'train': 4.5}

    os.makedirs(save_dir, exist_ok=True)
    render_image_size = max(image_sizes[cate]) // down_smaple_rate
    cameras = OpenGLPerspectiveCameras(device=device, fov=12.0)
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    all_distance = []
    for image_name in names:
        print(image_name, end=' ')
        annos_file = np.load(annos_path % image_name)

        # xvert, xface = load_off('../PASCAL3D/CAD_d4/car/%02d.off' % annos_file['cad_index'], to_torch=True)
        if 'build' in mesh_d:
            xvert, xface = load_off('../PASCAL3D/PASCAL3D+_release1.1/CAD_' + mesh_d + '/' + cate + '/%02d.off' % 1, to_torch=True)
            subtype = 'mesh%02d' % 1
        else:
            xvert, xface = load_off('../PASCAL3D/PASCAL3D+_release1.1/CAD_' + mesh_d + '/' + cate + '/%02d.off' % annos_file['cad_index'], to_torch=True)
            subtype = 'mesh%02d' % annos_file['cad_index']
        record_file = np.load(record_file_path)

        feature_bank = torch.from_numpy(record_file['memory_%s' % subtype])
        if image_name not in record_file.keys():
            continue
        predicted_map = record_file[image_name]
        predicted_map = torch.from_numpy(predicted_map).to(device)

        inter_module = MeshInterpolateModule(xvert, xface, feature_bank, rasterizer, post_process=center_crop_fun(predicted_map.shape[1::], (render_image_size, ) * 2))
        inter_module.cuda()

        azimuth_shifts = np.linspace(-3.14, 3.14, 121)
        elevation_shifts = np.linspace(-3.14 / 2, 3.14 / 2, 61)
        theta_shifts = np.linspace(-3.14 / 2, 3.14 / 2, 61)
        distance_shifts = np.linspace(-2, 2, 41)

        get = []
        # for elevation_shift in elevation_shifts:
        for azimuth_shift in azimuth_shifts:
            this_azum = (annos_file['azimuth'] + azimuth_shift + 2 * np.pi) % (2 * np.pi)
            this_elev = annos_file['elevation']
            C = camera_position_from_spherical_angles(distance_render[cate], this_elev, this_azum, degrees=False, device=device)
            theta = torch.from_numpy(annos_file['theta']).type(torch.float32).view(1)
            projected_map = inter_module(C, theta).squeeze()
            sim_ = torch.sum(projected_map * predicted_map, dim=0)

            get.append(1 - (torch.mean(sim_)).item())
            # if np.abs(azimuth_shift) < 1e-5:
            #     print(C, (torch.mean(sim_)).item())
        azum_scan = np.array(get)

        get = []
        for elevation_shift in elevation_shifts:
            this_azum = annos_file['azimuth']
            this_elev = annos_file['elevation'] + elevation_shift
            C = camera_position_from_spherical_angles(distance_render[cate], this_elev, this_azum, degrees=False, device=device)
            theta = torch.from_numpy(annos_file['theta']).type(torch.float32).view(1)
            projected_map = inter_module(C, theta).squeeze()
            sim_ = torch.sum(projected_map * predicted_map, dim=0)
            get.append(1 - (torch.mean(sim_)).item())
            # if np.abs(elevation_shift) < 1e-5:
            #     print(C, (torch.mean(sim_)).item())
        elev_scan = np.array(get)

        get = []
        for theta_shift in theta_shifts:
            this_azum = annos_file['azimuth']
            this_elev = annos_file['elevation']
            C = camera_position_from_spherical_angles(distance_render[cate], this_elev, this_azum, degrees=False, device=device)
            theta = torch.from_numpy(np.array(annos_file['theta'] + theta_shift)).type(torch.float32).view(1)
            projected_map = inter_module(C, theta).squeeze()
            sim_ = torch.sum(projected_map * predicted_map, dim=0)
            get.append(1 - (torch.mean(sim_)).item())
            # if np.abs(theta_shift) < 1e-5:
            #     print(C, (torch.mean(sim_)).item())
        theta_scan = np.array(get)

        get = []
        for distance_shift in distance_shifts:
            this_azum = annos_file['azimuth']
            this_elev = annos_file['elevation']
            C = camera_position_from_spherical_angles(distance_render[cate] + distance_shift, this_elev, this_azum, degrees=False,
                                                      device=device)
            theta = torch.from_numpy(np.array(annos_file['theta'])).type(torch.float32).view(1)
            projected_map = inter_module(C, theta).squeeze()
            sim_ = torch.sum(projected_map * predicted_map, dim=0)
            get.append(1 - (torch.mean(sim_)).item())
            # if np.abs(theta_shift) < 1e-5:
            #     print(C, (torch.mean(sim_)).item())
        distance_scan = np.array(get)
        this_dist = distance_shifts[np.argmax(distance_scan)]
        all_distance.append(this_dist)
        print(this_dist)

        # print(np.max(azum_scan), np.max(elev_scan), np.max(theta_scan))
        # print(azum_scan[75], elev_scan[30], theta_scan[30])
        # print(np.argmax(azum_scan), np.argmax(elev_scan), np.argmax(theta_scan))

        values_ = [azimuth_shifts, elevation_shifts, theta_shifts, distance_shifts]
        scans_ = [azum_scan, elev_scan, theta_scan, distance_scan]
        # colors_ = ['b', 'r', 'g', 'y']
        colors_ = ['b', 'r', 'g']
        img_ = get_one_image_from_plt(plot_functions=plot_fun, plot_args=(values_, scans_, colors_))
        # Image.fromarray(img_).show()

        Image.fromarray(img_).save(save_dir + image_name + '.png')

        # plt.plot(azimuth_shifts, azum_scan, 'b')
        # plt.plot(elevation_shifts, elev_scan, 'r')
        # plt.plot(theta_shifts, theta_scan, 'g')

        # plt.savefig('../junks/scan_align_new/' + image_name + '.png')
        # plt.show()
    print(np.mean(all_distance))
    np.save(save_dir + 'all_distance.npy', all_distance)
