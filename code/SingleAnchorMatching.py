import torch
import matplotlib.pyplot as plt
import BboxTools as bbt
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.linalg import logm
import argparse
import tqdm

from torch.utils.data import Dataset, DataLoader
from pytorch3d.renderer import PerspectiveCameras

from lib.MeshConvertPytorch3D import verts_proj_matrix, if_visiable_pytorch_given_pix_to_face, limit_to_img_size, limit_to_obj_mask
from lib.MeshUtils import load_off, MeshInterpolateModule, pre_process_mesh_pascal, center_crop_fun, RasterizationSettings, \
                      MeshRasterizer, camera_position_from_spherical_angles, campos_to_R_T, Textures, \
                      PointLights, MeshRenderer, HardPhongShader, Meshes
from lib.ProcessCameraParameters import get_transformation_matrix
from models.KeypointRepresentationNet import NetE2E



parser = argparse.ArgumentParser(description='Retrieve image using single anchor image.')
parser.add_argument('--net_type', default='resnet50_pre', type=str, help='')
parser.add_argument('--load_path', default='None', type=str, help='')
parser.add_argument('--mesh_path', default='./data/PASCAL3D+_release1.1', type=str, help='')
parser.add_argument('--data_path', default='./data/PASCAL3D_train_NeMo', type=str, help='')
parser.add_argument('--save_path', default='./exp/SingleAnchorMatching', type=str, help='')
parser.add_argument('--mesh_d', default='single', type=str, help='')
parser.add_argument('--cate', default='car', type=str, help='')
parser.add_argument('--n_retrieve', default=3, type=int, help='')
parser.add_argument('--do_plot', action='store_true', help='')

args = parser.parse_args()


cate = args.cate

net_type = args.net_type

# The 20 anchors are randomly selected from the training set. To reproduce the results, use this 20 anchors.
anchor_names = ['n02814533_11587', 'n02814533_11667', 'n02814533_11762', 'n02814533_26766', 'n03498781_1745',
                'n04285965_15719', 'n04285965_16270', 'n04285965_19641', 'n03770679_10140', 'n04166281_7116',
                'n02814533_18927', 'n02814533_2880', 'n02814533_11362', 'n02958343_71115', 'n03498781_1339',
                'n03498781_4086', 'n03498781_5306', 'n04166281_3331', 'n04166281_3593', 'n04166281_4796']

device = 'cuda'
mesh_d = args.mesh_d

if args.load_path == 'None':
    load_name = None
else:
    load_name = args.load_path


mesh_path = os.path.join(args.mesh_path, 'CAD_' + mesh_d, cate)
mesh_path_mask = os.path.join(args.mesh_path, 'CAD', cate)

anno_path = os.path.join(args.data_path, 'annotations', cate)
img_path = os.path.join(args.data_path, 'images', cate)

net_type_mapping = {'vggp4': 'vgg_pool4', 'resnet50_pre': 'resnet50_pre', 'resnetupsample': 'resnetupsample', 'resnetext': 'resnetext'}
net_dfeature_mapping = {'vggp4': 512, 'resnet50_pre': 1024, 'resnetupsample': 2048, 'resnetext': 128}

image_size_ori = {'car': (256, 672), }[cate]

num_noise = 0
max_group = 512

if net_type == 'resnetext':
    d_out_layer = 128
else:
    d_out_layer = -1

save_dir = args.save_path

standard_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

azimuth_samples = np.linspace(-np.pi / 3, np.pi / 3, 13)

print('Start retreve, save to: ', save_dir)


class Pascal3D(Dataset):
    useful_keys = {'elevation': np.float32, 'azimuth': np.float32, 'theta': np.float32, 'cad_index': np.int32}

    def __init__(self, img_path=img_path, anno_path=anno_path, image_list=None, enable_cache=True, transform=None):
        if image_list is None:
            all_imgs = os.listdir(img_path)
        elif isinstance(image_list, str):
            all_imgs = [t.strip() + '.JPEG' for t in open(image_list).readlines()]
        else:
            all_imgs = [t.strip().split('.')[0] + '.JPEG' for t in image_list]

        self.all_imgs = [t.split('.')[0] for t in all_imgs]
        self.img_path = img_path
        self.anno_path = anno_path

        self.cache_anno = dict()
        self.cache_img = dict()
        self.transform = transform
        self.enable_cache = enable_cache

        self.enabled_mask = False
        self.mask = None
        self.flip_thr = -1

    def __getitem__(self, item):
        if self.enabled_mask:
            img_name = self.all_imgs[self.mask[item]]
        else:
            img_name = self.all_imgs[item]
        if not self.enable_cache:
            img_ = Image.open(os.path.join(self.img_path, img_name + '.JPEG')).convert('RGB')
            anno = np.load(os.path.join(self.anno_path, img_name + '.npz'), allow_pickle=True)
            anno = {k_: anno[k_] for k_ in self.useful_keys}
        elif img_name not in self.cache_anno.keys():
            img_ = Image.open(os.path.join(self.img_path, img_name + '.JPEG')).convert('RGB')
            anno = np.load(os.path.join(self.anno_path, img_name + '.npz'), allow_pickle=True)
            anno = {k_: anno[k_] for k_ in self.useful_keys}

            self.cache_img[img_name] = img_
            self.cache_anno[img_name] = anno
        else:
            img_ = self.cache_img[img_name]
            anno = self.cache_anno[img_name]

        if item < self.flip_thr:
            anno = anno.copy()
            img_ = img_.copy()
            img_ = img_.transpose(Image.FLIP_LEFT_RIGHT)
            anno['azimuth'] = np.pi * 2 - anno['azimuth']

        anno['name'] = img_name

        return self.transform(img_), anno

    def __len__(self):
        if self.enabled_mask:
            return len(self.mask)
        return len(self.all_imgs)

    def get_list_images(self, img_name_list):
        imgs, annos = [], []
        for img_name in img_name_list:
            img, anno = self.__getitem__(self.all_imgs.index(img_name))
            imgs.append(img)
            annos.append(anno)
        return torch.stack(imgs), {k_: torch.stack([torch.from_numpy(anno_[k_]) for anno_ in annos]) for k_ in annos[0].keys() if k_ != 'name'}


def do_render(inter_module, cam_pos, theta, verts_mask=None):
    R, T = campos_to_R_T(cam_pos, theta, device=device)
    projected_map, fragment = inter_module.forward(cam_pos, theta, return_fragments=True)
    out_vis = []

    # (sum(V_n), 3)
    verts = inter_module.meshes.verts_packed()
    faces = inter_module.meshes.faces_packed()

    for i in range(cam_pos.shape[0]):
        isvisible_ = if_visiable_pytorch_given_pix_to_face(fragment.pix_to_face[i] - i * faces.shape[0], verts=verts, faces=faces).to(device)
        out_vis.append(isvisible_.unsqueeze(0))
    all_vis = torch.cat(out_vis, dim=0)

    # [n_azum * n_elev, n_vert, 2]
    all_vert = verts_proj_matrix(verts.unsqueeze(0), R, T, principal=(image_size_ori[0] // 2, image_size_ori[1] // 2))
    if verts_mask is None:
        all_vert = limit_to_img_size(all_vert, image_size_ori)
    else:
        all_vert, all_vis = limit_to_obj_mask(all_vert, mask=verts_mask, vis=all_vis)

    return projected_map, all_vert, all_vis


def single_sample(inter_module_, verts_label, vis_label, imgs, net_):
    extracted_feature = net_(X=imgs, keypoint_positions=verts_label, mode=-1)

    inter_module_.update_memory((vis_label.unsqueeze(2) * extracted_feature).sum(0))


def retrieve_similarity(target_maps, this_loader, network, return_annos=True):
    out_sims = []
    out_annos = []

    network.eval()
    with torch.no_grad():
        for img, anno in this_loader:
            if target_maps.device != torch.device('cpu'):
                img = img.cuda()
            feature_map = network(X=img, mode=0)

            # [n_img, n_pos]
            similarity = torch.sum(feature_map.unsqueeze(1) * target_maps.unsqueeze(0), dim=(2, 3, 4))
            out_sims.append(similarity)
            out_annos.append(anno)
    sims = torch.cat(out_sims, dim=0)
    annos = {k_: torch.cat([anno_[k_] for anno_ in out_annos]) if k_ != 'name' else sum([anno_[k_] for anno_ in out_annos], []) for k_ in out_annos[0].keys()}

    if return_annos:
        return sims, annos
    else:
        return sims


def rotation_theta(theta):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def cal_err(gt, pred):
    # return radius
    return ((logm(np.dot(np.transpose(pred), gt)) ** 2).sum()) ** 0.5 / (2. ** 0.5)


def cal_rotation_matrix(theta, elev, azum, dis):
    if dis <= 1e-10:
        dis = 0.5

    return rotation_theta(theta) @ get_transformation_matrix(azum, elev, dis)[0:3, 0:3]


def get_mask(theta, campos_, crop_size, render_image_size, this_mesh):
    C = campos_
    R, T = campos_to_R_T(C, theta, device=device)
    image = phong_renderer(meshes_world=this_mesh.clone(), R=R, T=T)
    image = image[:, ..., :3]
    box_ = bbt.box_by_shape(crop_size, (render_image_size // 2,) * 2)
    bbox = box_.bbox
    image = image[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :]
    image = (image / image.max())
    return image.sum(3) < 3 - 1e-10


all_mask_meshes = []
for k in range(10):
    verts, faces = load_off(os.path.join(mesh_path_mask, '%02d.off' % (k + 1)), to_torch=True)
    verts = pre_process_mesh_pascal(verts)

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    # textures = Textures(verts_rgb=verts_rgb.to(device))
    textures = Textures(verts_features=verts_rgb.to(device))
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures)
    meshes = meshes.to(device)
    all_mask_meshes.append(meshes)


cameras = PerspectiveCameras(focal_length=1.0 * 3000, principal_point=((max(image_size_ori) / 2, max(image_size_ori) / 2), ), image_size=((max(image_size_ori), ) * 2, ), device=device)

raster_settings = RasterizationSettings(
    image_size=(max(image_size_ori), ) * 2,
    blur_radius=0,
    faces_per_pixel=1,
    perspective_correct=False
    # bin_size=0
)
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, lights=lights, cameras=cameras)
)


net = NetE2E(net_type=net_type_mapping[net_type], local_size=(1, 1),
                output_dimension=d_out_layer, reduce_function=None, n_noise_points=num_noise,
                pretrain=True, noise_on_mask=True)

os.makedirs(save_dir, exist_ok=True)

verts, faces = load_off(os.path.join(mesh_path, '01.off'), to_torch=True)
down_sample_rate = net.net_stride

feature_size = (image_size_ori[0] // down_sample_rate, image_size_ori[1] // down_sample_rate)
feature_size_m = (max(feature_size), max(feature_size))

feature_d = net_dfeature_mapping[net_type]

# viewpoint * focal
camera = PerspectiveCameras(focal_length=1.0 * 3000 / down_sample_rate,
                            principal_point=((feature_size_m[0] / 2, feature_size_m[1] / 2),),
                            image_size=(feature_size_m,), device=device)
raster_settings = RasterizationSettings(
    image_size=feature_size_m[0],
    blur_radius=0.0,
    faces_per_pixel=1,
    bin_size=0
)
rasterizer = MeshRasterizer(
    cameras=camera,
    raster_settings=raster_settings
)


inter_module = MeshInterpolateModule(vertices=[verts], faces=[faces],
                                     memory_bank=[torch.zeros((verts.shape[0], feature_d))], rasterizer=rasterizer,
                                     post_process=center_crop_fun(feature_size, feature_size_m))
inter_module = inter_module.cuda()

net = torch.nn.DataParallel(net.cuda())

if load_name is not None:
    checkpoint = torch.load(load_name, map_location='cuda:0')
    if 'state' in checkpoint:
        net.load_state_dict(checkpoint['state'])
    else:
        net.load_state_dict(checkpoint['net'])

for anchor_name in anchor_names:
    with torch.no_grad():
        all_error = []
        indexs = []
        names = []

        dataset = Pascal3D(transform=standard_transforms)
        dataloader = DataLoader(dataset=dataset, batch_size=20, shuffle=False)

        img_anchor, anno_anchor = dataset.get_list_images([anchor_name])

        campos = camera_position_from_spherical_angles(distance=5, azimuth=anno_anchor['azimuth'],
                                                       elevation=anno_anchor['elevation'], degrees=False).type(
            torch.float32).to(device)

        this_mask = get_mask(theta=anno_anchor['theta'].type(torch.float32).to(device), campos_=campos,
                             crop_size=image_size_ori, render_image_size=max(image_size_ori),
                             this_mesh=all_mask_meshes[int(anno_anchor['cad_index'].item())])

        _, verts_anchor, vis_anchor = do_render(inter_module, campos,
                                                theta=anno_anchor['theta'].type(torch.float32).to(device),
                                                verts_mask=this_mask)

        inter_module.update_memory([torch.zeros((verts.shape[0], feature_d)).to(device)])

        single_sample(inter_module, verts_label=verts_anchor, vis_label=vis_anchor, imgs=img_anchor, net_=net)

        for azum_sample in tqdm.tqdm(azimuth_samples):
            shift = {'azimuth': azum_sample, 'elevation': 0, 'theta': 0}

            campos = camera_position_from_spherical_angles(distance=5, azimuth=(anno_anchor['azimuth'] + shift['azimuth']), 
                elevation=(anno_anchor['elevation'] + shift['elevation']), degrees=False).type(torch.float32).to(device)

            get_map = inter_module.forward(campos.to(device), anno_anchor['theta'].type(torch.float32).to(device) + shift['theta'])

            all_sim, all_annos = retrieve_similarity(get_map, this_loader=dataloader, network=net)

            # this_idx = all_sim.argmax()
            this_idx = torch.topk(all_sim, k=args.n_retrieve, dim=0)[1]

            rotation_matrix_anchor = cal_rotation_matrix(theta=anno_anchor['theta'].item() + shift['theta'],
                                                         elev=anno_anchor['elevation'].item() + shift['elevation'],
                                                         azum=anno_anchor['azimuth'].item() + shift['azimuth'],
                                                         dis=5)

            this_error = []
            this_name = []
            for this_idx_ in this_idx:
                rotation_matrix_sel = cal_rotation_matrix(theta=all_annos['theta'][this_idx_].item(),
                                                          elev=all_annos['elevation'][this_idx_].item(),
                                                          azum=all_annos['azimuth'][this_idx_].item(),
                                                          dis=5)
                error_ = cal_err(rotation_matrix_anchor, rotation_matrix_sel)
                this_error.append(error_)
                this_name.append(all_annos['name'][this_idx_])
            all_error.append(np.mean(this_error))
            indexs.append(this_idx.cpu().squeeze())
            names += this_name

    # print(all_error)
    indexs = torch.cat(indexs)
    print('Image name:', anchor_name, ' Mean error:', np.mean(all_error))
    np.savez(os.path.join(save_dir, anchor_name), samples=azimuth_samples, errors=np.array(all_error), indexs=np.array(indexs), names=np.array(names))

# Plot the figure
if args.do_plot:
    colors = ['r', ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(colors):
        fl_list = os.listdir(save_dir)

        out = []

        get = {}
        for fname in fl_list:
            get = np.load(os.path.join(save_dir, fname), allow_pickle=True)
            out.append(get['errors'])

        x_axis = get['samples']

        avg_error = np.mean(np.stack(out), axis=0)

        ax.plot(x_axis, avg_error, c=c)

    plt.ylabel('Rotation Error ↓')
    plt.xlabel('Δθ on azimuth')
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'MatchingSingleAnchor.png'))
