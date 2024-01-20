import os
import tqdm
import imageio
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.graphics_utils import focal2fov

from gaussian_core.cameras import Camera
from utils.colmap_utils import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


def _minify(basedir, factors=[], dir_name='images', resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, '{}_{}'.format(dir_name, r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, '{}_{}x{}'.format(dir_name, r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, dir_name)
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = '{}_{}'.format(dir_name, r)
            resizearg = '{}%'.format(100./r)
        else:
            name = '{}_{}x{}'.format(dir_name, r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            

def _preprocess_imgs(basedir, dir_name='images', factor=None, width=None, height=None, check_fn=lambda x: True):
    img0 = [os.path.join(basedir, dir_name, f) for f in sorted(os.listdir(os.path.join(basedir, dir_name))) if check_fn(f, 0)][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, dir_name=dir_name, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, dir_name=dir_name, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, dir_name=dir_name, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, dir_name + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for i, f in enumerate(sorted(os.listdir(imgdir))) if check_fn(f, i)]
    
    return imgfiles, factor

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, fg_mask=False, use_depth=False, gt_mask=True):

    check_colorimg_fn = lambda f, i: f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')

    if fg_mask:
        check_maskimg_fn = lambda f, i: f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')

    if use_depth:
        check_depthimg_fn = lambda f, i: f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    rgb_files, new_factor = _preprocess_imgs(basedir, dir_name='images', factor=factor, width=width, height=height, check_fn=check_colorimg_fn)

    if poses.shape[-1] != len(rgb_files):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(rgb_files), poses.shape[-1]))
        return
    
    sh = imageio.imread(rgb_files[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / new_factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, format="PNG-PIL", ignoregamma=True)
        else:
            return imageio.imread(f)
        
    rgb_imgs = [imread(f)[...,:3]/255. for f in rgb_files]
    rgb_imgs = np.stack(rgb_imgs, -1)

    mask_imgs = None
    if fg_mask:
        if gt_mask:
            mask_files, _ = _preprocess_imgs(basedir, dir_name='gt_masks', factor=factor, width=width, height=height, check_fn=check_maskimg_fn)    
        else:
            mask_files, _ = _preprocess_imgs(basedir, dir_name='masks', factor=factor, width=width, height=height, check_fn=check_maskimg_fn)

        if len(mask_files) != len(rgb_files):
            print( 'Mismatch between rgb imgs {} and mask imgs {} !!!!'.format(len(rgb_files), len(mask_files)))
            return
        
        mask_imgs = [imread(f) / 255.0 for f in mask_files]

        if mask_imgs[0].shape[:2] != rgb_imgs[..., 0].shape[:2]:
            print( 'Mismatch size between rgb imgs {} and mask imgs {} !!!!'.format(rgb_imgs[..., 0].shape[:2], mask_imgs[0].shape[:2]))
            return

        mask_imgs = np.stack(mask_imgs, -1)
        # Convert 0 for tool, 1 for not tool
        mask_imgs = 1.0 - mask_imgs

    depth_imgs = None
    if use_depth:
        depth_files, _ = _preprocess_imgs(basedir, dir_name='depth', factor=factor, width=width, height=height, check_fn=check_depthimg_fn)
        
        if len(depth_files) != len(rgb_files):
            print( 'Mismatch between rgb imgs {} and depth imgs {} !!!!'.format(len(rgb_files), len(depth_files)))
            return
        
        depth_imgs = [imread(f) for f in depth_files]

        if depth_imgs[0].shape[:2] != rgb_imgs[..., 0].shape[:2]:
            print( 'Mismatch size between rgb imgs {} and depth imgs {} !!!!'.format(rgb_imgs[..., 0].shape[:2], depth_imgs[0].shape[:2]))
            return
        
        depth_imgs = np.stack(depth_imgs, -1)
        # min_depth = np.percentile(depth_imgs, 3.0)
        # max_depth = np.percentile(depth_imgs, 99.9)
        # print('min depth:', min_depth, 'max depth:', max_depth)

    rgb_imgs = rgb_imgs[:500]
    mask_imgs = mask_imgs[:500]
    depth_imgs = depth_imgs[:500]

    print('Loaded image data', rgb_imgs.shape, poses[:,-1,0])
    return poses, bds, rgb_imgs, mask_imgs, depth_imgs

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def spatiotemporal_importance_from_masks(masks):
    freq = (1.0 - masks).sum(0)
    p = freq / torch.sqrt((torch.pow(freq, 2)).sum())

    return masks * (1.0 + 30*p)

class EndoDataset:
    def __init__(self, opt, device, type='train'):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, test
        self.root_path = opt.path
        self.start_index = opt.data_range[0]
        self.end_index = opt.data_range[1]
        self.training = self.type == 'train'
        self.sparse_path = os.path.join(self.root_path, 'sparse/')

        if not os.path.exists(os.path.join(self.root_path, 'gt_masks')):
            gt_mask = False
        else:
            gt_mask = True
        poses, _, imgs, masks, depth = _load_data(self.root_path, factor=None, fg_mask=True, use_depth=True, gt_mask=gt_mask) # factor=8 downsamples original imgs by 8x

        davinci_endoscopic = True
        if not davinci_endoscopic:
            poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(imgs, -1, 0).astype(np.float32)
        masks = np.moveaxis(masks, -1, 0).astype(np.float32)
        depth = np.moveaxis(depth, -1, 0).astype(np.float32)

        recenter = True
        if recenter and not davinci_endoscopic:
            poses = recenter_poses(poses)

        if self.end_index == -1:
            self.end_index = len(images)

        self.images = images[self.start_index:self.end_index]
        self.masks = masks[self.start_index:self.end_index]
        self.depth = depth[self.start_index:self.end_index]

        self.images = torch.from_numpy(self.images).permute(0, 3, 1, 2)
        self.masks = torch.from_numpy(self.masks)
        self.depth = torch.from_numpy(self.depth)

        self.spatialweight = spatiotemporal_importance_from_masks(self.masks)

        self.cameras = []
        self.time = []

        cameras_extrinsic_file = os.path.join(self.root_path, "sparse/", "images.bin")
        cameras_intrinsic_file = os.path.join(self.root_path, "sparse/", "cameras.bin")  

        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

        R_list = []
        T_list = []
        FovY_list = []
        FovX_list = []
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]

            height = intr.height
            width = intr.width

            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "OPENCV":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            
            R_list.append(R)
            T_list.append(T)
            FovY_list.append(FovY)
            FovX_list.append(FovX)

        R = np.mean(np.array(R_list), axis=0)
        T = np.mean(np.array(T_list), axis=0)
        FovY = np.mean(np.array(FovY_list))
        FovX = np.mean(np.array(FovX_list))

        for idx, f in tqdm.tqdm(enumerate(self.images), desc=f'Loading {type} data'):

            self.cameras.append(Camera(colmap_id=idx, R=R, T=T, 
                FoVx=FovX, FoVy=FovY, 
                image=f.to(self.device), gt_alpha_mask=None,
                image_name=str(idx), uid=id, data_device=self.device))
        
            self.time.append(torch.tensor(idx/len(self.images), dtype=torch.float32))

        print(f'[INFO] load {len(self.images)} {type} frames.')


    def mirror_index(self, index):
        size = len(self.cameras)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1


    def collate(self, index):

        B = len(index)
        assert B == 1

        results = {}

        index[0] = self.mirror_index(index[0])

        results['index'] = index
        results['H'] = self.images.shape[2]
        results['W'] = self.images.shape[3]

        results['camera'] = self.cameras[index[0]]
        results['time'] = self.time[index[0]]
        
        results['mask'] = self.masks[index[0]]
        results['depth'] = self.depth[index[0]]
        results['spatialweight'] = self.spatialweight[index[0]]

        results['sparse_path'] = self.sparse_path

        return results

    def __len__(self):
        return len(self.cameras)

    def dataloader(self):

        size = len(self.cameras)

        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self

        return loader