import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
# from scipy.ndimage.interpolation import zoom, gaussian_filter
from scipy.ndimage import zoom, gaussian_filter
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)     #角度收紧 避免极端旋转 之前为-20~20
    image = ndimage.rotate(image, angle, order=1, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

### add random aug function ###
# ========= 光照 / 对比度 =========

def random_gamma_contrast(image, p=0.5):
    """随机 gamma / 对比度 调整"""
    if random.random() < p:
        # 保证是 float
        img = image.astype(np.float32)

        # 如果是 0-255，则归一化到 0-1 再操作
        if img.max() > 1.5:
            img = img / 255.0

        # gamma
        gamma = np.random.uniform(0.8, 1.2)
        img = np.power(img, gamma)

        # 对比度缩放
        factor = np.random.uniform(0.8, 1.2)
        mean = img.mean()
        img = (img - mean) * factor + mean

        img = np.clip(img, 0.0, 1.0)
        # 若原来是 0-255，可以乘回去；这里直接保持 0-1 更方便后续归一化
        image = img
    return image

# # ========= 模糊 / 噪声 =========

# def random_gaussian_blur(image, p=0.3):
#     if random.random() < p:
#         sigma = np.random.uniform(0.5, 1.2)
#         if image.ndim == 2:
#             image = gaussian_filter(image, sigma=sigma)
#         else:
#             # 分通道模糊
#             for c in range(image.shape[2]):
#                 image[..., c] = gaussian_filter(image[..., c], sigma=sigma)
#     return image


def random_gaussian_noise(image, p=0.3):
    if random.random() < p:
        img = image.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        noise = np.random.normal(0, 0.02, size=img.shape)  # 适度小噪声
        img = np.clip(img + noise, 0.0, 1.0)
        image = img
    return image

# # ========= 遮挡（模拟叶片/反光） =========

def random_occlusion(image, label, p=0.3):
    """随机放一个浅色或深色块，只改 image，不改 label"""
    if random.random() < p:
        H, W = image.shape[:2]
        # 遮挡块尺度相对图像大小
        occ_h = np.random.randint(H // 10, H // 5)
        occ_w = np.random.randint(W // 10, W // 5)
        y1 = np.random.randint(0, H - occ_h)
        x1 = np.random.randint(0, W - occ_w)

        if image.ndim == 2:
            patch = np.random.uniform(0.7, 1.0)  # 偏亮块
            image[y1:y1+occ_h, x1:x1+occ_w] = patch
        else:
            patch = np.random.uniform(0.7, 1.0, size=(occ_h, occ_w, image.shape[2]))
            image[y1:y1+occ_h, x1:x1+occ_w, :] = patch

    return image, label

# ========= 根区域特异增强 =========

def random_root_intensity(image, label, p=0.5):
    """
    只对前景(根)区域做轻微亮度缩放，
    防止模型只记住固定灰度值的根。
    """
    if random.random() < p:
        img = image.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0

        fg = (label > 0)
        if fg.any():
            factor = np.random.uniform(0.9, 1.1)
            if img.ndim == 2:
                img[fg] = np.clip(img[fg] * factor, 0.0, 1.0)
            else:
                for c in range(img.shape[2]):
                    img[..., c][fg] = np.clip(img[..., c][fg] * factor, 0.0, 1.0)

        image = img
    return image
### add random aug function ###


# ## 原始代码 ###
# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']

#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)
        
#         x, y = image.shape

#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.float32))
#         sample = {'image': image, 'label': label.long()}
#         return sample
# ## 原始代码 ###

class RandomGenerator(object):
    def __init__(self, output_size):
        # output_size = [H, W]
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.asarray(image)
        label = np.asarray(label)

        # 你的随机旋转/翻转（注意它们也要支持 HWC，见下方备注）
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        ### new aug ###
        # ---- 光照 / 对比度 ----
        image = random_gamma_contrast(image, p=0.5)

        # ---- 模糊 / 噪声 ----
        # image = random_gaussian_blur(image, p=0.3)
        image = random_gaussian_noise(image, p=0.3)

        # ---- 遮挡 (模拟叶片 / 反光) ----
        image, label = random_occlusion(image, label, p=0.3)

        # ---- 根区域特异增强 ----
        image = random_root_intensity(image, label, p=0.5)
        ### new aug ###

        # 统一拿到 H, W
        H, W = image.shape[:2]
        outH, outW = self.output_size

        # 尺寸不一致就缩放：图像用三次插值，标签用最近邻
        if (H, W) != (outH, outW):
            if image.ndim == 2:
                # [H, W]
                image = zoom(image, (outH / H, outW / W), order=3)
            else:
                # [H, W, C] —— 通道不缩放
                image = zoom(image, (outH / H, outW / W, 1), order=3)
            label = zoom(label, (outH / H, outW / W), order=0)

        # 类型/取值规范
        image = image.astype(np.float32)
        label = (label > 0).astype(np.uint8)   # 保二值

        # 统一成 CHW
        if image.ndim == 2:                    # [H, W] -> [1, H, W]
            image = image[np.newaxis, ...]
        else:                                   # [H, W, C] -> [C, H, W]
            image = np.transpose(image, (2, 0, 1))

        # 转 Torch Tensor
        image = torch.from_numpy(image.copy())          # float32, [C,H,W]
        label = torch.from_numpy(label.copy()).long()   # long, [H,W]

        return {'image': image, 'label': label}

# --- 放在 dataset_synapse.py 中 RandomGenerator 类之后 ---

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.asarray(image)
        label = np.asarray(label)

        # 验证集严禁使用 random_rot_flip, random_rotate, 噪声, 遮挡等增强操作
        # 仅保留尺寸缩放和格式转换

        H, W = image.shape[:2]
        outH, outW = self.output_size

        if (H, W) != (outH, outW):
            if image.ndim == 2:
                image = zoom(image, (outH / H, outW / W), order=3)
            else:
                image = zoom(image, (outH / H, outW / W, 1), order=3)
            label = zoom(label, (outH / H, outW / W), order=0)

        image = image.astype(np.float32)
        label = (label > 0).astype(np.uint8)

        if image.ndim == 2:
            image = image[np.newaxis, ...]
        else:
            image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image.copy())
        label = torch.from_numpy(label.copy()).long()

        return {'image': image, 'label': label}

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split

        ### 增加了 if elif 判断 ###
        if split == 'train':
            self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()    ### 原始代码两行
            self.data_dir = base_dir                                                          ### 原始代码两行
        elif split == 'test_npz':
            fname = self.split + '.txt'
            fname = 'test.txt'  # ← 固定读取 test.txt
            self.sample_list = open(os.path.join(list_dir, fname), encoding='utf-8').read().splitlines()
            self.data_dir = base_dir
        ### 增加了 if elif 判断 ###
        else:
            self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
            self.data_dir = base_dir


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "val":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']

        ### add ###
        elif self.split == "test_npz":
            name = self.sample_list[idx].strip()  # 来自 test.txt 的基名
            # 允许清单里直接写绝对路径或带 .npz；否则拼到 data_dir
            if os.path.isabs(name) or name.endswith(".npz"):
                npz_path = name
            else:
                npz_path = os.path.join(self.data_dir, name + ".npz")

            data = np.load(npz_path)
            image = data["image"]                       # HxW 或 HxWxC
            label = data["label"] if "label" in data.files else None
            if label is None:
                # 若测试集没有标签就给个占位；若有标签用于评估则会覆盖
                import numpy as _np
                label = _np.zeros(image.shape[:2], dtype=_np.uint8)
            else:
                label = (label > 0).astype(np.uint8)
        ### add ###
          
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
