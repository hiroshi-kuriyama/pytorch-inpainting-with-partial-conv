import random
import torch
from PIL import Image
from PIL import ImageFilter
from glob import glob


class CelebA(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(CelebA, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        self.paths = glob('{:s}/{:s}/*.jpg'.format(img_root, split),
                          recursive=True)
        self.mask_paths = glob('{:s}/*.png'.format(mask_root))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        # https://nv-adlr.github.io/publication/partialconv-inpainting
        # 影の部分を太くする
        mask = mask.filter(ImageFilter.MinFilter(random.choice([3,5,7])))
        # 回転
        mask = mask.rotate(random.randint(0, 360), fillcolor=(255), expand=True)

        mask = self.mask_transform(mask.convert('RGB'))
        mask = (mask > 0.6) * 1.0
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
