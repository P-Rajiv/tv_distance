import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class add_noise:
    def __init__(self, type = 'noise', mean = 0, std = 0.1, img = None, patch_frac = 0.1):
        self.type = type
        self.mean = mean
        self.std = std
        self.img = img
        self.patch_frac = patch_frac
        self.torch_img = transforms.ToTensor()(img).unsqueeze(0)

    def random_patch(self, patch_frac):
        patch_size = int(patch_frac * self.torch_img.size(2))
        n_bins = int(1/patch_frac)
        x = np.random.randint(0, n_bins)
        x_start = x * patch_size
        x_end = x_start + patch_size
        y = np.random.randint(0, n_bins)
        y_start = y * patch_size
        y_end = y_start + patch_size
        return x_start, x_end, y_start, y_end


    def distorted_image(self, mean, std):
        image = self.torch_img
        size = image.size()
        noise = torch.normal(mean, std, size=size)
        image = image + noise
        # return PIL image
        return transforms.ToPILImage()(image.squeeze(0).cpu())

    def plot_img(self, image):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # image = (image - image.min()) / (image.max() - image.min())
        # image  = torch.clamp(image, 0, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def localised_noise(self, mean, std):
        img = self.torch_img
        size = img.size()
        noise = torch.normal(mean, std, size=size)
        x1, x2, y1, y2 = self.random_patch(self.patch_frac)
        noise[:, :, :x1, :] = 0
        noise[:, :, x2:, :] = 0
        noise[:, :, :, :y1] = 0
        noise[:, :, :, y2:] = 0

        image = img + noise  
        return transforms.ToPILImage()(image.squeeze(0).cpu())

    def flip_and_replace(self):
        img = self.torch_img
        size = img.size()
        x1, x2, y1, y2 = self.random_patch(self.patch_frac)
        img[:, :, x1:x2, y1:y2] = img[:, :, x1:x2, y1:y2].flip(2)
        return transforms.ToPILImage()(img.squeeze(0).cpu())
    
    def mask_image(self):
        img = self.torch_img
        size = img.size()
        x1, x2, y1, y2 = self.random_patch(self.patch_frac)
        img[:, :, x1:x2, y1:y2] = 0
        return transforms.ToPILImage()(img.squeeze(0).cpu())
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Add noise to image')  
    parser.add_argument('--corruption_type', type=str, default='noise', help='Type of corruption to apply')
    parser.add_argument('--mean', type=float, default=0, help='Mean of noise')
    parser.add_argument('--std', type=float, default=0.1, help='Standard deviation of noise')
    parser.add_argument('--src_folder', type=str, default='../../data/', help='Folder containing images')
    parser.add_argument('--dst_folder', type=str, default='../../data/', help='Folder containing images')
    parser.add_argument('--img', type=str, default='cat.jpg', help='Image to apply corruption to')
    parser.add_argument('--patch_frac', type=float, default=0.1, help='Fraction of image to apply corruption to')
    args = parser.parse_args()

    args = parser.parse_args()
    corruption_type = args.corruption_type
    mean = args.mean
    std = args.std
    src_folder = args.src_folder
    dst_folder = args.dst_folder
    img = args.img
    patch_frac = args.patch_frac

    dst_folder = os.path.join(src_folder.replace('val2017',''), dst_folder)
    os.makedirs(dst_folder, exist_ok=True)

    def corruption_fn(s, img, add_noise = add_noise):
        (mean, std,patch_frac) = (-0.5, 0.1, 0.1) if s == '1' else (0.5, 0.1, 0.1)
           
        add_noise = add_noise(corruption_type, mean, std, img, patch_frac)
        if corruption_type == 'noise':
            corruption_fn = add_noise.distorted_image(mean, std)
        elif corruption_type == 'localised_noise':
            corruption_fn = add_noise.localised_noise(mean, std)
        elif corruption_type == 'flip_and_replace':
            corruption_fn = add_noise.flip_and_replace()
        elif corruption_type == 'mask_image':
            corruption_fn = add_noise.mask_image()
    
        return corruption_fn


    for i in os.listdir(src_folder)[:len(os.listdir(src_folder))//2]:
        # add noise and save it in dst_folder
        img = Image.open(os.path.join(src_folder, i))
        corrupted_img = corruption_fn('1', img)
        corrupted_img.save(os.path.join(dst_folder, i))
        print(f"Saved {i} in {dst_folder}")
    for i in os.listdir(src_folder)[len(os.listdir(src_folder))//2:]:
        # add noise and save it in dst_folder
        img = Image.open(os.path.join(src_folder, i))
        corrupted_img = corruption_fn('2', img)
        corrupted_img.save(os.path.join(dst_folder, i))
        print(f"Saved {i} in {dst_folder}")
    

    