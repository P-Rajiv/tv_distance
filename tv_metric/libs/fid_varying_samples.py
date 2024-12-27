import random
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import numpy as np
import tqdm
import torch
import pandas as pd
import scipy


import os 
import shutil


real_images_path = '/data6/rajivporana_scratch/nih_test/images'
generated_images_path = '/data5/home/rajivporana/diffusers/generated_images/images'
noisy_image_path = '/data5/home/rajivporana/noisy_images/images' 
wh_real_images_path = '/data6/rajivporana_scratch/nih_data/test_images/whatsapp_compressed_test/images'

real_new_dir = 'real_new_dir'
gen_new_dir = 'gen_new_dir'                                                   
wh_new_dir = 'wh_new_dir'
noisy_new_dir = 'noisy_new_dir'



samples_ls = [200, 500, 1000, 5000, 10000, 15000]

# write code to compute fid using pytorch-fid library
from pytorch_fid import fid_score
import csv

# compute fid between real and generated images
device = 'cuda:1'
csv_file = 'fid_scores.csv'
csv_columns = ['num_samples', 'comparison', 'fid_score']

if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)

for i in samples_ls:

    os.makedirs(real_new_dir, exist_ok=True)
    os.makedirs(gen_new_dir, exist_ok=True)
    os.makedirs(wh_new_dir, exist_ok=True)
    os.makedirs(noisy_new_dir, exist_ok=True)
    # randomly sample i images from the real and generated data and compute the FID
    real_images = os.listdir(real_images_path)
    gen_images = os.listdir(generated_images_path)
    wh_images = os.listdir(wh_real_images_path)
    noisy_images = os.listdir(noisy_image_path)

    real_sample = random.sample(real_images, i)
    gen_sample = random.sample(gen_images, i)
    wh_sample = random.sample(wh_images, i)
    noisy_sample = random.sample(noisy_images, i)

    for img in real_sample:
        shutil.copy(real_images_path + '/' + img, real_new_dir)
    for img in gen_sample:
        shutil.copy(generated_images_path + '/' + img, gen_new_dir)
    for img in wh_sample:
        shutil.copy(wh_real_images_path + '/' + img, wh_new_dir)
    for img in noisy_sample:
        shutil.copy(noisy_image_path + '/' + img, noisy_new_dir)

    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
                
        fid_real_gen = fid_score.calculate_fid_given_paths([real_new_dir, gen_new_dir], 256, device, 2048)
        writer.writerow([i, 'real_vs_generated', fid_real_gen])
        print(f'FID between real and generated images with {i} samples: {fid_real_gen}')
        
        fid_real_wh = fid_score.calculate_fid_given_paths([real_new_dir, wh_new_dir], 256, device, 2048)
        writer.writerow([i, 'real_vs_whatsapp', fid_real_wh])
        print(f'FID between real and whatsapp images with {i} samples: {fid_real_wh}')
        
        fid_real_noisy = fid_score.calculate_fid_given_paths([real_new_dir, noisy_new_dir], 256, device, 2048)
        writer.writerow([i, 'real_vs_noisy', fid_real_noisy])
        print(f'FID between real and noisy images with {i} samples: {fid_real_noisy}')

        fid_gen_wh = fid_score.calculate_fid_given_paths([gen_new_dir, wh_new_dir], 256, device, 2048)
        writer.writerow([i, 'generated_vs_whatsapp', fid_gen_wh])
        print(f'FID between generated and whatsapp images with {i} samples: {fid_gen_wh}')


    print('-'*50)

    # remove the images from the new directories
    os.system(f'rm -rf {real_new_dir}')
    os.system(f'rm -rf {gen_new_dir}')
    os.system(f'rm -rf {wh_new_dir}')
    os.system(f'rm -rf {noisy_new_dir}')


    print(f'Removed images from {real_new_dir}, {gen_new_dir}, {wh_new_dir}, {noisy_new_dir}')  
    print('='*50)



