# from PIL import Image
# import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import torch
import torch.nn as nn
# from torchvision.models import inception_v3
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# from pytorch_fid import fid_score
# from pytorch_fid.inception import InceptionV3
# import numpy as np
import tqdm
# import pandas as pd
# import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms
# from urllib.request import urlopen
import json
import timm
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import math

 
#---------------------------------------------
 
def exp_func(x, c=1):
    x =  torch.norm(x, dim=1, p=3)   
    x = - x * c
    exp = torch.exp(x)
    return exp
 
 
def sum_func(x):
 
    return
#---------------------------------------------
 
def tv_distance(dist1, dist2, bins, c):
    phi1 = exp_func(dist1, c=c)
    phi2 = exp_func(dist2, c=c)
  
    max1 = torch.max(phi1)
    max2 = torch.max(phi2)
    min1 = torch.min(phi1)
    min2 = torch.min(phi2)
    max = torch.maximum(max1, max2).item()
    min = torch.minimum(min1, min2).item()
    print("Max : ", max, math.ceil(max))
    print("Min : ", min, math.floor(min))
    hist1, bin_edges1 = torch.histogram(phi1, bins=bins, range=(math.floor(min), math.ceil(max)))
    hist2, bin_edges2 = torch.histogram(phi2, bins=bins, range=(math.floor(min), math.ceil(max)))
    print(hist1)
    dist_diff = 0
    N1 = dist1.shape[0]
    N2 = dist2.shape[0]
 
    for i in range(bins):
        c1 = hist1[i]
        c2 = hist2[i]
        diff = abs(c1/N1 - c2/N2)
        dist_diff += diff
    print(dist_diff)        
    return
 
 
 
#---------------------------------------------
 
# dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
 
class CLIP_embeddings(nn.Module):    
    def __init__(self, img_size, batch_size=128, model_name='clip', device = None, embed_dim=None):
        super(CLIP_embeddings, self).__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.original_clip = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = device
        self.batch_size = batch_size
        if self.model_name == 'clip':
            self.model = CLIPVisionModelWithProjection.from_pretrained(self.original_clip).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.original_clip)
            self.model.eval()
 

 
        
    def forward(self, path):
        dataset = ImageFolder(root=path, transform=self.transform)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)
        ls_image_embeds = []
        for batch in tqdm.tqdm(data_loader):
            images = batch[0].to(self.device)
 
            img_embeds = self.model(images)
            image_embeds = img_embeds[0].cpu().detach()
            ls_image_embeds.append(image_embeds)
            torch.cuda.empty_cache()
        print(ls_image_embeds[0].shape)
        final_image_embeds = torch.cat(ls_image_embeds, dim=0)
        print(final_image_embeds.shape)
        return final_image_embeds

class get_embeddings(nn.Module):
    def __init__(self, device = None, embed_dim=None):
        super(get_embeddings, self).__init__()
        self.embed_dim = embed_dim
        self.model_name = "biomedclip_local"
        self.original_clip = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        self.medical_clip = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = device
        if self.model_name == 'clip':
            self.model = CLIPVisionModelWithProjection.from_pretrained(self.medical_clip).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.medical_clip)
            self.model.eval()

        with open('../diffusers/bio_clip_ckpts/open_clip_config.json', 'r') as f:
            self.config = json.load(f)
            self.model_cfg = self.config['model_cfg']
            self.preprocess_cfg = self.config['preprocess_cfg']

        if (not self.model_name.startswith(HF_HUB_PREFIX) and self.model_name not in _MODEL_CONFIGS and self.config is not None):
            _MODEL_CONFIGS[self.model_name] = self.model_cfg
        
        self.model, self._, self.preprocessor = create_model_and_transforms(
            model_name = self.model_name,
            pretrained = '../diffusers/bio_clip_ckpts/open_clip_pytorch_model.bin',
            **{f"image_{k}": v for k, v in self.preprocess_cfg.items()}
            )   
        self.model.to(self.device)
        self.model.eval()
        
    def forward(self, path):
        dataset = ImageFolder(root=path, transform=self.transform)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)
        ls_image_embeds = []
        for batch in tqdm.tqdm(data_loader):
            images = batch[0].to(self.device)

            img_embeds = self.model(images)
            image_embeds = img_embeds[0].cpu().detach()
            ls_image_embeds.append(image_embeds)
        print(ls_image_embeds[0].shape)
        final_image_embeds = torch.cat(ls_image_embeds, dim=0)
        return final_image_embeds


def save_embeddings(embeddings, path):
    torch.save(embeddings, f'{path.split("/")[-1]}.pt')
    return
 
# ------------------------
# ------------------------
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='clip')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='nih')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--c', type=float, default=-1)
    parser.add_argument('--path1', type=str, default='/data6/rajivporana_scratch/coco_2017/images/val2017')
    parser.add_argument('--path2', type=str, default='/data6/rajivporana_scratch/coco_2017/images/val2017')
    parser.add_argument('--save_path', type=str, default='.')
    args = parser.parse_args()
 
    dataset = args.dataset
    model_name = args.model_name
    device = args.device
    
    save_path = args.save_path

    path1 = args.path1
    # path2 = args.path2
 
    # clip = CLIP_embeddings(args.img_size, device=device, batch_size=args.batch_size)
    # path1_embeddings = clip(path1)
    path1_embeddings = get_embeddings(device=device)(path1)
 
    # tv_distance(path1_embeddings, path2_embeddings, args.bins, args.c)
    
    save_embeddings(path1_embeddings, save_path)
    print(f"Saved embeddings for {path1}")
    # save_embeddings(path2_embeddings, path2)
    # print(f"Saved embeddings for {path2}")
