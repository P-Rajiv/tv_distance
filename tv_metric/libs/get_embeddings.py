from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import numpy as np
import tqdm
import pandas as pd
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms
from urllib.request import urlopen
import json
import timm
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from torchvision.datasets import imagenet




# dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'

# import ImageDataset


class get_embeddings(nn.Module):    
    def __init__(self, model_name='clip', device = None, embed_dim=None):
        super(get_embeddings, self).__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name
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
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16)
        ls_image_embeds = []
        for batch in tqdm.tqdm(data_loader):
            images = batch[0].to(self.device)

            img_embeds = self.model(images)
            image_embeds = img_embeds[0].cpu().detach()
            ls_image_embeds.append(image_embeds)
        print(ls_image_embeds[0].shape)
        final_image_embeds = torch.cat(ls_image_embeds, dim=0)
        return final_image_embeds



# ------------------------
# Get activations from InceptionV3
# ------------------------

transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def get_activations(real_images_path, device="cuda"):
    # model = timm.create_model('inception_v3', pretrained=False, num_classes = 15)
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    #instead of pretrained weights use : /data5/home/rajivporana/tv_metric/saved_models/inception-lr:0.0001-wd:0.0_summer-mountain-107_best_model.pkl
    # model.load_state_dict(torch.load('/data5/home/rajivporana/tv_metric/saved_models/inception-lr:0.0001-wd:0.0_summer-mountain-107_best_model.pkl'), strict=False)
    model.fc = nn.Identity()  # Remove the classification layer
    model.eval()
    # get 2048 dimensional embeddings from the model
    # model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    # GET IMAGES from path and apply transform  
    dataset = ImageFolder(root=real_images_path, transform=transform)
    real_data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)

    real_activations = []

    with torch.no_grad():
        # using tqdm to display progress bar
        for img, _ in tqdm.tqdm(real_data_loader):
            img = img.to(device)
            activations = model(img).detach()
            # activations = activations.squeeze(-1).squeeze(-1) # b 2048 1 1 -> b 2048        
            real_activations.append(activations.cpu())
            # break
    real_activations = torch.cat(real_activations, dim=0)
    return real_activations


# ------------------------

def calculate_statistics(real_activations, generated_activations):
    # convert list to tensor
    real_activations = torch.cat(real_activations, dim=0)
    generated_activations = torch.cat(generated_activations, dim=0)
    real_mu = torch.mean(real_activations, dim = 1)
    real_sigma = torch.cov(real_activations)
    generated_mu = torch.mean(generated_activations, dim = 1)
    generated_sigma = torch.cov(generated_activations)

    return real_mu, real_sigma, generated_mu, generated_sigma

# ------------------------
# Classwise statistics
# ------------------------


class classwise_activation(Dataset):
    def __init__(self, model_name = 'inception',type_dataset='real', csv_path='/data5/home/rajivporana/cxr_project/neurips_dataset/test_images_labels.csv'):
        super().__init__()
        self.type_dataset = type_dataset
        self.csv_path = csv_path
        self.model_name = model_name
        # Set paths based on dataset type
        if self.model_name == 'inception':
            if self.type_dataset == 'real':
                self.activations_path = '/data5/home/rajivporana/tv_metric/real_actvn.pt'
                self.images_path = '/data6/rajivporana_scratch/nih_test'
            elif self.type_dataset == 'generated':
                self.activations_path = '/data5/home/rajivporana/tv_metric/gen_actvn.pt'
                self.images_path = '/data5/home/rajivporana/diffusers/generated_images'
            elif self.type_dataset == 'noisy':
                self.activations_path = '/data5/home/rajivporana/tv_metric/noisy_actvn.pt'
                self.images_path = '/data5/home/rajivporana/noisy_images'
        else:
            if self.type_dataset == 'real': 
                self.activations_path = '/data5/home/rajivporana/tv_metric/real_clip_embeddings.pt'
                self.images_path = '/data6/rajivporana_scratch/nih_test'
            elif self.type_dataset == 'generated':
                self.activations_path = '/data5/home/rajivporana/tv_metric/fake_clip_embeddings.pt'
                self.images_path = '/data5/home/rajivporana/diffusers/generated_images'
            elif self.type_dataset == 'noisy':
                self.activations_path = '/data5/home/rajivporana/tv_metric/noise_clip_embeddings.pt'
                self.images_path = '/data5/home/rajivporana/noisy_images'

        # Load activations
        self.activations = torch.load(self.activations_path, map_location='cpu')
        if not model_name == 'clip':
            self.activations = torch.cat(self.activations, dim=0)

        # Load labels CSV
        self.labels = pd.read_csv(self.csv_path)
        self.labels.set_index('image path', inplace=True)

        # Image dataset
        self.dataset = ImageFolder(self.images_path, transform=transforms.ToTensor())
        self.image_names = [os.path.splitext(os.path.basename(self.dataset.samples[i][0]))[0] for i in range(len(self.dataset))]

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        activation = self.activations[idx]
        if image_name in self.labels.index:
            label = torch.tensor(self.labels.loc[image_name].values, dtype=torch.float32)
        else:
            raise ValueError(f"Image {image_name} not found in labels CSV.")
        return activation, label, image_name

    def __len__(self):
        return len(self.dataset)

        
def get_classwise_statistics(real_activations, generated_activations, class_idx, type_dataset, model_name):
    if model_name == 'inception':
        dataset = classwise_activation(type_dataset=type_dataset)
    else:
        dataset = classwise_activation(type_dataset=type_dataset, model_name='clip')
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)

    class_activations = []
    for actvn, label, _ in data_loader:
        # Filter activations where the label for the given class index is 1
        mask = label[:, class_idx] == 1
        if mask.any():
            class_activations.append(actvn[mask])

    # Concatenate activations across batches
    if len(class_activations) > 0:
        class_activations = torch.cat(class_activations, dim=0)
    else:
        raise ValueError(f"No activations found for class index {class_idx}.")

    # Calculate mean and covariance
    mean = torch.mean(class_activations, dim=0)
    centered_activations = class_activations - mean
    covariance = centered_activations.T @ centered_activations / (centered_activations.size(0) - 1)

    # return mean, covariance
    return class_activations

# ------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='clip')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='nih')
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name
    device = args.device


    chexphoto = {'whatsapp_train_data': '/data6/rajivporana_scratch/chexphoto_data/CheXphoto-v1.0/train_data/wh_proper_labelled_train', 
                 'train_data':'/data6/rajivporana_scratch/chexphoto_data/CheXphoto-v1.0/train_data/all_train',
                 'whatsapp_test_data': '/data6/rajivporana_scratch/chexphoto_data/CheXphoto-v1.0/wh_proper_test/',
                 'test_data' : '/data6/rajivporana_scratch/chexphoto_data/CheXphoto-v1.0/all_valid'}
    
    nih = {'real_images_path' :'/data6/rajivporana_scratch/nih_test',
        'generated_images_path' :'/data5/home/rajivporana/diffusers/generated_images',
        'wh_real_images_path' :'/data6/rajivporana_scratch/nih_data/test_images/whatsapp_compressed_test'
        }

    if dataset == 'nih':
        real_images_path = nih['real_images_path']
        generated_images_path = nih['generated_images_path']
        wh_real_images_path = nih['wh_real_images_path']
    else:
        train_path = chexphoto['train_data']
        whatsapp_train_path = chexphoto['whatsapp_train_data']
        test_path = chexphoto['test_data']
        whatsapp_test_path = chexphoto['whatsapp_test_data']

    noisy_image_path = '/data5/home/rajivporana/noisy_images' 
    
    # real_actvn = get_activations(wh_real_images_path, device='cuda')
    # gen_actvn = get_activations(generated_images_path, device='cuda')
    # noise_actvn = get_activations(noisy_image_path, device='cuda')
    # wh_actvn = get_activations(wh_real_images_path, device='cuda')

    
    # mu1, sigma1, mu2, sigma2 = calculate_statistics(real_actvn, gen_actvn)
    # print(' --------- ')    
    # get clip embeddings 
    clip = get_embeddings(model_name='clip', device='cuda')
    train_clip_embeddings = clip(train_path)
    test_clip_embeddings = clip(test_path)
    wh_train_clip_embeddings = clip(whatsapp_train_path)
    wh_test_clip_embeddings = clip(whatsapp_test_path)


    # store them in .pt files
    torch.save(train_clip_embeddings, 'embeddings/chexphoto/train_clip_embeddings.pt')
    torch.save(test_clip_embeddings, 'embeddings/chexphoto/test_clip_embeddings.pt')
    torch.save(wh_train_clip_embeddings, 'embeddings/chexphoto/wh_train_clip_embeddings.pt')
    torch.save(wh_test_clip_embeddings, 'embeddings/chexphoto/wh_test_clip_embeddings.pt')






