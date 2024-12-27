import os
import torch
from torchvision import transforms
from PIL import Image
from taming.models import vqgan
import yaml


def load_vqgan(path):
    vqgan_config_path = 'vqgan.yaml'
    config = yaml.safe_load(open(vqgan_config_path))
    ddconfig = config['model']['params']['ddconfig']
    n_embed = config['model']['params']['n_embed']
    lossconfig = config['model']['params']['lossconfig']
    embed_dim = config['model']['params']['embed_dim']
    model = vqgan.VQModel(ddconfig=ddconfig, n_embed=n_embed, lossconfig=lossconfig, embed_dim=embed_dim)
    # load
    model.eval().requires_grad_(False)
    model.load_state_dict(torch.load(path))
    return model

def get_latent_codes(model, img):
    img = transforms.ToTensor()(img).unsqueeze(0).cuda()
    z, _, [_, _, indices] = model.encode(img)
    return z, indices   

def corrupt_latent_codes(latent_codes, corruption_factor=0.1):
    corrupted_codes = []
    for code in latent_codes:
        noise = torch.randn_like(code) * corruption_factor
        corrupted_code = code + noise  # Add noise to corrupt the latent code
        corrupted_codes.append(corrupted_code)
    return corrupted_codes

def generate_images(model, corrupted_codes):
    generated_images = []
    for code in corrupted_codes:
        with torch.no_grad():
            generated_image = model.decode(code)  # Decode to generate image
            generated_images.append(generated_image)
    return generated_images

folder = '/data5/home/rajivporana/diffusers/parti_images'

for i in os.listdir(folder):
    image = Image.open(f'{folder}/{i}').convert('RGB')  
    model = load_vqgan('model.ckpt').cuda()
    latent_codes, indices = get_latent_codes(model, image)
    corrupted_codes = corrupt_latent_codes(latent_codes)
    corrupted_image = generate_images(model, corrupted_codes)
    # save corrupted image
    corrupted_image[0].add_(1).div_(2).mul(255).clamp(0, 255)
    corrupted_image[0] = corrupted_image[0].permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    corrupted_image = Image.fromarray(corrupted_image[0])
    corrupted_image.save(f'/data5/home/rajivporana/diffusers/corrupted_parti_images/{i}')
    print(f'Corrupted image saved as corrupted_images/{i}')

