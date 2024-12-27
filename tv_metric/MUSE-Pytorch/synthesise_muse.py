from PIL import Image
import torch
import os
from torchvision.utils import save_image
import taming.models.vqgan
from libs.muse import MUSE
import einops
import utils
import torch
from transformers import BertTokenizer, BertModel
import ml_collections
import numpy as np
from torchvision import transforms


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.z_shape = (8, 16, 16)

    config.autoencoder = d(
        config_file='vq-f16-jax.yaml',
    )

    config.train = d(
        n_steps=99999999,
        batch_size=2048,
        log_interval=10,
        eval_interval=5000,
        save_interval=5000,
        fid_interval=50000,
    )

    config.eval = d(
        n_samples=10000,
        sample_steps=12,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0004,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_t2i_vq',
        img_size=16,
        codebook_size=2025,
        in_chans=256,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        use_checkpoint=False,
        skip=True,
    )

    config.muse = d(
        ignore_ind=-1,
        smoothing=0.1,
        gen_temp=4.5
    )

    config.dataset = d(
        name='imagenet256_features',
        path='assets/datasets/imagenet256_vq_features/vq-f16-jax',
        cfg=True,
        p_uncond=0.15,
    )

    config.sample = d(
        sample_steps=12,
        n_samples=50000,
        mini_batch_size=50,
        cfg=True,
        linear_inc_scale=True,
        scale=3.,
        path=''
    )
    return config


config = get_config()


nnnet_ckpt_path = '/data5/home/rajivporana/tv_metric/MUSE-Pytorch/pre_trained_models/nnet.pth'
vqgan_ckpt_path = '/data5/home/rajivporana/tv_metric/MUSE-Pytorch/pre_trained_models/vqgan_jax_strongaug.ckpt'

device = 'cuda:7'
# Initialize the autoencoder (e.g., VQGAN)
# autoencoder = taming.models.vqgan.get_model(**config.autoencoder)
# autoencoder.to(device)
# autoencoder.eval()

# Initialize MUSE and nnet (your pre-trained network)
# muse = MUSE(codebook_size=autoencoder.n_embed, device=device, **config.muse)
# nnet_kwargs = {k: v for k, v in config.nnet.items() if k != 'name'}
# nnet_ema =utils.get_nnet(name = config.nnet.name, **nnet_kwargs)
# nnet_ema.load_state_dict(torch.load(nnnet_ckpt_path), strict=False)

# nnet_ema.eval()
# nnet_ema.to(device)

# # Function for classifier-free guidance
# def cfg_nnet(x, context, scale=5.0):  # Adjust `scale` for guidance strength
#     _cond = nnet_ema(x, context=context)
#     _empty_context = torch.zeros_like(context)  # Assuming zeroed context represents unconditional
#     _uncond = nnet_ema(x, context=_empty_context)
#     return _cond + scale * (_cond - _uncond)

# Decoding function for converting discrete tokens to images
# @torch.no_grad()
# def decode(tokens):
#     return autoencoder.decode_code(tokens)


# context = text = 'a cartoon of a dog'

# # Load pre-trained model and tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# # Tokenize the input text
# inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=512)

# # Get the embeddings from the BERT model
# with torch.no_grad():
#     outputs = model(**inputs)

# # Extract the last hidden state, which contains the embeddings
# context_embeddings = outputs.last_hidden_state

# # Prepare text embedding (Replace this with your actual embedding)
# text_embedding = torch.tensor(context_embeddings, device=device)  # Shape: [B, ...]

# # Generate the image
# num_samples = 1  # Number of images to generate
# generated_images = muse.generate(
#     config=config,
#     _n_samples=num_samples,
#     nnet=cfg_nnet,
#     decode_fn=decode,
#     context=text_embedding
# )

# # Save or display the generated image
# output_path = "muse_generated_image.png"
# save_image(generated_images, output_path)
# print(f"Image saved to {output_path}")


# write a code to take image encode and reconstruct using taming transformer
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import taming.models.vqgan

# Load your configuration here
# config = ...

file_path = '/data6/rajivporana_scratch/coco_2017/images/val2017/images/000000574823.jpg'

# Encode image using taming transformer
autoencoder = taming.models.vqgan.get_model(**config.autoencoder)
autoencoder.to(device)
autoencoder.eval()

# Load and preprocess the image
img = Image.open(file_path)
img = img.convert('RGB')
img = np.array(img)

# Convert image to tensor and normalize
img = torch.tensor(img, device=device).permute(2, 0, 1).to(torch.float32) 

# Normalize image
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
img = transform(img)

# Encode the image to get latents
latents = autoencoder.encode(img)
latents = latents[0]  # Extract the tensor from the tuple

# Ensure latents are of type Long for embedding
latents = latents.long()  # Convert to Long tensor if necessary

#_z = rearrange(sampled_ids, 'b (i j) -> b i j', i=fmap_size, j=fmap_size)


# Decode the latents back to an image
rec_img = autoencoder.decode_code(latents)

# Save the reconstructed image
output_path = "vqgan_muse_reconstruction.png"
save_image(rec_img, output_path)
print(f"Reconstructed image saved to {output_path}")
