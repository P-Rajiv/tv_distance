import torch 
import torch.functional as F
from torch import abs as abs
from torch import sqrt as sqrt
from torch import inverse as inverse
from get_embeddings import *
import torch.distributions as dist

def simple_tv_distance(real_actvn, gen_actvn, c = 1.):
    # Step 1: Compute f(i) = exp(-c * ||i||^2) for each activation
    f_real = torch.exp(-c * torch.norm(real_actvn, p=2, dim=1) ** 2)
    f_fake = torch.exp(-c * torch.norm(gen_actvn, p=2, dim=1) ** 2)
    
    # Step 2: Compute the mean of f() for real and fake activations
    mean_real = f_real.mean()
    mean_fake = f_fake.mean()    

    # tv distance = (mean_real - mean_fake)**2 / (2* mean_real**2 + 2*mean_fake**2)
    tv_distance = (mean_real - mean_fake)**2 / (2* (mean_real**2 + mean_fake**2))
    return  tv_distance

def fisher_tv_distance(real_actvn, gen_actvn, c=1.):
    
    real_mean = real_actvn.mean(dim=0) # mean
    fake_mean = gen_actvn.mean(dim=0)

    f_real_centered = real_actvn - real_mean # centering
    f_fake_centered = gen_actvn - fake_mean

    cov_real = f_real_centered.T @ f_real_centered / (f_real_centered.size(0) - 1) # covariance
    cov_fake = f_fake_centered.T @ f_fake_centered / (f_fake_centered.size(0) - 1)

    mean_diff = real_mean - fake_mean
    cov_sum = cov_real + cov_fake
    inv_cov_sum = torch.linalg.inv(cov_sum)
    distance_fisher = mean_diff.T @ inv_cov_sum @ mean_diff # fisher distance

    result = distance_fisher / (2 + distance_fisher) # tv distance
    
    return result

def classwise_fisher_tv(mu1, sigma1, mu2, sigma2):
    mean_diff = mu1 - mu2
    cov_sum = sigma1 + sigma2
    inv_cov_sum = torch.linalg.inv(cov_sum)
    distance_fisher = mean_diff.T @ inv_cov_sum @ mean_diff # fisher distance

    tv_dist = distance_fisher / (2 + distance_fisher) # tv distance

    return tv_dist


# original_image_path = '/data6/rajivporana_scratch/nih_test'
generated_image_path = '/data5/home/rajivporana/diffusers/generated_images'
noisy_image_path = '/data5/home/rajivporana/noisy_images' 
# generated_image_path = original_image_path
# noisy_actvn, gen_actvn = get_activations(noisy_image_path, generated_image_path, device='cuda')

# save activations to disk
# torch.save(real_actvn, 'real_actvn.pt')
# torch.save(gen_actvn, 'gen_actvn.pt')
# torch.save(noisy_actvn, 'noisy_actvn.pt')

# # breakpoint()                                                                                                                                        
# real_actvn = torch.load('real_actvn.pt', weights_only= False)    
# gen_actvn = torch.load('gen_actvn.pt', weights_only= False) 
# noisy_actvn = torch.load('noisy_actvn.pt', weights_only= False)

# real_actvn = torch.cat(real_actvn, dim=0)
# gen_actvn = torch.cat(gen_actvn, dim=0)
# noisy_actvn = torch.cat(noisy_actvn, dim=0)    

# for i in range(1,15):
#     mu_r, cov_r = get_classwise_statistics(real_actvn, gen_actvn, i, 'real')
#     mu_g, cov_g = get_classwise_statistics(real_actvn, gen_actvn, i, 'generated')
#     tv_distance = classwise_fisher_tv(mu_r, cov_r, mu_g, cov_g)
#     print(f"Class {i}: {tv_distance}")


# get norm of differnece between the covariance matrices
# cov_diff = torch.norm(sigma1 - sigma2, p='fro')
# print('Norm of difference between covariance matrices:', cov_diff)

# compute_distance = compute_tv_distance(mu1, sigma1, mu2, sigma2)
# for c in [0.25, .24, 0.23, 0.22, 0.21, 0.2, 0.15, 0.1, 0.08, 0.05]:

    # print(f"TV distance for c = {c}:")

    # compute_distance = exp_tv_distance(gen_actvn, gen_actvn, c = c)
    # print(f'between same dist.: {compute_distance}')

    # compute_distance = simple_tv_distance(real_actvn, gen_actvn, c = c)
    # print(f'real and generated: {compute_distance}')

    # compute_distance = simple_tv_distance(noisy_actvn, real_actvn, c = c)
    # print(f'real and noisy    : {compute_distance}')
    # print(' ----------------- ')

# print(f"Fisher - TV distance")
# distance = fisher_tv_distance(real_actvn, gen_actvn)
# print(f'fisher distance between real and generated: {distance}')
# print(' ----------------- ')

# distance = fisher_tv_distance(noisy_actvn, real_actvn)
# print(f'fisher distance between real and noisy: {distance}')
# print(' ----------------- ')

# distance = fisher_tv_distance(noisy_actvn, gen_actvn)
# print(f'fisher distance between generated and noisy: {distance}')
# print(' ----------------- ')



