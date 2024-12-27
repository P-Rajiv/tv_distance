import numpy as np
import torch
real_actvn = torch.load('real_med_inception_embeddings.pt', weights_only= False)    
gen_actvn = torch.load('fake_med_inception_embeddings.pt', weights_only= False) 
noisy_actvn = torch.load('noise_med_inception_embeddings.pt', weights_only= False)
wh_actvn = torch.load('wh_med_inception_embeddings.pt', weights_only= False)

# print(real_actvn.shape, gen_actvn.shape, noisy_actvn.shape)
print(real_actvn[0])

import scipy.linalg

# frechet distance 
def frechet_dist(X_0, X_1, device = 'cuda'):
    
    epsilon = 1e-6
    X_0 = X_0/torch.max(X_0)
    X_1 = X_1/torch.max(X_1)
    

    X_0 = X_0.to(device)
    X_1 = X_1.to(device)

    mean_0 = torch.mean(X_0, dim=0)
    mean_1 = torch.mean(X_1, dim=0)


    cov_0 = torch.cov(X_0.T) + torch.eye(X_0.shape[1], device=device) * epsilon
    cov_1 = torch.cov(X_1.T) + torch.eye(X_1.shape[1], device=device) * epsilon

    diff = mean_0 - mean_1

    A = cov_0 @ cov_1
    eigvals, eigvecs = torch.linalg.eigh(A)  # eigenvalues and eigenvectors

    # Take the square root of eigenvalues (ensure no negative values due to numerical issues)
    eigvals_sqrt = torch.sqrt(torch.clamp(eigvals, min=0))

    # Reconstruct the square root matrix
    sqrtm_A = eigvecs @ torch.diag(eigvals_sqrt) @ eigvecs.T

    # sqrtm_A = scipy.linalg.sqrtm(cov_0 @ cov_1)    
    frech_dist = diff.T @ diff + torch.trace(cov_0 + cov_1 - 2*sqrtm_A)
    frech_dist = frech_dist.cpu().detach().numpy()
    return frech_dist
for i in [200, 500, 1000, 5000, 10000, 15000, 20000]:
    # randomly sample i samples from the embeddings
    shuffled_indices = torch.randperm(real_actvn.shape[0])[:i]
    real_actvn = real_actvn[shuffled_indices]
    gen_actvn = gen_actvn[shuffled_indices]
    # noisy_actvn = noisy_actvn[shuffled_indices]
    wh_actvn = wh_actvn[shuffled_indices]
    print('-'*100)
    print(f'for {i} samples')

    # print(" FID (med) :: real and noisy : ", frechet_dist(real_actvn, noisy_actvn))
    print(" FID (med) :: real and gen   : ", frechet_dist(real_actvn, gen_actvn))
    print(" FID (med) :: real and wh    : ", frechet_dist(real_actvn, wh_actvn))
    print(" FID (med) :: gen and wh     : ", frechet_dist(gen_actvn, wh_actvn))