import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.ensemble import IsolationForest



n_components = 3
grid_size = 10
contamination = 0.01

real_actvn = torch.load('../embeddings/nih/real_med_clip_embeddings.pt', weights_only= False)    
gen_actvn = torch.load('../embeddings/nih/fake_med_clip_embeddings.pt', weights_only= False) 
noisy_actvn = torch.load('../embeddings/nih/noise_med_clip_embeddings.pt', weights_only= False)
wh_actvn = torch.load('../embeddings/nih/wh_real_med_clip_embeddings.pt', weights_only= False)


real_actvn = real_actvn.numpy() if isinstance(real_actvn, torch.Tensor) else real_actvn
gen_actvn = gen_actvn.numpy() if isinstance(gen_actvn, torch.Tensor) else gen_actvn
wh_actvn = wh_actvn.numpy() if isinstance(wh_actvn, torch.Tensor) else wh_actvn
noisy_actvn = noisy_actvn.numpy() if isinstance(noisy_actvn, torch.Tensor) else noisy_actvn

def grid_tv(n_components, grid_size, contamination = 0.01, real_actvn = real_actvn, gen_actvn = gen_actvn , noisy_actvn= noisy_actvn, wh_actvn = wh_actvn):
    
    # do a PCA on the real and generated data
    pca = PCA(n_components=n_components)
    real_pca = pca.fit_transform(real_actvn)
    gen_pca = pca.transform(gen_actvn)
    noisy_actvn = pca.transform(noisy_actvn)



    clf = IsolationForest(contamination=contamination)
    clf.fit(real_pca)
    real_pca = real_pca[clf.predict(real_pca) == 1]

    clf.fit(gen_pca)
    gen_pca = gen_pca[clf.predict(gen_pca) == 1]

    # clf.fit(wh_pca)
    # wh_pca = wh_pca[clf.predict(wh_pca) == 1]

    clf.fit(noisy_pca)
    noisy_pca = noisy_pca[clf.predict(noisy_pca) == 1]
    # print(real_pca.shape, gen_pca.shape, wh_pca.shape, noisy_pca.shape)

    # along each dim min val 
    real_min, real_max = np.min(real_pca, axis=0), np.max(real_pca, axis=0)
    gen_min, gen_max = np.min(gen_pca, axis=0), np.max(gen_pca, axis=0)
    # wh_min, wh_max = np.min(wh_pca, axis=0), np.max(wh_pca, axis=0)
    noisy_min, noisy_max = np.min(noisy_pca, axis=0), np.max(noisy_pca, axis=0)

    min_val = np.min([real_min, gen_min, noisy_min], axis=0) - 0.05
    max_val = np.max([real_max, gen_max, noisy_max], axis=0) + 0.05

    # print(min_val, max_val)


    def num_points_in_grid(dataset, min_vals, max_vals, n_components ,grid_size):
        # Create a dictionary for counting points in each grid cell
        tuples = list(itertools.product(range(grid_size), repeat = n_components))
        point_grid_dict = {key: 0 for key in tuples}
        
        for point in dataset:
            grid_point = tuple(int((point[i].item() - min_vals[i].item()) / (max_vals[i].item() - min_vals[i].item()) * grid_size) for i in range(n_components))
            point_grid_dict[grid_point] += 1
        point_grid_dict = {key: value/len(dataset) for key, value in point_grid_dict.items()} 
        return point_grid_dict



    real_grid = num_points_in_grid(real_pca, min_val, max_val, n_components, grid_size)
    gen_grid = num_points_in_grid(gen_pca, min_val, max_val, n_components, grid_size)
    noisy_grid = num_points_in_grid(noisy_pca, min_val, max_val, n_components, grid_size)    
    # wh_grid = num_points_in_grid(wh_pca, min_val, max_val, n_components, grid_size)  



    tv_real_gen = 0
    for key in real_grid.keys():
        tv_real_gen += abs(real_grid[key] - gen_grid[key])
    tv_real_gen = tv_real_gen / 2

    tv_real_noisy = 0
    for key in real_grid.keys():
        tv_real_noisy += abs(real_grid[key] - noisy_grid[key])
    tv_real_noisy = tv_real_noisy / 2

    return tv_real_gen, tv_real_noisy
