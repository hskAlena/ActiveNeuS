import torch
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib.cm import inferno
from nerfstudio.utils import colormaps
from nerfstudio.model_components.losses import (
    compute_scale_and_shift
)


# original code from https://github.com/poetrywanderer/CF-NeRF/blob/66918a9748c137e1c0242c12be7aa6efa39ece06/run_nerf_helpers.py#L382

def ause(unc_vec, err_vec, err_type='rmse'):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    err_vec_sorted, _ = torch.sort(err_vec)
    import pdb
    # pdb.set_trace()
    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        if err_type == 'rmse':
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae' or err_type == 'mse':
            ause_err.append(err_slice.mean().cpu().numpy())
       

    ###########################################
    # pdb.set_trace()
    # Sort by variance
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []
    for r in ratio_removed:
        err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        if err_type == 'rmse':
            ause_err_by_var.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae'or err_type == 'mse':
            ause_err_by_var.append(err_slice.mean().cpu().numpy())
    
    #Normalize and append
    max_val = max(max(ause_err), max(ause_err_by_var))
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)
    
    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)
    # pdb.set_trace()
    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed)
    return ratio_removed, ause_err, ause_err_by_var, ause

def plot_errors(ratio_removed, ause_err, ause_err_by_var, err_type, scene_no, output_path): #AUSE plots, with oracle curve also visible
    plt.plot(ratio_removed, ause_err, '--')
    plt.plot(ratio_removed, ause_err_by_var, '-r')
    # plt.plot(ratio_removed, ause_err_by_var - ause_err, '-g') # uncomment for getting plots similar to the paper, without visible oracle curve
    path = output_path.parent / Path("plots") 
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path/ Path('plot_'+err_type+'_'+str(scene_no)+'.png'))
    plt.figure()

def visualize_ranks(unc ,gt, colormap='jet'):
    flattened_unc = unc.flatten()
    flattened_gt = gt.flatten()
    # Find the ranks of the pixel values
    # ranks_unc = np.argsort(np.argsort(flattened_unc)) 
    # ranks_gt = np.argsort(np.argsort(flattened_gt)) 
    ranks_unc = rankdata(flattened_unc, method='min')-1 
    ranks_gt = rankdata(flattened_gt, method='min')-1
    
    max_rank = max(np.max(ranks_unc),np.max(ranks_gt))
    
    cmap = plt.get_cmap(colormap, max_rank)
    # cmap = plt.get_cmap(colormap)
    # Normalize the ranks to the range [0, 1]
    normalized_ranks_unc = ranks_unc / max_rank
    normalized_ranks_gt = ranks_gt / max_rank

    # normalized_ranks_unc = normalized_ranks_unc.astype(int)
    # normalized_ranks_gt = normalized_ranks_gt.astype(int)
    # Apply the colormap to the normalized ranks
    colored_ranks_unc = cmap(normalized_ranks_unc)
    colored_ranks_gt = cmap(normalized_ranks_gt)

    colored_unc = colored_ranks_unc.reshape((*unc.shape,4))
    colored_gt = colored_ranks_gt.reshape((*gt.shape,4))
    
    return colored_unc, colored_gt

def visualize_rank_unc(unc, colormap='jet'):
    flattened_unc = unc.flatten()
    ranks_unc = rankdata(flattened_unc, method='min')-1 
    max_rank = np.max(ranks_unc)
    
    cmap = plt.get_cmap(colormap, max_rank)
    # Normalize the ranks to the range [0, 1]
    normalized_ranks_unc = ranks_unc / max_rank
    # Apply the colormap to the normalized ranks
    colored_ranks_unc = cmap(normalized_ranks_unc)
    colored_unc = colored_ranks_unc.reshape((*unc.shape,4))

    return colored_unc

def get_image_metrics_unc(self, no:int, outputs, batch, dataset_path, cameras): # -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """ From https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/nerfacto.py#L357 """

    if "fg_mask" in batch:
        fg_label = batch["fg_mask"].float().to(self.device)

    unc = outputs["uncertainty"]
    images_dict = {}
    metrics_dict = {} 
    depth = outputs["depth"]

    depth_gt_dir = str(dataset_path) + '/{:06d}_depth.npy'.format(no)
    depth_gt = np.load(depth_gt_dir)
    depth_gt = torch.tensor(depth_gt, device=depth.device)
    # pdb.set_trace()
    scale, shift = compute_scale_and_shift(
        depth[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
    )
    depth = depth * scale + shift
    # depth = depth.squeeze(-1)
    depth_gt = depth_gt.unsqueeze(-1)

    depth = depth/depth_gt.max()
    depth_gt = depth_gt/depth_gt.max()

    # if "fg_mask" in batch:
    #     squared_error = ((depth_gt - depth) ** 2)*(fg_label.squeeze(-1))
    #     absolute_error = (abs(depth_gt - depth))*(fg_label.squeeze(-1))
    #     unc = unc *fg_label
    # else:
    squared_error = ((depth_gt - depth) ** 2)
    absolute_error = (abs(depth_gt - depth))
    
    unc_flat = unc.flatten()
    absolute_error_flat = absolute_error.flatten()
    squared_error_flat = squared_error.flatten()

    ratio, err_mse, err_var_mse, ause_mse = ause(unc_flat, squared_error_flat, err_type='mse')
    # plot_errors(ratio, err_mse, err_var_mse, 'mse', no, self.output_path)
    metrics_dict["mse_ratio"] = ratio
    ratio, err_mae, err_var_mae, ause_mae =  ause(unc_flat, absolute_error_flat, err_type='mae')
    # plot_errors(ratio, err_mae, err_var_mae, 'mae', no, self.output_path)
    ratio, err_rmse, err_var_rmse, ause_rmse =  ause(unc_flat, squared_error_flat, err_type='rmse')
    # plot_errors(ratio, err_rmse, err_var_rmse, 'rmse', no, self.output_path)

    #for visualizaiton
    depth_img = torch.clip(depth, min=0., max=1.)
    absolute_error_img = torch.clip(absolute_error, min=0., max=1.)

    images_dict['depth_gt'] = depth_gt # Image.fromarray((depth_gt.cpu().numpy()* 255).astype('uint8'))
    # im.save(path / Path(str(no)+"_depth_gt.jpeg"))
    images_dict['depth'] = depth_img #Image.fromarray((depth_img.cpu().numpy()* 255).astype('uint8'))
    images_dict['abs_error_img'] = absolute_error_img
    uu, errr = visualize_ranks(unc.squeeze(-1).cpu().numpy(), absolute_error.squeeze(-1).cpu().numpy())
    images_dict['unc_colored'] = torch.from_numpy(uu[...,:-1]) #Image.fromarray(np.uint8(uu * 255))
    # im.save(path / Path(str(no)+"_unc_colored.png"))

    # errr = torch.from_numpy(errr[:, ::-1, :].copy()).to(self.device)
    images_dict['error_colored'] = torch.from_numpy(errr[...,:-1])

    # all of these metrics will be logged as scalars
    metrics_dict["ause_mse"] = ause_mse
    metrics_dict["ause_mae"] = ause_mae
    metrics_dict["ause_rmse"] = ause_rmse
    metrics_dict["mse"] = float(squared_error.mean().item())
    metrics_dict["err_mse"] = err_mse
    metrics_dict["err_mae"] = err_mae
    metrics_dict["err_rmse"] = err_rmse
    metrics_dict["err_var_mse"] = err_var_mse
    metrics_dict["err_var_mae"] = err_var_mae
    metrics_dict["err_var_rmse"] = err_var_rmse

    return metrics_dict, images_dict