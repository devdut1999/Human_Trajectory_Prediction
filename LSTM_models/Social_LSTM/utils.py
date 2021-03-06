import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def en_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def diff_axis_0(a):
    ret = a[1:] - a[:-1]
    ret = torch.cat([torch.zeros(1, a.size()[1], a.size()[2]).cuda(), ret])
    return ret


def compute_speeds(a):
    ret = a[1:] - a[:-1]
    ret = torch.cat([en_cuda(torch.zeros(1, a.size()[1])), ret])
    return ret


def get_coef(mux, muy, sx, sy, corr):
    # eq 20 -> 22 of Graves (2013)

    # The output must be exponentiated for the std devs
    o_sx = torch.exp(sx)
    o_sy = torch.exp(sy)
    # Tanh applied to keep it in the range [-1, 1]
    o_corr = torch.tanh(corr)
    return [mux, muy, o_sx, o_sy, o_corr]


def import_data(scene, batch_size, factor=10, rotate_=True):
    '''Import Data file for training'''
    # Read data and group by frame
    selector = en_cuda(torch.LongTensor([2, 3]))
    df_positions = pd.read_table(scene, delim_whitespace=True, header=None)
    df_positions.loc[:, [2, 3]] = df_positions.loc[:, [2, 3]] * factor
    df_positions_by_frame = []

    map_frame_index = {}
    index = 0
    for _, x in df_positions.groupby(df_positions[0]):
        df_positions_by_frame.append(en_cuda(torch.Tensor(x.as_matrix())))
        map_frame_index[x.iloc[0, 0]] = index
        index += 1

    df = []

    map_ped_index = {}

    first_positions = []
    # Group by ped_id
    index = 0
    for _, x in df_positions.groupby(df_positions[1]):
        first_positions.append(x.iloc[8, :])
        map_ped_index[x.iloc[0, 1]] = (
            int(index / batch_size), index % batch_size)
        index += 1
        tmp = en_cuda(torch.Tensor(x.as_matrix()))
        df.append(tmp.unsqueeze(0))

    # Get_first positions so that we can recover the position from the velocity
    first_positions = en_cuda(torch.Tensor(
        pd.concat(first_positions, axis=1).as_matrix().T)).split(batch_size, 0)

    # creates Batches
    tmp = torch.cat(df, 0).transpose(1, 0)
    count = tmp.size()[1]

    # Apply rotations
    rotations = None
    result_speeds = tmp.clone()
    result_speeds[:, :, selector] = diff_axis_0(result_speeds[:, :, selector])
    if rotate_:
        rotations = create_rotation_matrices(torch.atan2(
            result_speeds[1, :, 3], result_speeds[1, :, 2]))
        result_speeds[:, :, selector] = rotate(
            rotations, result_speeds[:, :, selector])
        for i in range(rotations.size()[0]):
            rotations[i, :, :] = torch.mul(
                rotations[i, :, :], en_cuda(torch.Tensor([[1, -1], [-1, 1]])))
    result_speeds = Variable(result_speeds)
    result_pts = tmp.split(batch_size, 1)
    result_speeds = result_speeds.split(batch_size, 1)
    rotations = rotations.split(batch_size, 0) if rotate_ else None
    return map_ped_index, result_speeds, rotations, result_pts, first_positions, df_positions_by_frame, map_frame_index, count


def sample_gaussian_2d(mux, muy, sx, sy, rho):
    '''
    Function to sample a point from a given 2D normal distribution
    params:
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''
    # Extract mean
    mean = [mux, muy]
    # print("Mean:" , mean)
    # Extract covariance matrix
    cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    # print("Covariance" , cov)
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, size = 1)
    # print(x)
    return x[0][0], x[0][1]


def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
    # PDF normal
    normx = x.sub(mux)
    normy = y.sub(muy)
    # Calculate sx*sy
    sxsy = torch.mul(sx, sy)
    # Calculate the exponential factor
    z = (torch.div(normx, sx))**2 + (torch.div(normy, sy))**2 - 2 * \
        torch.div(torch.mul(rho, torch.mul(normx, normy)), sxsy)
    negRho = 1 - rho**2
    # exp part
    result = torch.exp(torch.div(-z, 2 * negRho))
    # Normalization constant
    denom = 2 * np.pi * torch.mul(sxsy, torch.sqrt(negRho))
    # Final PDF calculation
    result = torch.div(result, denom)

    return result


def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
    '''
    Function to calculate given a 2D distribution over x and y, and target data
    of observed x and y points
    params:
    z_mux : mean of the distribution in x
    z_muy : mean of the distribution in y
    z_sx : std dev of the distribution in x
    z_sy : std dev of the distribution in y
    z_rho : Correlation factor of the distribution
    x_data : target x points
    y_data : target y points
    '''
    # Calculate the PDF of the data w.r.t to the distribution
    result = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    epsilon = 1e-20

    # Apply the log operation
    result1 = -torch.log(torch.clamp(result, min=epsilon)
                         )  # Numerical stability
    print(result1.size())
    return torch.sum(result1).div(result1.size()[0])


def get_avg_displacement(true, pred):
    # Compute MSE
    result = pred.sub(true)**2
    result = torch.sum(result, 2)
    norm = result.size()[0]
    result = torch.sum(result, 0)
    result = torch.div(result, norm)
    return result


def get_accuracy(true, pred):
    # Compute L2 distance and Final displacement error
    result = torch.sqrt(pred.sub(true)**2)
    result = torch.sum(result, 2)
    norm = result.size()[0]
    result_ = torch.sum(result, 0)
    result_ = torch.div(result_, norm)
    return result_, result[-1, :]


def plot_trajectory(true, pred, neighbs, n_paths, name, obs=8, xlim=None, ylim=None, debug=False):
    plt.figure()
    # Plot predicted and true trajectories
    true_ = true.cpu().numpy()
    x1, y1, x2, y2 = pred[:, 0].cpu().numpy(), pred[:, 1].cpu().numpy(), true_[
        :, 2], true_[:, 3]
    plt.plot(x1, y1, 'ro-', x2, y2, 'go-')
    plt.plot([x2[7], x1[0]], [y2[7], y1[0]], 'ro-')
    red_patch = mpatches.Patch(color='red', label='predicted trajectory')
    # Add timestamps
    for i in range(x1.shape[0]):
        plt.text(x1[i], y1[i], str(i))
        plt.text(x2[i + 8], y2[i + 8], str(i))

    map_ped_index = {}
    min_frame_all, step = true[0, 0], (true[1, 0] - true[0, 0])
    if step == 0:
        step = 1
    # Group neighbors by ped_id
    pedestrians = []
    idx = 0
    for frame in neighbs:
        for entry in frame:
            if entry[1] in map_ped_index:
                pedestrians[map_ped_index[entry[1]]].append(
                    np.expand_dims(entry.cpu().numpy(), axis=0))
            else:
                map_ped_index[entry[1]] = idx
                pedestrians.append([])
                pedestrians[idx].append(
                    np.expand_dims(entry.cpu().numpy(), axis=0))
                idx += 1

    if len(pedestrians):
        pedestrians = [np.concatenate(x, axis=0) for x in pedestrians]
        # Find closest neighbors
        dists = []
        for x in pedestrians:
            min_frame = np.min(x[:, 0])
            max_frame = np.max(x[:, 0])
            sub_traj = true_[
                (true_[:, 0] >= min_frame) & (true_[:, 0] <= max_frame), :]
            dist = np.mean(
                np.sum((sub_traj[:, [2, 3]] - x[:, [2, 3]])**2, 1))
            dists.append(dist)

        idxs = np.argsort(np.array(dists))[:n_paths]
        pedestrians = [pedestrians[i] for i in idxs]
        # print(pedestrians[0])
        # Plot neighbors
        for path in pedestrians:
            plt.plot(path[:, 2], path[:, 3], 'bo-')
            plt.text(path[0, 2], path[0, 3], int(
                (path[0, 0] - min_frame_all) / step))
            plt.text(path[-1, 2], path[-1, 3],
                     int((path[-1, 0] - min_frame_all) / step))

    green_patch = mpatches.Patch(color='green', label='true trajectory')
    blue_patch = mpatches.Patch(color='blue', label='neighboring trajectories')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(name + '.png')

    plt.close()
    return
