from dummy_threading import local
import numpy as np
import random
from builtins import range
import torch
from typing import Tuple, Union, Callable
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage import map_coordinates, fourier_gaussian

def augment_rot90(sample_data, sample_seg=None, num_rot=(1, 2, 3), axes=(2, 3)):
    """
    :param sample_data:
    :param sample_seg:
    :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
    :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
    :return:
    """
    num_rot = np.random.choice(num_rot)
    # axes = np.random.choice(axes, size=2, replace=False)
    # axes.sort()
    # axes = [i + 1 for i in axes]
    sample_data = np.rot90(sample_data, num_rot, axes)
    if sample_seg is not None:
        sample_seg = np.rot90(sample_seg, num_rot, axes)
    return sample_data, num_rot

def augment_rot90_reverse(sample_data, num_rot=1, axes=(2, 3)):
    """
    :param sample_data:
    :param sample_seg:
    :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
    :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
    :return:
    """
    sample_data = np.rot90(sample_data, -num_rot, axes)

    return sample_data

def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data

def augment_transpose_axes(data_sample, seg_sample=None, axes=(0, 1)):
    """
    :param data_sample: c,x,y(,z)
    :param seg_sample: c,x,y(,z)
    :param axes: list/tuple
    :return:
    """
    axes = list(np.array(axes) + 1)  # need list to allow shuffle; +1 to accomodate for color channel

    assert np.max(axes) <= len(data_sample.shape), "axes must only contain valid axis ids"
    static_axes = list(range(len(data_sample.shape)))
    for i in axes: static_axes[i] = -1
    np.random.shuffle(axes)

    ctr = 0
    for j, i in enumerate(static_axes):
        if i == -1:
            static_axes[j] = axes[ctr]
            ctr += 1

    data_sample = data_sample.transpose(*static_axes)
    if seg_sample is not None:
        seg_sample = seg_sample.transpose(*static_axes)
    return data_sample

def extract_amp_spectrum(trg_img):

    fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target

def amp_spectrum_swap(amp_local, amp_target, L=0.1 , ratio=0):
    
    a_local = np.fft.fftshift( amp_local, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_target, axes=(-2, -1) )

    _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )
    return a_local

def freq_space_interpolation( local_img, amp_target, L=0 , ratio=0):
    
    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = np.fft.fft2( local_img_np, axes=(-2, -1) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)

    return local_in_trg

def fourier_interpolation(im_local, im_target):
    L = 0.003
    ratio = random.uniform(0.5,1)
    amp_target = extract_amp_spectrum(im_target)
    local_in_trg = freq_space_interpolation(im_local, amp_target, L=L, ratio=ratio)
    local_in_trg = torch.from_numpy(local_in_trg).float()
    # local_in_trg = np.clip(local_in_trg / 255, 0, 1)

    return local_in_trg

def fourier_interpolation_amp(im_local, amp_target):
    L = 0.003
    ratio = random.uniform(0.5,1)

    local_in_trg = freq_space_interpolation(im_local, amp_target, L=L, ratio=ratio)
    local_in_trg = torch.from_numpy(local_in_trg).float()
    # local_in_trg = np.clip(local_in_trg / 255, 0, 1)

    return local_in_trg

def fourier_aug_amp(data, amp_local2):
    p = random.uniform(0.7,1)

    img_local = np.asarray(data, np.float32)
    fft_local_np = np.fft.fft2( img_local, axes=(-2, -1)) 
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    amp_mixup = p*amp_local + (1-p) * amp_local2
    
    fft_local_ = amp_mixup * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)
    local_in_trg = torch.from_numpy(local_in_trg).float()

    return local_in_trg


def fourier_aug(data, data2):
    p = random.uniform(0.8, 1)
    # p = 0.9

    img_local = np.asarray(data, np.float32)
    fft_local_np = np.fft.fft2( img_local, axes=(-2, -1)) 
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    img_local2 = np.asarray(data2, np.float32)
    fft_local_np2 = np.fft.fft2( img_local2, axes=(-2, -1)) 
    amp_local2, pha_local2 = np.abs(fft_local_np2), np.angle(fft_local_np2)

    amp_mixup = p*amp_local + (1-p) * amp_local2
    
    fft_local_ = amp_mixup * np.exp( 1j * pha_local )
    # fft_local_ = amp_local2 * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)
    local_in_trg = torch.from_numpy(local_in_trg).float()

    return local_in_trg



def fourier_aug_tta(data, data1, data2):
    p = random.uniform(0, 1)
    # p = 0.9

    img_local = np.asarray(data, np.float32)
    fft_local_np = np.fft.fft2( img_local, axes=(-2, -1)) 
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    img_local1 = np.asarray(data1, np.float32)
    fft_local_np1 = np.fft.fft2( img_local1, axes=(-2, -1)) 
    amp_local1, pha_local1 = np.abs(fft_local_np1), np.angle(fft_local_np1)

    img_local2 = np.asarray(data2, np.float32)
    fft_local_np2 = np.fft.fft2( img_local2, axes=(-2, -1)) 
    amp_local2, pha_local2 = np.abs(fft_local_np2), np.angle(fft_local_np2)

    amp_mixup = p*amp_local1 + (1-p) * amp_local2
    
    fft_local_ = amp_mixup * np.exp( 1j * pha_local )
    # fft_local_ = amp_local2 * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)
    local_in_trg = torch.from_numpy(local_in_trg).float()

    return local_in_trg

def augment_contrast(data_sample: np.ndarray,
                     contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.95, 1.05),
                     preserve_range: bool = True,
                     per_channel: bool = True) -> np.ndarray:
    if not per_channel:
        if callable(contrast_range):
            factor = contrast_range()
        else:
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()

            data_sample[c] = (data_sample[c] - mn) * factor + mn

            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            if callable(contrast_range):
                factor = contrast_range()
            else:
                if np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()

            data_sample[c] = (data_sample[c] - mn) * factor + mn

            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
                
    return data_sample


def augment_brightness_additive(data_sample, mu=0, sigma=0.5 , per_channel:bool=True):
    """
    data_sample must have shape (c, x, y(, z)))
    :param data_sample: 
    :param mu: 
    :param sigma: 
    :param per_channel: 
    :param p_per_channel: 
    :return: 
    """
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            rnd_nb = np.random.normal(mu, sigma)
            data_sample[c] += rnd_nb
    return data_sample


def augment_gamma(data_sample, gamma_range=(0.9, 1), invert_image=True, epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = True):
    if invert_image:
        data_sample = - data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            # print(retain_stats_here)
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = - data_sample
    return data_sample

def augment_gaussian_noise(data_sample: np.ndarray, noise_variance: Tuple[float, float] = (0, 0.5)) -> np.ndarray:

    variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
        random.uniform(noise_variance[0], noise_variance[1])
    for c in range(data_sample.shape[0]):
        variance_here = variance if variance is not None else \
            noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                random.uniform(noise_variance[0], noise_variance[1])
        data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
    return data_sample

        