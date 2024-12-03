
import os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage, stats
from skimage.transform import rotate
import nibabel as nib
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt



def read_nii_vol(target_filename):
    img = nib.load(target_filename)
    VOL = np.array(img.dataobj)
    return VOL


def rotate_volume(VOL, DEG, is_Mask):
    if is_Mask:
        vol_rot = ndimage.rotate(VOL, DEG, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0.0, prefilter=True)
    else:
        vol_rot = ndimage.rotate(VOL, DEG, axes=(0, 1), reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
    return vol_rot


def save_as_Nifti(VOL, save_Dir, filename):
    VOL_32f = np.array(VOL, dtype=np.float32)
    affine = np.eye(4)
    VOL_nifti_file = nib.Nifti1Image(VOL_32f, affine)
    VOL_save_file = filename + ".nii.gz"
    nib.save(VOL_nifti_file, os.path.join(save_Dir, VOL_save_file))



def show_MIPS(PET_MIP_5, PET_MIP_10, PET_MIP, PET_DEPTH, MASK_MIP):
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))

    im1 = axes[0].imshow(np.rot90(PET_MIP), cmap='gray')
    axes[0].set_title('PET_MIP')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(np.rot90(PET_MIP_10), cmap='gray')
    axes[1].set_title('PET_MIP_10')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(np.rot90(PET_MIP_5), cmap='gray')
    axes[2].set_title('PET_MIP_5')
    plt.colorbar(im3, ax=axes[2])

    im4 = axes[3].imshow(np.rot90(PET_DEPTH), cmap='gray')
    axes[3].set_title('PET_DEPTH')
    plt.colorbar(im4, ax=axes[3])

    im5 = axes[4].imshow(np.rot90(MASK_MIP), cmap='gray')
    axes[4].set_title('MASK_MIP')
    plt.colorbar(im5, ax=axes[4])

    plt.tight_layout()

    plt.show()
