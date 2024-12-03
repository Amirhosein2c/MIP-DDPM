
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
from Utilities import *

from Data_Preparation.config import Configuration, get_config

CONFIG: Configuration = get_config()


def NIFTI_Vol_to_MIP_for_Seg():
    
    volumes_path = CONFIG.nifti_vols_dir
    save_path = CONFIG.temp_data_dir
    PET_path = 'PET'
    CT_path = "CT"
    MASK_path = 'MASK'
    fileformat = '*.nii.gz'
    
    PET_Dir = os.path.join(volumes_path, PET_path)
    CT_Dir = os.path.join(volumes_path, CT_path)
    MASK_Dir = os.path.join(volumes_path, MASK_path)
    
    PET_directory = os.path.join(PET_Dir, fileformat)
    
    files_list = glob.glob(PET_directory)
    for file in files_list:
        patient_id = os.path.splitext(os.path.basename(file))[0][0:-9]
        print(patient_id)
        PET_filename = os.path.join(PET_Dir, patient_id + "__PET.nii.gz")
        CT_filename = os.path.join(CT_Dir, patient_id + "__CT.nii.gz")
        MASK_filename = os.path.join(MASK_Dir, patient_id + "__MASK.nii.gz")
        
        PET = read_nii_vol(PET_filename)
        PET = PET.astype(np.float64)
        
        PET[PET <= 0.1] = 0
        
        CT = read_nii_vol(CT_filename)
        CT = CT.astype(np.float64)
        
        MASK = read_nii_vol(MASK_filename)
        MASK = MASK.astype(np.uint8)
        
        
        for Rotation in tqdm(range(0, 360, 5)):
            
            Rotated_PET = rotate_volume(PET, Rotation, is_Mask=False)
            Rotated_MASK = rotate_volume(MASK, Rotation, is_Mask=True)
            
            Rotated_PET[Rotated_PET <= 0.05] = 0
            PET_DEPTH = np.argmax(Rotated_PET, axis=1)
            PET_MIP = np.max(Rotated_PET, axis=1)
            PET_MIP_5 = PET_MIP.copy()
            PET_MIP_10 = PET_MIP.copy()
            PET_MIP_5[PET_MIP > 5] = 5
            PET_MIP_10[PET_MIP > 10] = 10
            MASK_MIP = np.max(Rotated_MASK, axis=1)

            show_MIPS(PET_MIP_5, PET_MIP_10, PET_MIP, PET_DEPTH, MASK_MIP)

            save_rotation_dir = os.path.join(save_path, f"{patient_id}-{str(Rotation).zfill(3)}")
            if not os.path.exists(save_rotation_dir):
                os.makedirs(save_rotation_dir)
            
            # saving format --> patientID_MODALITY_Rotation.nii.gz
            
            PET_MIP_filename = f"{patient_id}_PET_{str(Rotation).zfill(3)}"
            save_as_Nifti(PET_MIP, save_rotation_dir, PET_MIP_filename)
            
            PET_MIP_5_filename = f"{patient_id}_PET5_{str(Rotation).zfill(3)}"
            save_as_Nifti(PET_MIP_5, save_rotation_dir, PET_MIP_5_filename)
            
            PET_MIP_10_filename = f"{patient_id}_PET10_{str(Rotation).zfill(3)}"
            save_as_Nifti(PET_MIP_10, save_rotation_dir, PET_MIP_10_filename)
            
            PET_DEPTH_filename = f"{patient_id}_PETD_{str(Rotation).zfill(3)}"
            save_as_Nifti(PET_DEPTH, save_rotation_dir, PET_DEPTH_filename)
            
            MASK_MIP_filename = f"{patient_id}_SEG_{str(Rotation).zfill(3)}"
            save_as_Nifti(MASK_MIP, save_rotation_dir, MASK_MIP_filename)


def main():
    NIFTI_Vol_to_MIP_for_Seg()

if __name__ == "__main__":
    main()
    
    