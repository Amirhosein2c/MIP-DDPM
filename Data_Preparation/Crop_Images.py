import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from Utilities import *

from Data_Preparation.config import Configuration, get_config

CONFIG: Configuration = get_config()

def crop_MIPs():
    
    Data_root = CONFIG.data_dir
    Target_root = CONFIG.temp_data_dir

    all_dirs = os.listdir(Data_root)

    for sample_dir in all_dirs:
        # PSMA-01-001-000
        # PSMA-01-001_PET_000.nii.gz
        name_parts = sample_dir.split("-")
        
        print(sample_dir)
        
        PET_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET_{name_parts[3]}.nii.gz"
        PET5_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET5_{name_parts[3]}.nii.gz"
        PET10_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET10_{name_parts[3]}.nii.gz"
        PETD_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PETD_{name_parts[3]}.nii.gz"
        MASK_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_SEG_{name_parts[3]}.nii.gz"
        
        PET_filename = os.path.join(os.path.join(Data_root, sample_dir),PET_filename)
        PET5_filename = os.path.join(os.path.join(Data_root, sample_dir),PET5_filename)
        PET10_filename = os.path.join(os.path.join(Data_root, sample_dir),PET10_filename)
        PETD_filename = os.path.join(os.path.join(Data_root, sample_dir),PETD_filename)
        MASK_filename = os.path.join(os.path.join(Data_root, sample_dir),MASK_filename)
        
        PET = read_nii_vol(PET_filename)
        PET5 = read_nii_vol(PET5_filename)
        PET10 = read_nii_vol(PET10_filename)
        PETD = read_nii_vol(PETD_filename)
        MASK = read_nii_vol(MASK_filename)
        
        PET = np.float64(PET)
        PET5 = np.float64(PET5)
        PET10 = np.float64(PET10)
        PETD = np.float64(PETD)
        MASK = np.uint8(MASK)
        
        if MASK.shape[1] > 500:
            top_start = -250
            top_end = 0
            bottom_start = -500
            bottom_end = -250
            
            PET_top = PET[:,top_start:]
            PET5_top = PET5[:,top_start:]
            PET10_top = PET10[:,top_start:]
            PETD_top = PETD[:,top_start:]
            MASK_top = MASK[:,top_start:]
            
            target_folder = os.path.join(Target_root, f"{sample_dir}-01")
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                
            PET_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(PET_top, target_folder, PET_top_filename)
            
            PET5_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET5_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(PET5_top, target_folder, PET5_top_filename)
            
            PET10_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET10_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(PET10_top, target_folder, PET10_top_filename)
            
            PETD_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PETD_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(PETD_top, target_folder, PETD_top_filename)
            
            MASK_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_SEG_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(MASK_top, target_folder, MASK_top_filename)
            
            PET_bottom = PET[:,bottom_start:bottom_end]
            PET5_bottom = PET5[:,bottom_start:bottom_end]
            PET10_bottom = PET10[:,bottom_start:bottom_end]
            PETD_bottom = PETD[:,bottom_start:bottom_end]
            MASK_bottom = MASK[:,bottom_start:bottom_end]
            
            target_folder = os.path.join(Target_root, f"{sample_dir}-02")
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                
            PET_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(PET_bottom, target_folder, PET_bottom_filename)
            
            PET5_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET5_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(PET5_bottom, target_folder, PET5_bottom_filename)
            
            PET10_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET10_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(PET10_bottom, target_folder, PET10_bottom_filename)
            
            PETD_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PETD_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(PETD_bottom, target_folder, PETD_bottom_filename)
            
            MASK_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_SEG_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(MASK_bottom, target_folder, MASK_bottom_filename)
            
        else:
            top_start = -250
            top_end = 0
            bottom_start = 0
            bottom_end = 250
            
            PET_top = PET[:,top_start:]
            PET5_top = PET5[:,top_start:]
            PET10_top = PET10[:,top_start:]
            PETD_top = PETD[:,top_start:]
            MASK_top = MASK[:,top_start:]
            
            target_folder = os.path.join(Target_root, f"{sample_dir}-01")
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                
            PET_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(PET_top, target_folder, PET_top_filename)
            
            PET5_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET5_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(PET5_top, target_folder, PET5_top_filename)
            
            PET10_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET10_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(PET10_top, target_folder, PET10_top_filename)
            
            PETD_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PETD_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(PETD_top, target_folder, PETD_top_filename)
            
            MASK_top_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_SEG_{name_parts[3]}-01.nii.gz"
            save_as_Nifti(MASK_top, target_folder, MASK_top_filename)
            
            PET_bottom = PET[:,bottom_start:bottom_end]
            PET5_bottom = PET5[:,bottom_start:bottom_end]
            PET10_bottom = PET10[:,bottom_start:bottom_end]
            PETD_bottom = PETD[:,bottom_start:bottom_end]
            MASK_bottom = MASK[:,bottom_start:bottom_end]
            
            target_folder = os.path.join(Target_root, f"{sample_dir}-02")
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                
            PET_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(PET_bottom, target_folder, PET_bottom_filename)
            
            PET5_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET5_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(PET5_bottom, target_folder, PET5_bottom_filename)
            
            PET10_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PET10_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(PET10_bottom, target_folder, PET10_bottom_filename)
            
            PETD_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_PETD_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(PETD_bottom, target_folder, PETD_bottom_filename)
            
            MASK_bottom_filename = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}_SEG_{name_parts[3]}-02.nii.gz"
            save_as_Nifti(MASK_bottom, target_folder, MASK_bottom_filename)
        
        # break


def main():
    crop_MIPs()

if __name__ == "__main__":
    main()
    