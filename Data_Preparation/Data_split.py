

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
import random
import shutil
from Data_Preparation.config import Configuration, get_config


CONFIG: Configuration = get_config()

Data_root = CONFIG.data_dir
Training_target_root = os.path.join(CONFIG.data_dir, 'training')
Testing_target_root = os.path.join(CONFIG.data_dir, 'testing')

def main():
    
    if not os.path.exists(Training_target_root):
        os.mkdir(Training_target_root)
    if not os.path.exists(Testing_target_root):
        os.mkdir(Testing_target_root)
    
    directories = glob.glob(f"{Data_root}/*-000-01")
    patient_names = [directory[-18:-7] for directory in directories]
    
    random.seed(1111)
    random.shuffle(patient_names)
    
    for idx in tqdm(range(len(patient_names))):
        patient_directories = glob.glob(os.path.join(Data_root, f"{patient_names[idx]}*"))
        if idx <= int(len(patient_names) * 0.85):
            for patient_directory in patient_directories:
                shutil.move(patient_directory, Training_target_root)
        else:
            for patient_directory in patient_directories:
                shutil.move(patient_directory, Testing_target_root)
                

if __name__ == "__main__":
    main()