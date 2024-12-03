# config.py
import argparse
import yaml
import os
from dataclasses import dataclass

@dataclass
class Configuration:
    nifti_vols_dir: str
    temp_data_dir: str
    data_dir: str
    
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

def get_config():
    parser = argparse.ArgumentParser(description='Process configuration parameters')
    
    # Allow overriding config file location
    parser.add_argument('--config', 
                       default='config.yaml',
                       help='Path to configuration (config.yaml) file')
    
    # Allow command line overrides for specific parameters
    parser.add_argument('--nifti_vols_dir', 
                       help='Override original 3D nifti vols directory from config file')
    
    parser.add_argument('--temp_data_dir', 
                       help='Override temp data directory (uncropped MIPs) from config file')
    
    parser.add_argument('--data_dir', 
                       help='Override data directory (cropped MIPs) from config file')
    
    args = parser.parse_args()
    
    
    
    config = Configuration.from_yaml(args.config)
    
    if args.temp_data_dir:
        config.temp_data_dir = args.temp_data_dir
        
    if args.data_dir:
        config.data_dir = args.data_dir
    
    os.makedirs(config.temp_data_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    return config