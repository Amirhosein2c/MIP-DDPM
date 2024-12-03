"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
from visdom import Visdom
viz = Visdom(port=9040)
import sys
import random
sys.path.append("/datassd/PSMA_SEGMENTATION/Diffusion-based-Segmentation")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import PSMADataset # BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import nibabel as nib

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def read_nii_vol(target_filename):
    img = nib.load(target_filename)
    VOL = np.array(img.dataobj)
    return VOL

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist('0')
    logger.configure()

    print(args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = PSMADataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    print(f"dist_util.dev(): {dist_util.dev()}")
    
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    while len(all_images) * args.batch_size < args.num_samples:
        b, path = next(data)  #should return an image from the dataloader "data"
        # print(f"b: {b.shape} | path: {path}")
        
        folder_name = path[0].split("/")[-2]
        top_bottom = folder_name[-2:]
        rotation = folder_name[-6:-3]
        # print(f"top or bottom? : {top_bottom} | rotation degree: {rotation}")
        
        if rotation == '000' and top_bottom == '02':
            # print(f"b: {b.shape} | path: {path}")
            # print(f"top or bottom? : {top_bottom} | rotation degree: {rotation}")
            print(f" path: {path}")
            print(f"Patient ID: {folder_name[:-7]}")
            
            # '/datassd/PSMA_SEGMENTATION/DATA/testing/PSMA-01-097-000-02/PSMA-01-097_PETD_000-02.nii.gz'
            gt_filename = f"/datassd/PSMA_SEGMENTATION/DATA/testing/{folder_name[:-7]}-{rotation}-{top_bottom}/{folder_name[:-7]}_SEG_{rotation}-{top_bottom}.nii.gz"
            gt = read_nii_vol(gt_filename)
            gt = np.uint8(gt)
            
            torch_gt = th.tensor(gt)
            
            c = th.randn_like(b[:, :1, ...])
            # print(f"c: {c.shape}")
            img = th.cat((b, c), dim=1)     #add a noise channel$
            # print(f"img: {img.shape}")
            slice_ID=path[0].split("/", -1)[3]

            viz.image(visualize(th.rot90(img[0, 0, ...])), opts=dict(caption=folder_name[:-7]))
            viz.image(visualize(th.rot90(torch_gt)), opts=dict(caption="GT Mask"))
            # viz.image(visualize(th.rot90(img[0, 1, ...])), opts=dict(caption="img input1"))
            # viz.image(visualize(th.rot90(img[0, 2, ...])), opts=dict(caption="img input2"))
            # viz.image(visualize(th.rot90(img[0, 3, ...])), opts=dict(caption="img input3"))
            # viz.image(visualize(th.rot90(img[0, 4, ...])), opts=dict(caption="img input4"))

            logger.log("sampling...")

            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)


            # for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            for i in range(8):
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                # print(sample_fn)
                sample, x_noisy, org = sample_fn(
                    model,
                    (args.batch_size, 4, args.image_size, args.image_size), img,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

                s = th.tensor(sample)
                viz.image(visualize(th.rot90(sample[0, 0, ...])), opts=dict(caption=f"sampled output {i}"))
                # th.save(s, './results/'+str(slice_ID)+'_output'+str(i)) #save the generated mask
        else:
            pass


def create_argparser():
    defaults = dict(
        data_dir="/datassd/PSMA_SEGMENTATION/DATA/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False, #False
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
