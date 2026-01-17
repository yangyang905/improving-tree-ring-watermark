import argparse
import json
import os

import omegaconf
import torch
from deps.wam.watermark_anything.augmentation.augmenter import Augmenter
from deps.wam.watermark_anything.data.transforms import (
    normalize_img,
    unnormalize_img,
)
from deps.wam.watermark_anything.models import Wam, build_embedder, build_extractor
from deps.wam.watermark_anything.modules.jnd import JND

def load_model_from_checkpoint(json_path, ckpt_path):
    """
    Load a model from a checkpoint file and a JSON file containing the parameters.
    Args:
    - json_path (str): the path to the JSON file containing the parameters
    - ckpt_path (str): the path to the checkpoint file
    """
    # Load the JSON file
    with open(json_path, 'r') as file:
        params = json.load(file)
    # Create an argparse Namespace object from the parameters
    args = argparse.Namespace(**params)
    # print(args)
    
    # Load configurations
    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    embedder_params = embedder_cfg[args.embedder_model]
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    extractor_params = extractor_cfg[args.extractor_model]
    augmenter_cfg = omegaconf.OmegaConf.load(args.augmentation_config)
    attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
        
    # Build models
    embedder = build_embedder(args.embedder_model, embedder_params, args.nbits)
    extractor = build_extractor(extractor_cfg.model, extractor_params, args.img_size, args.nbits)
    augmenter = Augmenter(**augmenter_cfg)
    try:
        attenuation = JND(**attenuation_cfg[args.attenuation], preprocess=unnormalize_img, postprocess=normalize_img)
    except:
        attenuation = None
    
    # Build the complete model
    wam = Wam(embedder, extractor, augmenter, attenuation, args.scaling_w, args.scaling_i)
    
    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        wam.load_state_dict(checkpoint)
        print("Model loaded successfully from", ckpt_path)
        print(params)
    else:
        print("Checkpoint path does not exist:", ckpt_path)
    
    return wam