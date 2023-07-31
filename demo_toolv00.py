import argparse
from argparse import ArgumentParser
import tkinter as tk

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
import os
from detectron2.data import transforms as T
from detectron2.projects.deeplab import add_deeplab_config
from dynamite.config import add_maskformer2_config, add_hrnet_config
from detectron2.config import get_cfg

from interactive_demo_tool.app import InteractiveDemoApp


def main():
    args = parse_args()

    cfg = get_cfg()
    # # cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_hrnet_config(cfg)
    # cfg.merge_from_file("interactive_output/config.yaml")
    cfg.merge_from_file(args.config_file)

    from train_net import Trainer
    model = Trainer.build_model(cfg)
    model.eval()
    cfg.MODEL.WEIGHTS = args.model
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])
    torch.set_grad_enabled(False)

    # torch.backends.cudnn.deterministic = True
    # checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    # model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)
    # import debugpy
    # debugpy.listen(5678)
    # print("Waiting for debugger")
    # debugpy.wait_for_client()

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, cfg, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser = ArgumentParser()
    parser.add_argument('--image', default='datasets/coco/val2017/000000218362.jpg')
    parser.add_argument('--config_file', default="configs/coco/iterative/instance-segmentation/iterative_maskformer2_R50_bs16_50ep.yaml")
    parser.add_argument('--model', default="./all_data/iterative_train_scratch/model_final.pth")
    parser.add_argument('--save_mask', default = 'interactive_output_multiscale/inference/')
    parser.add_argument('--mask', default=None)
    args = parser.parse_args()

    # parser.add_argument('--checkpoint', type=str, required=True,
    #                     help='The path to the checkpoint. '
    #                          'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
    #                          'or an absolute path. The file extension can be omitted.')

    # parser.add_argument('--gpu', type=int, default=0,
    #                     help='Id of GPU to use.')

    # parser.add_argument('--cpu', action='store_true', default=False,
    #                     help='Use only CPU for inference.')

    # parser.add_argument('--limit-longest-size', type=int, default=800,
    #                     help='If the largest side of an image exceeds this value, '
    #                          'it is resized so that its largest side is equal to this value.')

    # parser.add_argument('--cfg', type=str, default="config.yml",
    #                     help='The path to the config file.')

    # args = parser.parse_args()
    # if args.cpu:
    #     args.device =torch.device('cpu')
    # else:
    #     args.device = torch.device(f'cuda:{args.gpu}')
    # cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args


if __name__ == '__main__':
    main()
