import argparse
from argparse import ArgumentParser
import tkinter as tk
import torch
from detectron2.projects.deeplab import add_deeplab_config
from dynamite.config import add_maskformer2_config, add_hrnet_config
from detectron2.config import get_cfg

from interactive_demo.app import InteractiveDemoApp

def main():
    args = parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_hrnet_config(cfg)
    cfg.merge_from_file(args.config_file)

    from train_net import Trainer
    model = Trainer.build_model(cfg)
    model.eval()
    cfg.MODEL.WEIGHTS = args.model_weights
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])
    torch.set_grad_enabled(False)

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, cfg, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser = ArgumentParser()
    parser.add_argument('--config-file', default=None, help="path to config file")
    parser.add_argument('--model-weights', default=None, help="path to model checkpoint")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
