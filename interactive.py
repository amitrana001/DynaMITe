import os
from os import path
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
from detectron2.data import transforms as T
from detectron2.projects.deeplab import add_deeplab_config
from mask2former.config import add_maskformer2_config
from detectron2.config import get_cfg

class InteractiveManager:
    def __init__(self, cfg, model, image, mask):
        self.model = model

        self.transforms = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.image = image
        self.mask = TF.to_tensor(mask).unsqueeze(0).cuda()

        h, w = self.image.shape[:2]
        # self.image, self.pad = pad_divide_by(self.image, 16)
        # self.mask, _ = pad_divide_by(self.mask, 16)
        self.last_mask = None

        # Positive and negative scribbles
        self.p_srb = np.zeros((h, w), dtype=np.uint8)
        self.n_srb = np.zeros((h, w), dtype=np.uint8)

        # Used for drawing
        self.pressed = False
        self.last_ex = self.last_ey = None
        self.positive_mode = True
        self.need_update = True #True

    def mouse_down(self, ex, ey):
        self.last_ex = ex
        self.last_ey = ey
        self.pressed = True
        if self.positive_mode:
            cv2.circle(self.p_srb, (ex, ey), radius=8, color=(1), thickness=-1)
        else:
            cv2.circle(self.n_srb, (ex, ey), radius=8, color=(1), thickness=-1)
        self.need_update = True

    def mouse_move(self, ex, ey):
        if not self.pressed:
            return
        if self.positive_mode:
            cv2.line(self.p_srb, (self.last_ex, self.last_ey), (ex, ey), (1), thickness=3)
        else:
            cv2.line(self.n_srb, (self.last_ex, self.last_ey), (ex, ey), (1), thickness=3)
        self.need_update = True
        self.last_ex = ex
        self.last_ey = ey

    def mouse_up(self):
        self.pressed = False

    def run_s2m(self):
        # Convert scribbles to tensors
        
        # Rsp = torch.from_numpy(self.p_srb).unsqueeze(0).unsqueeze(0).float().cuda()
        # Rsn = torch.from_numpy(self.n_srb).unsqueeze(0).unsqueeze(0).float().cuda()
        # Rs = torch.cat([Rsp, Rsn], 1)
        # Rs, _ = pad_divide_by(Rs, 16)

        # Use the network to do stuff
        inputs = {}
        h,w = self.image.shape[:2]
        inputs["height"] = h
        inputs["width"] = w
        img=  self.transforms.get_transform(self.image).apply_image(self.image)
        inputs["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        fg_scrbs = self.transforms.get_transform(self.p_srb).apply_segmentation(self.p_srb)
        bg_scrbs = self.transforms.get_transform(self.n_srb).apply_segmentation(self.n_srb)
        fg_scrbs = torch.from_numpy(fg_scrbs).unsqueeze(0).float()
        bg_scrbs = torch.from_numpy(bg_scrbs).unsqueeze(0).float()
        inputs["fg_scrbs"] = fg_scrbs
        inputs["bg_scrbs"] = bg_scrbs
        inputs["scrbs_count"] = 2


        inputs = [inputs]
        pred = self.model(inputs)[0]
        mask = pred['instances'].pred_masks[0]
        # print("mask", mask.shape)

        # We don't overwrite current mask until commit
        self.last_mask = mask
        np_mask = (mask.detach().cpu().numpy() * 255).astype(np.uint8)

        return np_mask

    def commit(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        if self.last_mask is not None:
            self.mask = self.last_mask

    def clean_up(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        self.mask.zero_()
        self.last_mask = None



parser = ArgumentParser()
parser.add_argument('--image', default='datasets/coco/val2017/000000218362.jpg')
parser.add_argument('--config_file', default="configs/coco/instance-segmentation/interactive_maskformer2_R50_bs16_50ep.yaml")
parser.add_argument('--model', default="all_data/interactive_output_multiscale/model_final.pth")
parser.add_argument('--save_mask', default = 'interactive_output_multiscale/inference/')
parser.add_argument('--mask', default=None)
args = parser.parse_args()



cfg = get_cfg()
# # cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
# cfg.merge_from_file("interactive_output/config.yaml")
cfg.merge_from_file(args.config_file)


from train_net import Trainer
model = Trainer.build_model(cfg)
model.eval()
cfg.MODEL.WEIGHTS = args.model
model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])
torch.set_grad_enabled(False)

# Reading stuff
image = cv2.imread(args.image, cv2.IMREAD_COLOR)
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
if args.mask is None:
    mask = np.zeros((h, w), dtype=np.uint8)
else:
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

manager = InteractiveManager(cfg, model, image, mask)

def mouse_callback(event, x, y, *args):
    if event == cv2.EVENT_LBUTTONDOWN:
        manager.mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        manager.mouse_up()
    elif event == cv2.EVENT_MBUTTONDOWN:
        manager.positive_mode = not manager.positive_mode
        if manager.positive_mode:
            print('Entering foreground scribble mode.')
        else:
            print('Entering background scribble mode.')

    # Draw
    if event == cv2.EVENT_MOUSEMOVE:
        manager.mouse_move(x, y)

from detectron2.utils.colormap import colormap
color_map = colormap(rgb=True, maximum=255)

def comp_image(image, mask, p_srb, n_srb):
    # print(image.shape, mask.shape, p_srb.shape, n_srb.shape)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    # color_mask[:,:,1] = 1
    color_mask[:,:,:] = color_map[0]
    if len(mask.shape) == 2:
        mask = mask[:,:,None]
    comp = (image*0.5 + color_mask*mask*0.5).astype(np.uint8)
    comp[p_srb>0.5, :] = np.array([0, 255, 255], dtype=np.uint8)
    comp[n_srb>0.5, :] = np.array([255, 0, 0], dtype=np.uint8)

    return comp[:,:,::-1]

# OpenCV setup
cv2.namedWindow('S2M demo')
cv2.setMouseCallback('S2M demo', mouse_callback)

print('Usage: python interactive.py --image <image> --model <model> [Optional: --mask initial_mask]')
print('This GUI is rudimentary; the network is naively designed.')
print('Mouse Left - Draw scribbles')
print('Mouse middle key - Switch positive/negative')
print('Key f - Commit changes, clear scribbles')
print('Key r - Clear everything')
print('Key d - Switch between overlay/mask view')
print('Key s - Save masks into a temporary output folder (./output/)')

display_comp = True
while 1:
    if manager.need_update:
        if np.any(manager.p_srb):
            np_mask = manager.run_s2m()
        else:
            np_mask = np.zeros((h, w), dtype=np.uint8)
        if display_comp:
            display = comp_image(image, np_mask, manager.p_srb, manager.n_srb)
        else:
            display = np_mask
        manager.need_update = False

    cv2.imshow('S2M demo', display)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('f'):
        manager.commit()
        manager.need_update = True
    elif k == ord('s'):
        print('saved')
        os.makedirs('output', exist_ok=True)
        # cv2.imwrite('output/%s' % path.basename(args.mask), mask)
        cv2.imwrite('output/' + args.save_mask + ".jpg", np_mask)
    elif k == ord('d'):
        display_comp = not display_comp
        manager.need_update = True
    elif k == ord('r'):
        manager.clean_up()
        manager.need_update = True
    elif k == 27:
        break

cv2.destroyAllWindows()
