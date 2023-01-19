import torch
import numpy as np
from tkinter import messagebox
import cv2
import pdb
# from .clicker import Clicker as clicker
from interactive_demo_tool import clicker
# from isegm.inference.predictors import get_predictor
# from isegm.utils.vis import draw_with_blend_and_clicks
from detectron2.data import transforms as T
from detectron2.utils.visualizer import Visualizer
import torchvision.transforms.functional as F
from torch.nn import functional as Fn
from detectron2.utils.colormap import colormap
# color_map = colormap(rgb=True, maximum=1)

def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))
color_map = get_palette(80)[1:]

import copy
class InteractiveController:
    def __init__(self, model, update_image_callback, cfg, prob_thresh=0.5):
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()

        self.transforms = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_masks = None
        self._init_mask = None

        self.fg_scrbs = None
        self.bg_scrbs = None

        self.p_scrbs = None
        self.n_scrbs = None
        self._inputs = {}

        self.image = None
        self.predictor = model
        # self.device = device
        self.update_image_callback = update_image_callback
        # self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        self.image = copy.deepcopy(image)
        self.fg_scrbs = []
        self.bg_scrbs = []

        self.p_scrbs = None
        self.list_p_scrbs = []
        self.n_scrbs = None
        self.num_scrbs_per_mask = []
        h,w = self.image.shape[:2]
        self._inputs["height"] = h
        self._inputs["width"] = w
        img=  self.transforms.get_transform(self.image).apply_image(self.image)
        self._inputs["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # calculate and store multi-scale image features once
        self.features = None
        self.mask_features = None
        self.transformer_encoder_features = None
        self.multi_scale_features = None

        # self._result_masks = torch.zeros((1,h,w), dtype=torch.uint8)
        self._result_masks = None
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    
    def add_click(self, x, y, bg_click =False, inst_num=1):

        if inst_num==0:
            print("add an instance")
            self.update_image_callback()
            return 
        if bg_click:
            # self.bg_scrbs.append(np.zeros(self.image.shape[:2], dtype = np.uint8))
            self.bg_scrbs.append(np.zeros(self.image.shape[:2], dtype = np.uint8))
            cv2.circle(self.bg_scrbs[-1], (x, y), radius=5, color=(1), thickness=-1)
            # if len(self.bg_scrbs)>1:
            #     self.bg_scrbs[0] = np.max(np.stack(self.bg_scrbs[1:],0),0)
        else:
            if inst_num > len(self.list_p_scrbs):
                self.list_p_scrbs.append(np.zeros(self.image.shape[:2], dtype = np.uint8))
            cv2.circle(self.list_p_scrbs[inst_num-1], (x, y), radius=5, color=(1), thickness=-1)

            point_mask = np.zeros(self.image.shape[:2], dtype = np.uint8)
            cv2.circle(point_mask, (x, y), radius=5, color=(1), thickness=-1)
            new_h, new_w = self._inputs["image"].shape[-2:]
            point_mask = F.resize(torch.from_numpy(point_mask).unsqueeze(0), (new_h, new_w)).float()
            
            if inst_num > len(self.fg_scrbs):
                self.num_scrbs_per_mask.append(1)
                self.fg_scrbs.append(point_mask)
            else:
                self.num_scrbs_per_mask[inst_num-1]+=1
                self.fg_scrbs[inst_num-1] = torch.cat((self.fg_scrbs[inst_num-1],point_mask))
        
        self.p_scrbs = np.stack(self.list_p_scrbs,axis=0)

        new_h, new_w = self._inputs["image"].shape[-2:]
        # fg_scrbs = F.resize(torch.from_numpy(self.p_scrbs), (new_h, new_w)).float()
        if len(self.bg_scrbs)==0:
            bg_scrbs = None
        else:
            self.n_scrbs = np.stack(self.bg_scrbs,axis=0)
            bg_scrbs = [F.resize(torch.from_numpy(self.n_scrbs), (new_h, new_w)).float()]
        
        self._inputs["fg_scrbs"] = self.fg_scrbs
        self._inputs["bg_scrbs"] = bg_scrbs
        self._inputs["scrbs_count"] =  5
        self._inputs["num_scrbs_per_mask"] = self.num_scrbs_per_mask
        # print(self._inputs["scrbs_count"])
        # pdb.set_trace()
        # im, _ = self.get_visualization()
        # im = cv2.cvtColor(im,  cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"output/demo_results/save_before_pred_{inst_num}.png", im)
        inputs = [self._inputs]
        self.pred = self.predictor(inputs)[0][0]
        self._result_masks = self.pred['instances'].pred_masks
        # print("mask", mask.shape)

        # We don't overwrite current mask until commit
        self.last_masks = self._result_masks

        self.update_image_callback()
    
    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def reset_clicks(self):
        self._result_masks = None
        self.p_scrbs = None
        self.n_scrbs = None
        self.fg_scrbs = []
        self.bg_scrbs = []

        self.p_scrbs = None
        self.list_p_scrbs = []
        self.n_scrbs = None
        self.num_scrbs_per_mask = []
        # self.set_image(self.image)
        self.update_image_callback(reset_clicks=True)

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        pass
        # if predictor_params is not None:
        #     self.predictor_params = predictor_params
        # self.predictor = get_predictor(self.net, device=self.device,
        #                                **self.predictor_params)
        # if self.image is not None:
        #     self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_masks(self):
        if self._result_masks is not None:
            result_masks = self._result_masks.clone()
        else:
            result_masks = self._result_masks
        # if self.probs_history:
        #     result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_masks

    def get_visualization(self, alpha_blend=0.3, click_radius=3, reset_clicks=False,show_only_masks=False):
        from detectron2.utils.visualizer import Visualizer
        if self.image is None:
            return None, None

        result_masks_for_vis = self.result_masks
        image = np.asarray(copy.deepcopy(self.image))
        if (result_masks_for_vis is None) or (reset_clicks):
            return image, None

        result_masks_for_vis = result_masks_for_vis.to(device ='cpu')
        # image = np.asarray(self.image)
        
        visualizer = Visualizer(image, metadata=None)
        pred_masks = F.resize(result_masks_for_vis.to(dtype=torch.uint8), image.shape[:2])
        c = []
        for i in range(pred_masks.shape[0]):
            # c.append(color_map[2*(i)+2]/255.0)
            c.append(color_map[i]/255.0)
        # pred_masks = np.asarray(pred_masks).astype(np.bool_)
        vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c,alpha=alpha_blend)
        # [Optional] prepare labels

        image = vis.get_image()
        # # Laminate your image!
        # fig = overlay_masks(image, masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
        total_colors = len(color_map)-1
        
        point_clicks_map = np.ones_like(image)*255
        if self.p_scrbs is not None:
            for i, scrb in enumerate(self.p_scrbs):
                color = np.array(color_map[total_colors-5*i-4], dtype=np.uint8)
                # color = np.array([0,255,0], dtype=np.uint8)
                if not show_only_masks:
                    image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
                point_clicks_map[scrb>0.5, :] = np.array(color, dtype=np.uint8)
        if self.n_scrbs is not None:
            for i, scrb in enumerate(self.n_scrbs):
                color = np.array([255,0,0], dtype=np.uint8)
                if not show_only_masks:
                    image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
                point_clicks_map[scrb>0.5, :] = np.array(color, dtype=np.uint8)
        image = image.clip(0,255)
        return image, point_clicks_map


def get_color_from_map(index):
    # color_c='#%02x%02x%02x' % (color_map[2*(index)+2][0], color_map[2*(index)+2][1], color_map[2*(index)+2][2])
    color_c='#%02x%02x%02x' % (color_map[index][0], color_map[index][1], color_map[index][2])
    return color_c
    