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
from mask2former.evaluation.eval_utils import prepare_scribbles

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
        # # store input for transformer decoder
        # self.device = None
        # self.click_counts = 0
        # self.prev_mask_logits = None
        # self.processed_results = None
        # self.outputs = None
        # self.images=None
        # self.scribbles=None
        # self.num_insts = None
        # self.features = None
        # self.mask_features = None
        # self.transformer_encoder_features = None 
        # self.multi_scale_features=None
        # self.batched_num_scrbs_per_mask = None
        # self.batched_fg_coords_list = None
        # self.batched_bg_coords_list = None
        self.update_image_callback = update_image_callback
        # self.predictor_params = predictor_params

        self.fg_orig_list = []
        self.bg_orig_list = []
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

        self.new_h, self.new_w = self._inputs["image"].shape[-2:]
        self.rh = self.new_h/h
        self.rw = self.new_w/w

         # store input for transformer decoder
        self.device = None
        self.click_counts = 0
        self.prev_mask_logits = None
        self.processed_results = None
        self.outputs = None
        self.images=None
        self.scribbles=None
        self.num_insts = None
        self.features = None
        self.mask_features = None
        self.transformer_encoder_features = None 
        self.multi_scale_features=None
        self.batched_num_scrbs_per_mask = None
        self.batched_fg_coords_list = None
        self.batched_bg_coords_list = None
        self.batched_max_timestamp = [1]
        # calculate and store multi-scale image features once
        # self.features = None
        # self.mask_features = None
        # self.transformer_encoder_features = None
        # self.multi_scale_features = None
        self.fg_orig_list = []
        self.bg_orig_list = []
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
        # new_y = round(y*self.rh)
        # new_x = round(x*self.rw)
        new_y = y*self.rh
        new_x = x*self.rw
        if bg_click:
            # self.bg_scrbs.append(np.zeros(self.image.shape[:2], dtype = np.uint8))
            
            if self.scribbles:
                scrbs = self.get_scrbs_from_click(x,y)
                if self.batched_bg_coords_list[0]:
                    self.scribbles[0][-1] = torch.cat((self.scribbles[0][-1],scrbs))
                    self.batched_bg_coords_list[0].extend([[new_y, new_x,self.click_counts]])
                else:
                    self.scribbles[0][-1] = scrbs
                    self.batched_bg_coords_list[0] = [[new_y, new_x,self.click_counts]]
            else:
                self.bg_scrbs.append(np.zeros(self.image.shape[:2], dtype = np.uint8))
                cv2.circle(self.bg_scrbs[-1], (x, y), radius=5, color=(1), thickness=-1)
            self.bg_orig_list.append([x, y, self.click_counts])
            # if len(self.bg_scrbs)>1:
            #     self.bg_scrbs[0] = np.max(np.stack(self.bg_scrbs[1:],0),0)
        else:
            # scrbs = self.get_scrbs_from_click(scrbs, x,y,self.images)
            if self.scribbles and inst_num <= len(self.batched_num_scrbs_per_mask[0]):
                scrbs = self.get_scrbs_from_click(x,y)
                self.scribbles[0][inst_num-1] = torch.cat([self.scribbles[0][inst_num-1], scrbs], 0)
                self.batched_num_scrbs_per_mask[0][inst_num-1] += 1
                self.batched_fg_coords_list[0][inst_num-1].extend([[new_y, new_x,self.click_counts]])
                self.fg_orig_list[inst_num-1].extend([[x, y, self.click_counts]])
            elif self.scribbles:
                scrbs = self.get_scrbs_from_click(x, y)
                # print(len(self.scribbles[0]))
                # print(self.scribbles[0][-1])
                # print(scrbs.shape)
                # print(self.scribbles[0][0].shape)
                self.batched_num_scrbs_per_mask[0].append(1)
                self.batched_fg_coords_list[0].append([[new_y, new_x,self.click_counts]])
                self.fg_orig_list.append([[x, y, self.click_counts]])
                self.scribbles[0].insert(inst_num-1, scrbs)
                # print(len(self.scribbles[0]))
                # print(self.scribbles[0][-1])
            else:
                # if inst_num > len(self.list_p_scrbs):
                #     self.list_p_scrbs.append(np.zeros(self.image.shape[:2], dtype = np.uint8))
                # cv2.circle(self.list_p_scrbs[inst_num-1], (x, y), radius=5, color=(1), thickness=-1)
                self.fg_orig_list = [[[x, y, self.click_counts]]]
                point_mask = np.zeros(self.image.shape[:2], dtype = np.uint8)
                cv2.circle(point_mask, (x, y), radius=5, color=(1), thickness=-1)
                new_h, new_w = self._inputs["image"].shape[-2:]
                point_mask = F.resize(torch.from_numpy(point_mask).unsqueeze(0), (new_h, new_w)).int()
                
                self._inputs["fg_scrbs"] = [point_mask]
                self._inputs["bg_scrbs"] = None
                self._inputs["fg_click_coords"] = [[[new_y, new_x, self.click_counts]]]
                self._inputs["bg_click_coords"] = None
                self._inputs["scrbs_count"] =  5

                self._inputs["num_scrbs_per_mask"] = [1]
        self.click_counts+=1
        if self.num_insts is not None:
            self.num_insts[0] = len(self.batched_num_scrbs_per_mask[0])
        # print(self._inputs["scrbs_count"])
        # pdb.set_trace()
        # im, _ = self.get_visualization()
        # im = cv2.cvtColor(im,  cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"output/demo_results/save_before_pred_{inst_num}.png", im)
        inputs = [self._inputs]

        if self.features is None:
            (processed_results, outputs, self.images, self.scribbles,
            self.num_insts, self.features, self.mask_features,
            self.transformer_encoder_features, self.multi_scale_features,
            self.batched_num_scrbs_per_mask, self.batched_fg_coords_list,
            self.batched_bg_coords_list) = self.predictor(inputs,batched_max_timestamp=self.batched_max_timestamp)
            self.device = self.images.tensor.device
        else:
            (processed_results, outputs, self.images, self.scribbles,
            self.num_insts, self.features, self.mask_features,
            self.transformer_encoder_features, self.multi_scale_features,
            self.batched_num_scrbs_per_mask, self.batched_fg_coords_list,
            self.batched_bg_coords_list) = self.predictor(inputs, self.images, self.scribbles, self.num_insts,
                                                self.features, self.mask_features, self.transformer_encoder_features,
                                                self.multi_scale_features, self.prev_mask_logits,
                                                self.batched_num_scrbs_per_mask,
                                                self.batched_fg_coords_list, self.batched_bg_coords_list,
                                                batched_max_timestamp=self.batched_max_timestamp)

        # self.pred = self.predictor(inputs)[0][0]
        self._result_masks = processed_results[0]['instances'].pred_masks
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
        self.device = None
        self.click_counts = 0
        self.prev_mask_logits = None
        self.processed_results = None
        self.outputs = None
        self.images=None
        self.scribbles=None
        self.num_insts = None
        self.features = None
        self.mask_features = None
        self.transformer_encoder_features = None 
        self.multi_scale_features=None
        self.batched_num_scrbs_per_mask = None
        self.batched_fg_coords_list = None
        self.batched_bg_coords_list = None
        self._result_masks = None
        self.batched_max_timestamp = [1]
        self.p_scrbs = None
        self.n_scrbs = None
        self.fg_scrbs = []
        self.bg_scrbs = []
        self.fg_orig_list = []
        self.bg_orig_list = []

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

    def get_scrbs_from_click(self,x,y):
        point_mask = np.zeros(self.image.shape[:2], dtype = np.uint8)
        cv2.circle(point_mask, (x, y), radius=3, color=(1), thickness=-1)
        point_mask = torch.from_numpy(point_mask).to(self.device, dtype = torch.uint8).unsqueeze(0)
        point_mask = F.resize(point_mask, (self.new_h, self.new_w)).int()
        point_mask = prepare_scribbles(point_mask, self.images)
        return point_mask
        
    # def get_visualization(self, alpha_blend=0.3, click_radius=3, reset_clicks=False,show_only_masks=False):
    #     from detectron2.utils.visualizer import Visualizer
    #     if self.image is None:
    #         return None, None

    #     result_masks_for_vis = self.result_masks
    #     image = np.asarray(copy.deepcopy(self.image))
    #     if (result_masks_for_vis is None) or (reset_clicks):
    #         return image, None

    #     result_masks_for_vis = result_masks_for_vis.to(device ='cpu')
    #     # image = np.asarray(self.image)
        
    #     visualizer = Visualizer(image, metadata=None)
    #     pred_masks = F.resize(result_masks_for_vis.to(dtype=torch.uint8), image.shape[:2])
    #     c = []
    #     for i in range(pred_masks.shape[0]):
    #         # c.append(color_map[2*(i)+2]/255.0)
    #         c.append(color_map[i]/255.0)
    #     # pred_masks = np.asarray(pred_masks).astype(np.bool_)
    #     vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c,alpha=alpha_blend)
    #     # [Optional] prepare labels

    #     image = vis.get_image()
    #     # # Laminate your image!
    #     # fig = overlay_masks(image, masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
    #     total_colors = len(color_map)-1
        
    #     point_clicks_map = np.ones_like(image)*255
    #     if not show_only_masks:
    #         if len(self.fg_orig_list):
    #             for j, fg_coords_per_mask in enumerate(self.fg_orig_list):
    #                 for i, coords in enumerate(fg_coords_per_mask):
    #                     color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
    #                     color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
    #                     image = cv2.circle(image, (int(coords[0]), int(coords[1])), click_radius, tuple(color), -1)
            
    #         if len(self.bg_orig_list):
    #             for i, coords in enumerate(self.bg_orig_list):
    #                 color = np.array([255,0,0], dtype=np.uint8)
    #                 color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
    #                 image = cv2.circle(image, (int(coords[0]), int(coords[1])), click_radius, tuple(color), -1)
    #     return image, point_clicks_map
    
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
        
        # visualizer = Visualizer(image, metadata=None)
        # print(result_masks_for_vis.shape)
        # pred_masks = F.resize(result_masks_for_vis.to(dtype=torch.uint8), image.shape[:2])
        # print(pred_masks.shape)
        pred_masks =np.asarray(result_masks_for_vis,dtype=np.uint8)
        c = []
        for i in range(pred_masks.shape[0]):
            # c.append(color_map[2*(i)+2]/255.0)
            c.append(color_map[i]/255.0)
        # pred_masks = np.asarray(pred_masks).astype(np.bool_)
        # vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c,alpha=alpha_blend)

        # [Optional] prepare labels

        # image = vis.get_image()
        for i in range(pred_masks.shape[0]):
            image = self.apply_mask(image, pred_masks[i], c[i],alpha_blend)
        # # Laminate your image!
        # fig = overlay_masks(image, masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
        total_colors = len(color_map)-1
        
        point_clicks_map = np.ones_like(image)*255
        if not show_only_masks:
            if len(self.fg_orig_list):
                for j, fg_coords_per_mask in enumerate(self.fg_orig_list):
                    for i, coords in enumerate(fg_coords_per_mask):
                        color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
                        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                        image = cv2.circle(image, (int(coords[0]), int(coords[1])), click_radius, tuple(color), -1)
            
            if len(self.bg_orig_list):
                for i, coords in enumerate(self.bg_orig_list):
                    color = np.array([255,0,0], dtype=np.uint8)
                    color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                    image = cv2.circle(image, (int(coords[0]), int(coords[1])), click_radius, tuple(color), -1)
        return image, point_clicks_map

    def apply_mask(self, image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        return image
# def get_visualization(image, instances, prev_output=None, batched_fg_coords_list=None,batched_bg_coords_list=None,
#                   alpha_blend=0.6, num_iter = 0):
#     import copy
#     image = copy.deepcopy(image.cpu())
#     batched_fg_coords_list = copy.deepcopy(batched_fg_coords_list)
#     batched_bg_coords_list = copy.deepcopy(batched_bg_coords_list)
#     image = np.asarray(image.permute(1,2,0))
#     visualizer = Visualizer(image, metadata=None)
#     if prev_output is not None:
#         import torchvision.transforms.functional as F
#         pred_masks = F.resize(prev_output.pred_masks.detach().to(dtype=torch.uint8), image.shape[:2])
#     else:
#         pred_masks = instances.gt_masks.cpu()
#     c = []
#     for i in range(pred_masks.shape[0]):
#         # c.append(color_map[2*(i)+2]/255.0)
#         c.append(color_map[i]/255.0)
#     # pred_masks = np.asarray(pred_masks).astype(np.bool_)
#     vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c, alpha=alpha_blend)
#     # [Optional] prepare labels

#     image = vis.get_image()
#     # # Laminate your image!
#     total_colors = len(color_map)-1
    
#     h,w = image.shape[:2]
#     if batched_fg_coords_list is not None:
        
#         for j, fg_coords_per_mask in enumerate(batched_fg_coords_list[0]):
#             for i, coords in enumerate(fg_coords_per_mask):
#                 color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
#                 color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
#                 if i==0:
#                     image = cv2.circle(image, (int(coords[1]), int(coords[0])), 8, tuple(color), -1)
#                 else:
#                     image = cv2.circle(image, (int(coords[1]), int(coords[0])), 3, tuple(color), -1)
        
#         if batched_bg_coords_list[0]:
#             for i, coords in enumerate(batched_bg_coords_list[0]):
#                 color = np.array([255,0,0], dtype=np.uint8)
#                 color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
#                 image = cv2.circle(image, (int(coords[1]), int(coords[0])), 3, tuple(color), -1)

#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     cv2.imshow("Image",image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def get_color_from_map(index):
    # color_c='#%02x%02x%02x' % (color_map[2*(index)+2][0], color_map[2*(index)+2][1], color_map[2*(index)+2][2])
    color_c='#%02x%02x%02x' % (color_map[index][0], color_map[index][1], color_map[index][2])
    return color_c
    