import torchvision
import torch
import numpy as np
import cv2
from mask2former.data.points.annotation_generator import create_circular_mask
from mask2former.evaluation.eval_utils import prepare_scribbles

class Clicker:

    def __init__(self, model, inputs, sampling_strategy =1, normalize_time=True, 
                click_radius = 3):
        
        self.predictor = model
        self.inputs = inputs
        
        self.click_radius = click_radius
        #TO DO
        self.normalize_time= normalize_time
        self.batched_max_timestamp = None
        
        # For sampling next click
        self.sampling_strategy = sampling_strategy
        self.not_clicked_map = None
        self.fg_click_map = None
        self.bg_click_map = None

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
        self._set_gt_info()
        if self.normalize_time:
            self.batched_max_timestamp = [self.num_instances-1]
    
    def _set_gt_info(self):

        self.gt_masks = self.inputs[0]['instances'].gt_masks.to('cpu')
        self.num_instances, self.orig_h, self.orig_w = self.gt_masks.shape[:]

        self.trans_h, self.trans_w = self.inputs[0]['image'].shape[-2:]
        
        self.ratio_h = self.trans_h/self.orig_h
        self.ratio_w = self.trans_w/self.orig_w
        self.semantic_map = self.inputs[0]['semantic_map'].to('cpu')

        self.not_clicked_map = np.ones_like(self.gt_masks[0], dtype=np.bool)
        if self.sampling_strategy == 0:
            # coords = inputs[0]["coords"]
            for coords_list in self.inputs[0]['orig_fg_click_coords']:
                for coords in coords_list:
                    self.not_clicked_map[coords[0], coords[1]] = False
        elif self.sampling_strategy == 1:
            all_scribbles = torch.cat(self.inputs[0]['fg_scrbs']).to('cpu')
            point_mask = torch.max(all_scribbles,dim=0).values
            self.not_clicked_map[torch.where(point_mask)] = False

    
    def predict(self):
        if self.features is None:
            (processed_results, outputs, self.images, self.scribbles,
            self.num_insts, self.features, self.mask_features,
            self.transformer_encoder_features, self.multi_scale_features,
            self.batched_num_scrbs_per_mask, self.batched_fg_coords_list,
            self.batched_bg_coords_list) = self.predictor(self.inputs, batched_max_timestamp=self.batched_max_timestamp)
            # self.device = self.images.tensor.device
        else:
            (processed_results, outputs, self.images, self.scribbles,
            self.num_insts, self.features, self.mask_features,
            self.transformer_encoder_features, self.multi_scale_features,
            self.batched_num_scrbs_per_mask, self.batched_fg_coords_list,
            self.batched_bg_coords_list) = self.predictor(self.inputs, self.images, self.scribbles, self.num_insts,
                                                self.features, self.mask_features, self.transformer_encoder_features,
                                                self.multi_scale_features, self.prev_mask_logits,
                                                self.batched_num_scrbs_per_mask,
                                                self.batched_fg_coords_list, self.batched_bg_coords_list,
                                                self.batched_max_timestamp)
        self.device = self.images.tensor.device
        self.pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
        # self.trans_pred_masks = self.transformation(self.pred_masks)

        return self.compute_iou()

    def get_next_click(self, refine_obj_index, time_step,padding=True):

        gt_mask = self.trans_gt_masks[refine_obj_index]
        pred_mask = self.trans_pred_masks[refine_obj_index]

        gt_mask = np.asarray(gt_mask, dtype = np.bool_)
        pred_mask = np.asarray(pred_mask, dtype = np.bool_)

        fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
        fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)
        
        H, W = gt_mask.shape

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        import cv2
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist

        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        sample_locations = [[coords_y[0], coords_x[0]]]

        obj_index = self.semantic_map[coords_y[0]][coords_x[0]] - 1
        pm = create_circular_mask(H, W, centers=sample_locations, radius=self.click_radius)
        
        if self.sampling_strategy == 0:
            self.not_clicked_map[coords_y[0], coords_x[0]] = False
        elif self.sampling_strategy == 1:
            self.not_clicked_map[np.where(pm==1)] = False

        scrbs = torch.from_numpy(pm).to(self.device, dtype = torch.float).unsqueeze(0)
        scrbs = prepare_scribbles(scrbs,self.images)
        if obj_index == -1:
            if self.batched_bg_coords_list[0]:
                self.scribbles[0][-1] = torch.cat((self.scribbles[0][-1],scrbs))
                self.batched_bg_coords_list[0].extend([[coords_y[0],coords_x[0],time_step]])
            else:
                self.scribbles[0][-1] = scrbs
                self.batched_bg_coords_list[0] = [[coords_y[0], coords_x[0],time_step]]
        else:
            self.scribbles[0][obj_index] = torch.cat([self.scribbles[0][obj_index], scrbs], 0)
            self.batched_num_scrbs_per_mask[0][obj_index] += 1
            self.batched_fg_coords_list[0][obj_index].extend([[coords_y[0], coords_x[0],time_step]])

        if self.normalize_time:
            self.batched_max_timestamp[0]+=1          

        return obj_index

    # def compute_iou(self):
    #     ious = []
    #     for gt_mask, pred_mask in zip(self.trans_gt_masks, self.trans_pred_masks):
    #         intersection = (gt_mask * pred_mask).sum()
    #         union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
    #         ious.append(intersection/union)
    #     return ious

    def compute_iou(self):

        ious = []
        for gt_mask, pred_mask in zip(self.gt_masks, self.pred_masks):
            intersection = (gt_mask * pred_mask).sum()
            union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
            ious.append(intersection/union)
        return ious

    def get_visualization(self):
        pass
    
    def get_obj_areas(self):
        obj_areas = np.zeros(self.num_instances)
        for i in range(self.num_instances):
            obj_areas[i] = self.gt_masks[i].sum()/(self.orig_h * self.orig_w)
        return obj_areas
    # @property
    # def num_instances(self):
    #     return self._num_instances
