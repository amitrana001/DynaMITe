import torch
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np

class Predictor:

    def __init__(self,clicker):
        
        self.sam = sam_model_registry["vit_h"](checkpoint="/home/rana/Thesis/segment-anything/sam_vit_h_4b8939.pth")
        self.sam.to(device='cuda')
        # self.sam = sam_model_registry["vit_b"](checkpoint="/home/rana/Thesis/segment-anything/sam_vit_b_01ec64.pth")
        self.predictor = SamPredictor(self.sam)
        image = np.asarray(clicker.inputs[0]['image'].permute(1,2,0))
        image = cv2.resize(image, (clicker.orig_w, clicker.orig_h))
        self.predictor.set_image(image)

    def get_prediction(self, clicker):
        # self.point_coords = []
        # self.point_labels = []
        # for j, fg_coords_per_mask in enumerate(clicker.fg_orig_coords):
        #     for i, coords in enumerate(fg_coords_per_mask):
        #         self.point_coords.append([coords[1], coords[0]])
        #         self.point_labels.append(1)
        # if len(clicker.bg_orig_coords):
        #     for i, coords in enumerate(clicker.bg_orig_coords):
        #         self.point_coords.append([coords[1], coords[0]])
        #         self.point_labels.append(0)
        self.point_coords= np.asarray(clicker.point_coords)
        self.point_labels = np.asarray(clicker.click_sequence)+1

        prev_masks = np.asarray(clicker.pred_masks, dtype=np.float32) if (clicker.pred_masks is not None) else None
        if prev_masks is not None:
            prev_masks = cv2.resize(prev_masks[0][:,:,None], (256,256))
            pred_masks, _, _ = self.predictor.predict(point_coords=self.point_coords, point_labels=self.point_labels,
                                                    mask_input=prev_masks[None,:,:])
        else:
            pred_masks, _, _ = self.predictor.predict(point_coords=self.point_coords, point_labels=self.point_labels,
                                                    mask_input=None)
        # pred_masks = np.max(pred_masks, axis=0)
        return pred_masks
        # return torch.from_numpy(pred_masks[None,:,:])

 
    

