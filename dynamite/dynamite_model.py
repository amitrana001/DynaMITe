#Modified by Amit Rana from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py

from itertools import accumulate
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import random
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

# from mask2former.evaluation import iterative_evaluator
from .modeling.criterion import SetFinalCriterion


@META_ARCH_REGISTRY.register()
class DynamiteModel(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        iterative_evaluation: bool,
        
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            interactive_evaluation: bool to indicate if it's just one time inference or iterative evaluation
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # iterative
        self.iterative_evaluation = iterative_evaluation

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION

        # loss weights
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
     
        # building criterion
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        losses = ["masks"]

        criterion = SetFinalCriterion(
            weight_dict=weight_dict,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        if deep_supervision:
            enc_layers = cfg.MODEL.MASK_FORMER.ENC_LAYERS
            aux_weight_dict = {}
            for i in range(enc_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,

            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,

            #iterative
            "iterative_evaluation": cfg.ITERATIVE.TEST.INTERACTIVE_EVALAUTION,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, inputs, images=None,  num_instances=None,
                features=None, mask_features=None, 
                multi_scale_features=None, num_clicks_per_object= None,
                fg_coords = None, bg_coords = None, max_timestamp=None):
        """
        Args:
            inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * "fg_click_coords": list of per-instance click coordinates
                   * "bg_click_coords": list of background coordinates
                   * "num_clicks_per_object": number of clicks sampled per instance
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
            features, mask_features, multi_scale_features:
                these are computed once per image and passed as an argument to avoid recomputation
                during iterative evaluation/inference
            fg_coords: a batched list where each item is
                * list of list of clicks coordinates for each object
            bg_coords: a batched list where each item is
                * list of background coordinates
            num_clicks_per_object: a batched list where each item  is
                * list of number of clicks per object/instance
            max_timestamp: a batched list where each item  is
                * maximum number of clicks for that image
            num_instances:  a batched list where each item  is
                * number of instances per image
        Returns:
            list[Instances]:
                each Instances has the predicted masks for one image.

        """
        
        if (images is None) or (num_clicks_per_object is None) or (fg_coords is None):        
            (images, num_instances, num_clicks_per_object,
            fg_coords, bg_coords) = self.preprocess_batch_data(inputs)
        if features is None:
            features = self.backbone(images.tensor)

        if self.training:

            if "instances" in inputs[0]:
                gt_instances = [(x["instances"].to(self.device), x['padding_mask'].to(self.device),x['bg_mask'].to(self.device)) for x in inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            
            outputs, num_clicks_per_object = self.sem_seg_head(inputs, images, features, num_instances,
                                                                    mask_features, multi_scale_features,
                                                                    num_clicks_per_object,
                                                                    fg_coords,
                                                                    bg_coords,
                                                                )
            losses = self.criterion(outputs, targets, num_clicks_per_object)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
           
        else:
            (outputs, mask_features, multi_scale_features, num_clicks_per_object) = self.sem_seg_head(inputs,  images, features, num_instances,
                                                                                    mask_features, 
                                                                                    multi_scale_features, num_clicks_per_object,
                                                                                    fg_coords, bg_coords, max_timestamp)
            processed_results = self.process_results(inputs, images, outputs, num_instances, num_clicks_per_object)
            if self.iterative_evaluation:
                return (processed_results, outputs, images,  num_instances, features, mask_features,
                        multi_scale_features, num_clicks_per_object, fg_coords, bg_coords)
            else:
                return processed_results

    def process_results(self, inputs, images, outputs, num_instances, num_clicks_per_object=None):
       
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        if num_clicks_per_object is None:
            num_clicks_per_object = [[1]*inst_per_image for inst_per_image in num_instances]

        processed_results = []
        for mask_pred_result, input_per_image, image_size, inst_per_image, num_clicks_per_object in zip(
            mask_pred_results, inputs, images.image_sizes, num_instances, num_clicks_per_object
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})
            
            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
    
            # interactive instance segmentation inference
            instance_r = retry_if_cuda_oom(self.interactive_instance_inference)(mask_pred_result, inst_per_image, num_clicks_per_object)
            processed_results[-1]["instances"] = instance_r

        return processed_results

    def preprocess_batch_data(self, inputs):
        images = [x["image"].to(self.device) for x in inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # images: [Bs, 3, H, W]
        if 'num_clicks_per_object' in inputs[0]:
            num_clicks_per_object = []
            fg_coords = []
            bg_coords = []
            num_instances = []

            for x in inputs:
                num_clicks_per_object.append(x['num_clicks_per_object'])
                fg_coords.append(x['fg_click_coords'])
                bg_coords.append(x['bg_click_coords'])
                num_instances.append(len(x['num_clicks_per_object']))
               
            return images, num_instances, num_clicks_per_object, fg_coords, bg_coords

    def prepare_targets(self, targets, images):

        new_targets = []
        for (targets_per_image, padding_mask_per_image, bg_mask_per_image) in targets:
            
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": targets_per_image.gt_masks,
                    "padding_mask": padding_mask_per_image,
                    "bg_mask": bg_mask_per_image
                }
            )
        return new_targets

    def interactive_instance_inference(self, mask_pred, num_instances, num_clicks_per_object=None):

        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        result = Instances(image_size)
        # mask (before sigmoid)
        import copy
        num_clicks_per_object_copy = copy.deepcopy(num_clicks_per_object)
        num_clicks_per_object_copy.append(mask_pred.shape[0]-sum(num_clicks_per_object))

        temp_out = []
        splited_masks = torch.split(mask_pred, num_clicks_per_object_copy, dim=0)
        for m in splited_masks[:-1]:
            temp_out.append(torch.max(m, dim=0).values)
        mask_pred = torch.cat([torch.stack(temp_out),splited_masks[-1]]) # can remove splited_masks[-1] all together

        mask_pred = torch.argmax(mask_pred,0)
        m = []
        for i in range(num_instances):
            m.append((mask_pred == i).float())
        
        mask_pred = torch.stack(m)
        result.pred_masks = mask_pred
        
        return result
