import torch
import torchvision
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from ..data.scribble.gen_scribble import get_iterative_scribbles
import numpy as np
from ..data.points.annotation_generator import get_corrective_points, get_iterative_points, create_circular_mask,point_candidates_dt
import copy
import cv2
import random

def preprocess_batch_data(batched_inputs, device, pixel_mean, pixel_std, size_divisibility, random_bg_queries=False):
        images = [x["image"].to(device) for x in batched_inputs]
        images = [(x - pixel_mean) / pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility)

        # images: [Bs, 3, H, W]
        # for scribbles, scribbles are of varying size in a batch
        # pad the scribbles with combined bg_scribbles to match the size of the maximum scribbles
        batched_num_scrbs_per_mask = None
        if 'num_scrbs_per_mask' in batched_inputs[0]:
            batched_num_scrbs_per_mask = []
            scribbles = []
            fg_scribbles_count = []
            for x in batched_inputs:
                batched_num_scrbs_per_mask.append(x['num_scrbs_per_mask'])
                # fg_scribbles_count.append(x["instances"].gt_masks.shape[0])
                fg_scribbles_count.append(len(x['num_scrbs_per_mask']))
                scribbles_per_image = []
                # split_fg = torch.split(x['fg_scrbs'], x['num_scrbs_per_mask'], dim=0)
                for scrbs_per_mask in x['fg_scrbs']:
                    scribbles_per_image.append(prepare_scribbles(scrbs_per_mask.to(device = images.device),images))
                if x['bg_scrbs']  is not None:
                    scribbles_per_image.append(prepare_scribbles(x['bg_scrbs'][0].to(device = images.device),images))
                else:
                    scribbles_per_image.append(x['bg_scrbs'])
                scribbles.append(scribbles_per_image)
            return images, scribbles, fg_scribbles_count, batched_num_scrbs_per_mask
            
            
        scribbles_count = []
        fg_scribbles_count =[]
        for x in batched_inputs:
            assert 'scrbs_count' in x, "Expected atleast one fg scribble"
            scribbles_count.append(x["scrbs_count"])
            # fg_scribbles_count.append(x["fg_scrbs"].shape[0])
            fg_scribbles_count.append(x["fg_scrbs"].shape[0])

        if random_bg_queries:
            scribbles = []
            for x in batched_inputs:
                if x['bg_scrbs'] is not None:
                    scribbles_per_image = torch.cat([x['fg_scrbs'], x['bg_scrbs']],0).to(torch.float).to(device = images.device)
                else:
                    scribbles_per_image = x['fg_scrbs'].to(torch.float).to(device = images.device)
                scribbles_per_image = prepare_scribbles(scribbles_per_image,images)
                scribbles.append(scribbles_per_image)
        else:
            max_scribbles = max(scribbles_count)
            
            scribbles = []
            for x in batched_inputs:
                if x['bg_scrbs'] is None:
                    x['bg_scrbs'] = torch.zeros_like(x['fg_scrbs'][0]).unsqueeze(0)
                scribbles_per_image = torch.cat([x['fg_scrbs'], x['bg_scrbs']],0)
                if x['scrbs_count'] < max_scribbles:
                    padding_scribbles = get_padded_scribbles(x['bg_scrbs'], max_scribbles-x['scrbs_count'])
                    scribbles_per_image = torch.cat([scribbles_per_image, padding_scribbles],0)
                scribbles_per_image = prepare_scribbles(scribbles_per_image,images)
                scribbles.append(scribbles_per_image)

            scribbles = torch.stack(scribbles,0).to(torch.float).to(device = images.device)
        return images, scribbles, fg_scribbles_count

def get_padded_scribbles(bg_scribbles, count):
    bg_scrbs_combined = torch.zeros((bg_scribbles.shape[-2], bg_scribbles.shape[-1]), dtype=bg_scribbles.dtype, device=bg_scribbles.device).int()
    for scribble in bg_scribbles:
        bg_scrbs_combined = bg_scrbs_combined | scribble

    bg_scrbs_pad =  [bg_scrbs_combined]*count
    return torch.stack(bg_scrbs_pad,0)

def prepare_scribbles(scribbles,images):
    h_pad, w_pad = images.tensor.shape[-2:]
    padded_scribbles = torch.zeros((scribbles.shape[0],h_pad, w_pad), dtype=scribbles.dtype, device=scribbles.device)
    padded_scribbles[:, : scribbles.shape[1], : scribbles.shape[2]] = scribbles
    return padded_scribbles
           
def get_new_scribbles(gt_data, pred_output, prev_scribbles):

    return prev_scribbles

def get_new_scribbles_opt(targets, pred_output, prev_scribbles, random_bg_queries=False):
    # scibbles [bs, Q, h_gt, w_gt]
    device = prev_scribbles[0].device

    # OPTIMIZATION
    # directly take targets as input as they are already on the device
    gt_masks_batch= [x['instances'].gt_masks.to(device) for x in targets]
    pred_masks_batch = [x["instances"].pred_masks for x in pred_output]

    _, h_gt, w_gt = gt_masks_batch[0].shape
    # # bs_queries, h, w =  pred_masks_batch[0].shape
    # # import torchvision
    t =  torchvision.transforms.Resize(size = (h_gt,w_gt))
    pred_masks_batch = [t(pred_mask) for pred_mask in pred_masks_batch]

    new_scribbles = []
    for i, (gt_masks_per_image, pred_masks_per_image) in enumerate(zip(gt_masks_batch,pred_masks_batch)):
        new_scrbs_per_image = prev_scribbles[i]
        full_bg_mask = 1 - torch.max(gt_masks_per_image,dim=0).values
        full_bg_mask = full_bg_mask.to(device)
        indices = compute_iou(gt_masks_per_image,pred_masks_per_image)
        for j in indices:
            corrective_scrbs, is_fg = get_iterative_scribbles(pred_masks_per_image[j], gt_masks_per_image[j], full_bg_mask, device)
            
            if is_fg:
                new_scrbs_per_image[j] = torch.logical_or(corrective_scrbs[0], new_scrbs_per_image[j])
            else:
                if targets[i]['bg_scrbs'] is not None:
                    new_scrbs_per_image[-1] = torch.logical_or(corrective_scrbs[0], new_scrbs_per_image[-1])
                else:
                    new_scrbs_per_image = torch.cat((new_scrbs_per_image, corrective_scrbs[0].unsqueeze(0)))
        new_scribbles.append(new_scrbs_per_image)
    if random_bg_queries:
        new_scribbles= [s.to(device = device, dtype= prev_scribbles[0].dtype) for s in new_scribbles]
    else:
        new_scribbles = torch.stack(new_scribbles,0)
        new_scribbles = new_scribbles.to(device = device, dtype= prev_scribbles.dtype)        
    return new_scribbles

def get_new_points(targets, pred_output, prev_scribbles, radius = 8, random_bg_queries=False, batched_num_scrbs_per_mask=None):
    # scibbles [bs, Q, h_gt, w_gt]
    device = prev_scribbles[0][0].device

    gt_masks_batch= [x['instances'].gt_masks.to(dtype = torch.uint8, device=device) for x in targets]
    pred_masks_batch = [x["instances"].pred_masks.to(dtype = torch.uint8) for x in pred_output]

    # gt_masks_batch= [x.gt_masks for x in gt_instances]
    # pred_masks_batch = [x.pred_masks for x in pred_instances]
    
    _, h_gt, w_gt = gt_masks_batch[0].shape
    # # bs_queries, h, w =  pred_masks_batch[0].shape
    # # import torchvision
    t =  torchvision.transforms.Resize(size = (h_gt,w_gt))
    pred_masks_batch = [t(pred_mask) for pred_mask in pred_masks_batch]

    new_scribbles = []
    for i, (gt_masks_per_image, pred_masks_per_image) in enumerate(zip(gt_masks_batch,pred_masks_batch)):
        # print(gt_masks_per_image.shape)
        # print(pred_masks_per_image.shape)
        new_scrbs_per_image = copy.deepcopy(prev_scribbles[i])
        full_bg_mask = 1 - torch.max(gt_masks_per_image,dim=0).values
        full_bg_mask= full_bg_mask.to(device)
        indices = compute_iou(gt_masks_per_image,pred_masks_per_image)
        # print("indices:", indices)
        num_masks_image = gt_masks_per_image.shape[0]
        for j in indices:
            corrective_scrbs, is_fg = get_corrective_points(pred_masks_per_image[j], gt_masks_per_image[j],
                                                            full_bg_mask, device, radius, max_num_points=2)
            
            if is_fg:
                new_scrbs_per_image[j] = torch.logical_or(corrective_scrbs[0], new_scrbs_per_image[j])    
            else:
                # if targets[i]['bg_scrbs'] is not None:
                if new_scrbs_per_image.shape[0] > num_masks_image:
                    new_scrbs_per_image = torch.cat((new_scrbs_per_image[:num_masks_image],corrective_scrbs[0].unsqueeze(0), new_scrbs_per_image[num_masks_image:]))
                    new_scrbs_per_image[-1] = torch.logical_or(corrective_scrbs[0], new_scrbs_per_image[-1])
                else:
                    new_scrbs_per_image = torch.cat((new_scrbs_per_image, corrective_scrbs[0].unsqueeze(0)))
                
        new_scribbles.append(new_scrbs_per_image)

    if random_bg_queries:
        new_scribbles= [s.to(device = device, dtype= prev_scribbles[0].dtype) for s in new_scribbles]
    else:
        new_scribbles = torch.stack(new_scribbles,0)
        new_scribbles = new_scribbles.to(device = device, dtype= prev_scribbles.dtype)        
    return new_scribbles

def get_new_points_mq(targets, pred_output, prev_scribbles, radius = 8, random_bg_queries=False, batched_num_scrbs_per_mask=None):
    # scibbles [bs, Q, h_gt, w_gt]
    device = prev_scribbles[0][0].device

    # OPTIMIZATION
    # directly take targets as input as they are already on the device
    gt_masks_batch= [x['instances'].gt_masks.to(dtype = torch.uint8, device=device) for x in targets]
    pred_masks_batch = [x["instances"].pred_masks.to(dtype = torch.uint8) for x in pred_output]
    padding_masks_batch = [x['padding_mask'].to(device=device) for x in targets]

    # new_scribbles = []
    for i, (gt_masks_per_image, pred_masks_per_image, padding_mask) in enumerate(zip(gt_masks_batch,pred_masks_batch, padding_masks_batch)):
        # print(gt_masks_per_image.shape)
        # print(pred_masks_per_image.shape)
        comb_pred_mask = torch.max(pred_masks_per_image,dim=0).values.to(dtype=torch.bool)
        
        # new_scrbs_per_image = copy.deepcopy(prev_scribbles[i])
        full_fg_mask = torch.max(gt_masks_per_image,dim=0).values.to(device)
        # full_bg_mask = 1 - full_fg_mask
        full_bg_mask = torch.logical_not(torch.logical_or(padding_mask, full_fg_mask)).to(dtype=torch.uint8)
        # full_bg_mask= full_bg_mask
        all_fp = torch.logical_and(full_bg_mask, comb_pred_mask).to(dtype=torch.uint8)
        all_fn = torch.logical_and(torch.logical_not(comb_pred_mask), full_fg_mask).to(dtype=torch.uint8)

        if all_fp.sum() > all_fn.sum():
            corrective_bg_point = get_next_points_bg(all_fp,device)
            if prev_scribbles[i][-1] is None:
                prev_scribbles[i][-1] = corrective_bg_point
            else:
                prev_scribbles[i][-1] = torch.cat([prev_scribbles[i][-1], corrective_bg_point],0)
        else:
            fn_per_object = get_fn_per_object(pred_masks_per_image,gt_masks_per_image)
            worst=5
            indxs = torch.topk(torch.tensor(fn_per_object), min(worst, len(fn_per_object)),largest=True).indices
            for j in indxs:
                corrective_fg_point = get_next_points_fg(pred_masks_per_image[j], gt_masks_per_image[j],device)
                prev_scribbles[i][j] = torch.cat([prev_scribbles[i][j],corrective_fg_point],0)
                batched_num_scrbs_per_mask[i][j]+=corrective_fg_point.shape[0]
 
    return prev_scribbles, batched_num_scrbs_per_mask   

def get_new_points_mq_per_obj(targets, pred_output, prev_scribbles, radius = 8, random_bg_queries=False, batched_num_scrbs_per_mask=None):
    # scibbles [bs, Q, h_gt, w_gt]
    device = prev_scribbles[0][0].device

    # OPTIMIZATION
    # directly take targets as input as they are already on the device
    gt_masks_batch= [x['instances'].gt_masks.to(dtype = torch.uint8, device=device) for x in targets]
    pred_masks_batch = [x["instances"].pred_masks.to(dtype = torch.uint8) for x in pred_output]
    padding_masks_batch = [x['padding_mask'].to(device=device) for x in targets]

    # new_scribbles = []
    for i, (gt_masks_per_image, pred_masks_per_image, padding_mask) in enumerate(zip(gt_masks_batch,pred_masks_batch, padding_masks_batch)):
        # print(gt_masks_per_image.shape)
        # print(pred_masks_per_image.shape)
        indices = compute_iou(gt_masks_per_image,pred_masks_per_image)
        full_fg_mask = torch.max(gt_masks_per_image,dim=0).values.to(device)
        # full_bg_mask = 1 - full_fg_mask
        full_bg_mask = torch.logical_not(torch.logical_or(padding_mask, full_fg_mask)).to(dtype=torch.uint8)
        for j in indices:
            corrective_scrbs, is_fg = get_corrective_points_mq(pred_masks_per_image[j], gt_masks_per_image[j],
                                                            full_bg_mask, device, radius, max_num_points=2)
            
            if is_fg:
                prev_scribbles[i][j] = torch.cat([prev_scribbles[i][j],corrective_scrbs],0)
                batched_num_scrbs_per_mask[i][j]+=corrective_scrbs.shape[0]    
            else:
                if prev_scribbles[i][-1] is None:
                    prev_scribbles[i][-1] = corrective_scrbs
                else:
                    prev_scribbles[i][-1] = torch.cat([prev_scribbles[i][-1], corrective_scrbs],0)
 
    return prev_scribbles, batched_num_scrbs_per_mask   

def compute_iou(gt_masks, pred_masks, worst=5):

    intersections = torch.sum(torch.logical_and(gt_masks, pred_masks), (1,2))
    unions = torch.sum(torch.logical_or(gt_masks,pred_masks), (1,2))
    ious = intersections/unions
    # print(ious)
    return torch.topk(ious, min(worst, len(ious)),largest=False).indices

@torch.jit.script
def mask_iou(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
) -> torch.Tensor:

    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """

    N, H, W = gt_masks.shape
    M, H, W = pred_masks.shape

    gt_masks = gt_masks.view(N, H*W)
    pred_masks = pred_masks.view(M, H*W)

    intersection = torch.matmul(gt_masks, pred_masks.t())

    area1 = gt_masks.sum(dim=1).view(1, -1)
    area2 = pred_masks.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0., device=gt_masks.device),
        intersection / union,
    )

    return ret

def merge_dicts(dict1, dict2):
    merged_dict = {}
    for key in dict1:
        if key in dict2:
            merged_dict[key] = dict1[key] + dict2[key]
        else:
            merged_dict[key] = dict1[key]
    
    for key in dict2:
        if key not in merged_dict:
            merged_dict[key] = dict2[key]
    return merged_dict

def get_fn_per_object(pred_masks, gt_masks):
    fn_per_object = []
    for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks,gt_masks)):
        pred_mask = pred_mask>0.5
        fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).sum()
        fn_per_object.append(fn)
    return fn_per_object

def get_next_points_fg(gt_mask, pred_mask, device, max_num_pts=1, radius_size=8):

    gt_mask = np.asarray(gt_mask.cpu(), dtype = np.bool_)
    pred_mask = np.asarray(pred_mask.cpu(), dtype = np.bool_)
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    
    H, W = gt_mask.shape
    if fn_mask.sum()==0:
        fn_mask = gt_mask

    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    k=2
    candidates = np.argwhere(fn_mask_dt > (fn_mask_dt.max()/2))
    num_pts = np.random.randint(1,max_num_pts+1)
    num_pts = min(candidates.shape[0], num_pts)
    indices = random.sample(range(candidates.shape[0]),num_pts)
    sample_locations = candidates[indices]
    
    _pm = np.zeros_like(gt_mask)
    if sample_locations.shape[0] > 0:
        # all_points_per_mask = np.zeros_like(_m)
        points_per_mask = []
        for loc in sample_locations:
            _pm = create_circular_mask(H, W, centers=[loc], radius=radius_size)
            # all_points_per_mask = np.logical_or(all_points_per_mask, _pm)
            points_per_mask.append(_pm)
        return torch.from_numpy(np.stack(points_per_mask, axis=0)).to(device=device, dtype=torch.float)
    else:
        return torch.from_numpy(np.stack([_pm], axis=0)).to(device=device, dtype=torch.float)

def get_next_points_bg(all_fp, device, max_num_pts=1, radius_size=8):

    all_fp = np.asarray(all_fp.cpu(), dtype = np.bool_)
    # pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    # fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    
    H, W = all_fp.shape

    fp_mask = np.pad(all_fp, ((1, 1), (1, 1)), 'constant')
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    k=2
    candidates = np.argwhere(fp_mask_dt > (fp_mask_dt.max()/2))
    num_pts = np.random.randint(1,max_num_pts+1)
    num_pts = min(candidates.shape[0], num_pts)
    indices = random.sample(range(candidates.shape[0]),num_pts)
    sample_locations = candidates[indices]
    
    _pm = np.zeros_like(fp_mask_dt)
    if sample_locations.shape[0] > 0:
        # all_points_per_mask = np.zeros_like(_m)
        bg_points = []
        for loc in sample_locations:
            _pm = create_circular_mask(H, W, centers=[loc], radius=radius_size)
            # all_points_per_mask = np.logical_or(all_points_per_mask, _pm)
            bg_points.append(_pm)
        return torch.from_numpy(np.stack(bg_points, axis=0)).to(device=device, dtype=torch.float)
    else:
        return torch.from_numpy(np.stack([_pm], axis=0)).to(device=device, dtype=torch.float)

def get_corrective_points_mq(pred_mask, gt_mask, bg_mask, device, radius=8, max_num_points=2):
    
    # pred_mask = (pred_mask*255) > 128
    # gt_mask = (gt_mask*255) > 128
    pred_mask = pred_mask>0.5
    gt_mask = gt_mask > 0.5

    # torch functionalities
    fp = torch.logical_and(pred_mask, torch.logical_not(gt_mask)).to(dtype=torch.uint8)
    fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).to(dtype=torch.uint8)

    is_fg = True
    if torch.sum(fn) > torch.sum(fp):
        error_list = [fn]
    else:
        fp = torch.logical_and(fp, bg_mask).to(dtype=torch.uint8)
        error_list = [fp]
        is_fg=False
    H, W = pred_mask.shape
   
    # processing tensors
    scribbles = []
    for m in error_list:
        if torch.nonzero(m).shape[0] == 0:
            if is_fg:
                scribbles.append(m)
            else:
                m = m.cpu()
                sample_locations = point_candidates_dt(np.asarray(bg_mask.to('cpu')).astype(np.uint8), max_num_pts=max_num_points)
                _pm = np.zeros_like(m)
                if sample_locations.shape[0] > 0:
                    # all_points_per_mask = np.zeros_like(_m)
                    points_per_mask = []
                    for loc in sample_locations:
                        _pm = create_circular_mask(H, W, centers=[loc], radius=radius)
                        # all_points_per_mask = np.logical_or(all_points_per_mask, _pm)
                        points_per_mask.append(_pm)
                    return torch.from_numpy(np.stack(points_per_mask, axis=0)).to(device=device, dtype=torch.float), is_fg
                else:
                    return torch.from_numpy(np.stack([_pm], axis=0)).to(device=device, dtype=torch.float), is_fg
        else:
            m = m.cpu()
            sample_locations = point_candidates_dt(np.asarray(m), max_num_pts=max_num_points)
            _pm = np.zeros_like(m)
            if sample_locations.shape[0] > 0:
                # all_points_per_mask = np.zeros_like(_m)
                points_per_mask = []
                for loc in sample_locations:
                    _pm = create_circular_mask(H, W, centers=[loc], radius=radius)
                    # all_points_per_mask = np.logical_or(all_points_per_mask, _pm)
                    points_per_mask.append(_pm)
                return torch.from_numpy(np.stack(points_per_mask, axis=0)).to(device=device, dtype=torch.float), is_fg
            else:
                return torch.from_numpy(np.stack([_pm], axis=0)).to(device=device, dtype=torch.float), is_fg
   