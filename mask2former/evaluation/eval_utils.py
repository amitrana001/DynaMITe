import os
import sys
import numpy as np
import torch, torchvision
import cv2
from detectron2.utils.visualizer import Visualizer
import torchvision.transforms.functional as F

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

def compute_iou(gt_masks, pred_masks, ious, iou_threshold, ignore_masks=None):

    if ignore_masks is None:
        for i in range(len(ious)):
            intersection = (gt_masks[i] * pred_masks[i]).sum()
            union = torch.logical_or(gt_masks[i], pred_masks[i]).to(torch.int).sum()
            if ious[i] < iou_threshold:
                ious[i]= intersection/union
            else:
                ious[i]= max(intersection/union, ious[i])
            # print(ious)
        return ious

    for i in range(len(ious)):
        # intersection = (gt_masks[i] * pred_masks[i]).sum()
        # union = torch.logical_or(gt_masks[i], pred_masks[i]).to(torch.int).sum()
        n_iou = get_iou_per_mask(gt_masks[i], pred_masks[i],ignore_masks[i])
        if ious[i] < iou_threshold:
            ious[i]= n_iou
        else:
            ious[i]= max(n_iou, ious[i])
    # print(ious)
    return ious

def get_iou_per_mask(gt_mask, pred_mask, ignore_mask):
    # ignore_gt_mask_inv = gt_mask != ignore_label
    ignore_gt_mask_inv = ~(ignore_mask.to(dtype=torch.bool))
    # ignore_gt_mask_inv = 
    obj_gt_mask = gt_mask

    intersection = torch.logical_and(torch.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = torch.logical_and(torch.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union

def save_visualization(inputs, pred_masks, scribbles, dir_path, iou, num_iter,  alpha_blend=0.3):
    
    image = np.asarray(inputs['image'].permute(1,2,0))

    visualizer = Visualizer(image, metadata=None)
    pred_masks = F.resize(pred_masks.to(dtype=torch.uint8), image.shape[:2])
    c = []
    for i in range(pred_masks.shape[0]):
        # c.append(color_map[2*(i)+2]/255.0)
        c.append(color_map[i]/255.0)
    # pred_masks = np.asarray(pred_masks).astype(np.bool_)
    vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c, alpha=alpha_blend)
    # [Optional] prepare labels

    image = vis.get_image()
    # # Laminate your image!
    total_colors = len(color_map)-1
    
    h,w = image.shape[:2]
    for i, scrb in enumerate(scribbles[:-1]):
        scrb = torch.max(scrb,0).values.to('cpu')
        scrb = scrb[:h, :w]
        color = np.array(color_map[total_colors-5*i-4], dtype=np.uint8)
        image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
        
    if scribbles[-1] is not None:
        scrb = torch.max(scribbles[-1],0).values.to('cpu')
        scrb = scrb[:h, :w]
        color = np.array([255,0,0], dtype=np.uint8)
        image[scrb>0.5, :] = np.array(color, dtype=np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (inputs["width"],inputs["height"]))
    save_dir = os.path.join(dir_path, str(inputs['image_id']))
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"iter_{num_iter}_{iou}.jpg"), image)

from mask2former.data.points.annotation_generator import point_candidates_dt_determinstic, create_circular_mask
def get_next_click_bg(all_fp, not_clicked_map, radius=5, num_points=1,device='cpu'):
    H, W = all_fp.shape
    # all_fp = np.asarray(all_fp).astype(np.uint8)
    all_fp = np.asarray(all_fp, dtype = np.bool_)
    # print("all_fp:", all_fp.sum())
    # pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    # fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    # # fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask)
    # H, W = gt_mask.shape
    padding=True
    if padding:
        all_fp = np.pad(all_fp, ((1, 1), (1, 1)), 'constant')
        # fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    all_fp_dt = cv2.distanceTransform(all_fp.astype(np.uint8), cv2.DIST_L2, 0)
    # fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    if padding:
        all_fp_dt = all_fp_dt[1:-1, 1:-1]
        # fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    all_fp_dt = all_fp_dt * not_clicked_map
    # fp_mask_dt = fp_mask_dt * not_clicked_map

    all_fp_max_dt = np.max(all_fp_dt)
    # fp_max_dist = np.max(fp_mask_dt)

    # is_positive = fn_max_dist > fp_max_dist
    # if is_positive:
    coords_y, coords_x = np.where(all_fp_dt == all_fp_max_dt)  # coords is [y, x]
    # else:
    #     coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]
    sample_locations = [[coords_y[0], coords_x[0]]]
    # print(sample_locations)
    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    not_clicked_map[coords_y[0], coords_x[0]] = False
    return torch.from_numpy(pm).to(device, dtype = torch.uint8), not_clicked_map

def get_fn_area(pred_masks, gt_masks):
    fn_per_object = []
    for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks,gt_masks)):
        pred_mask = pred_mask>0.5
        fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).sum()
        fn_per_object.append(fn)
    return fn_per_object

def get_next_click_fg(pred_mask, gt_mask, not_clicked_map, radius=5, device='cpu'):

    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    
    if fn_mask.sum()==0:
        fn_mask = gt_mask

    H, W = gt_mask.shape
    
    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    fn_mask_dt = fn_mask_dt * not_clicked_map

    fn_max_dist = np.max(fn_mask_dt)
    coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
    # else:
    #     coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]
    sample_locations = [[coords_y[0], coords_x[0]]]
    # print(sample_locations)
    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    not_clicked_map[coords_y[0], coords_x[0]] = False
    return torch.from_numpy(pm).to(device, dtype = torch.uint8), not_clicked_map

def get_next_click(pred_mask, gt_mask, not_clicked_map, radius=5, device='cpu', ignore_mask=None, padding=True):

    not_ignore_mask = np.logical_not(np.asarray(ignore_mask, dtype=np.bool_))
    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    fn_mask =  np.logical_and(np.logical_and(gt_mask, np.logical_not(pred_mask)), not_ignore_mask)
    fp_mask =  np.logical_and(np.logical_and(np.logical_not(gt_mask), pred_mask), not_ignore_mask)
    
    if fn_mask.sum()==0:
        fn_mask = gt_mask
    # print("fn_mask:",fn_mask.sum())
    # fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask)
    H, W = gt_mask.shape

    if padding:
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    if padding:
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    fn_mask_dt = fn_mask_dt * not_clicked_map
    fp_mask_dt = fp_mask_dt * not_clicked_map

    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    is_positive = fn_max_dist > fp_max_dist

    if is_positive:
        coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
    else:
        coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

    sample_locations = [[coords_y[0], coords_x[0]]]

    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
    not_clicked_map[coords_y[0], coords_x[0]] = False
    return torch.from_numpy(pm).to(device, dtype = torch.float).unsqueeze(0), is_positive, not_clicked_map

def post_process(pred_masks,scribbles,ious=None, iou_threshold = 0.85):
    out = []
    # print(pred_masks.shape)
    # print(scribbles.shape)
    for (pred_mask, points, iou) in zip(pred_masks,scribbles, ious):
        if iou < iou_threshold:
            # opening_size = np.random.randint(5, 20)
            # pred_mask = cv2.morphologyEx(np.asarray(pred_mask), cv2.MORPH_OPEN, disk_kernel(opening_size))
            num_labels, labels_im = cv2.connectedComponents(np.asarray(pred_mask).astype(np.uint8))
            points_comp = labels_im[torch.where(points==1)]
            # print(points_comp)
            vals,counts = np.unique(points_comp, return_counts=True)
            cc_mask = np.zeros_like(labels_im)
            for val in vals:
                cc_mask = np.logical_or(cc_mask, labels_im==val)
            # index = np.argmax(counts)
            # print(vals[index])
            # pred_mask = torch.from_numpy(labels_im==vals[index])

            # pred_mask = pred_mask.to(dtype=torch.uint8)
            pred_mask = torch.from_numpy(cc_mask).to(dtype=torch.uint8)
        out.append(pred_mask)
    return torch.stack(out,0)

def prepare_scribbles(scribbles,images):
    h_pad, w_pad = images.tensor.shape[-2:]
    padded_scribbles = torch.zeros((scribbles.shape[0],h_pad, w_pad), dtype=scribbles.dtype, device=scribbles.device)
    padded_scribbles[:, : scribbles.shape[1], : scribbles.shape[2]] = scribbles
    return padded_scribbles