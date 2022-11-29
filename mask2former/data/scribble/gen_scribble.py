import numpy as np
import cv2
import bezier
import torch
# from .tamed_robot import TamedRobot
from .mask_perturb import random_erode


def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def get_boundary_scribble(region):
    # Draw along the boundary of an error region
    erode_size = np.random.randint(3, 50)
    eroded = cv2.erode(region, disk_kernel(erode_size))
    scribble = cv2.morphologyEx(eroded, cv2.MORPH_GRADIENT, np.ones((3,3)))

    h, w = region.shape
    for _ in range(4):
        lx, ly = np.random.randint(w), np.random.randint(h)
        lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)
        scribble[ly:lh, lx:lw] = random_erode(scribble[ly:lh, lx:lw], min=5)

    return scribble

def get_curve_scribble(region, min_srb=1, max_srb=2, sort=True):
    # Draw random curves
    num_lines = np.random.randint(min_srb, max_srb)

    scribbles = []
    lengths = []
    eval_pts = np.linspace(0.0, 1.0, 1024)
    if sort:
        # Generate more anyway, pick the best k at last
        num_gen = 10
    else:
        num_gen = num_lines
    for _ in range(num_gen):
        region_indices = np.argwhere(region)
        try:
            include_idx = np.random.choice(region_indices.shape[0], size=3, replace=False)
        except ValueError:
            include_idx = np.random.choice(region_indices.shape[0], size=3, replace=True)
        y_nodes = np.asfortranarray([
            [0.0, 0.5, 1.0],
            region_indices[include_idx, 0],
        ])
        x_nodes = np.asfortranarray([
            [0.0, 0.5, 1.0],
            region_indices[include_idx, 1],
        ])
        x_curve = bezier.Curve(x_nodes, degree=2)
        y_curve = bezier.Curve(y_nodes, degree=2)
        x_pts = x_curve.evaluate_multi(eval_pts)
        y_pts = y_curve.evaluate_multi(eval_pts)

        this_scribble = np.zeros_like(region)
        pts = np.stack([x_pts[1,:], y_pts[1,:]], 1)
        pts = pts.reshape((-1, 1, 2)).astype(np.int32)
        this_scribble = cv2.polylines(this_scribble, [pts], isClosed=False, color=(1), thickness=3)

        # Mask away path outside the allowed region, allow some error in labeling
        # allowed_error = np.random.randint(3, 7)
        # allowed_region = cv2.dilate(region, disk_kernel(allowed_error))
        # this_scribble = this_scribble * allowed_region
        this_scribble = this_scribble * region

        scribbles.append(this_scribble)
        lengths.append(this_scribble.sum())

    # Sort according to length, we want the long lines
    scribbles = [x for _, x in sorted(zip(lengths, scribbles), key=lambda pair: pair[0], reverse=True)]
    scribble = sum(scribbles[:num_lines])

    return (scribble>0.5).astype(np.uint8)

def get_thinned_scribble(region):
    # Use the thinning algorithm for scribbles
    thinned = (cv2.ximgproc.thinning(region*255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)>128).astype(np.uint8)

    scribble = cv2.dilate(thinned, np.ones((3,3)))
    h, w = region.shape
    for _ in range(4):
        lx, ly = np.random.randint(w), np.random.randint(h)
        lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)
        scribble[ly:lh, lx:lw] = random_erode(scribble[ly:lh, lx:lw], min=5)

    return scribble

def get_scribble_gt_mask(gt, bg = False):
    
    gt = gt > 128
    # we just draw scribbles referring only to the GT 

    if bg:
        bg_scribbles = []
        m = (~gt).astype(np.uint8)
        if np.argwhere(m).shape[0] == 0:
            bg_scribbles = [torch.from_numpy(m)]
            return torch.stack(bg_scribbles,0)
        num_bg_scrb = np.random.randint(2, 6)
        while(num_bg_scrb>0):
            
            region_scribble = get_curve_scribble(m, max_srb=2, sort=False)
            
            bg_scribbles.append(torch.from_numpy(region_scribble.astype(np.uint8)))
            num_bg_scrb-=1
        
        bg_scribbles = torch.stack(bg_scribbles,0)
        return bg_scribbles
    else:
        fg_scribbles = np.zeros_like(gt)
        m = gt.astype(np.uint8)
        if np.argwhere(m).shape[0] == 0:
            return m
        pick = np.random.rand()
        if pick < 0.5:
            region_scribble = get_thinned_scribble(m)
        else:
            region_scribble = get_curve_scribble(m)
        fg_scribbles = (fg_scribbles | region_scribble)
            
        # Optionally use a second scribble type
        pick = np.random.rand()
        if pick < 0.5:
            pick = np.random.rand()
            if pick < 0.5:
                region_scribble = get_thinned_scribble(m)
            else:
                region_scribble = get_curve_scribble(m)
            fg_scribbles = (fg_scribbles | region_scribble)

        this_scribble = get_curve_scribble(m, max_srb=2, sort=False)
        fg_scribbles = fg_scribbles | this_scribble

        return fg_scribbles.astype(np.uint8)


def get_iterative_scribbles(pred_mask, gt_mask, bg_mask, device):

    # pred_mask = pred_mask*255
    # gt_mask = gt_mask*255
    
    # pred_mask = pred_mask > 128
    # gt_mask = gt_mask > 128

    pred_mask = (pred_mask*255) > 128
    gt_mask = (gt_mask*255) > 128

    # False positive and false negative
    # fp = (pred_mask & ~gt_mask).astype(np.uint8)
    # fn = (~pred_mask & gt_mask).astype(np.uint8)

    # torch functionalities
    fp = torch.logical_and(pred_mask, torch.logical_not(gt_mask)).to(dtype=torch.uint8)
    fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).to(dtype=torch.uint8)

    # opening_size = np.random.randint(5, 20)
    # fp = cv2.morphologyEx(fp, cv2.MORPH_OPEN, disk_kernel(opening_size))
    # fn = cv2.morphologyEx(fn, cv2.MORPH_OPEN, disk_kernel(opening_size))

    # processing tensors
    # scribbles = []
    # for m in [fn, fp]:
    #     if torch.nonzero(m).shape[0] == 0:
    #         scribbles.append(m)
    #     else:
    #         region_scribble = get_curve_scribble(m.cpu().numpy(),max_srb=2)
    #         scribbles.append(torch.from_numpy(region_scribble).to(device))
    # return torch.stack(scribbles,0)

    is_fg = True
    if torch.sum(fn) > torch.sum(fp):
        error_list= [fn]
    else:
        fp = torch.logical_and(fp, bg_mask).to(dtype=torch.uint8)
        error_list = [fp]
        is_fg= False
    # opening_size = np.random.randint(5, 20)
    # fp = cv2.morphologyEx(fp, cv2.MORPH_OPEN, disk_kernel(opening_size))
    # fn = cv2.morphologyEx(fn, cv2.MORPH_OPEN, disk_kernel(opening_size))

    # processing tensors
    scribbles = []
    for m in error_list:
        if torch.nonzero(m).shape[0] == 0:
            scribbles.append(m)
        else:
            region_scribble = get_curve_scribble(m.cpu().numpy(),max_srb=2)
            scribbles.append(torch.from_numpy(region_scribble).to(device))
    return torch.stack(scribbles,0), is_fg

def get_scribble_gt(gt):
    gt = gt > 128

    scribbles = [np.zeros_like(gt)]*2
    # Sometimes we just draw scribbles referring only to the GT but not the given mask

    for i, m in enumerate([gt.astype(np.uint8), (~gt).astype(np.uint8)]):
        # if m.sum() < 100:
        #     continue
        # Initial pass, pick a scribble type
        if i==0:
            pick = np.random.rand()
            if pick < 0.5:
                region_scribble = get_thinned_scribble(m)
            else:
                region_scribble = get_curve_scribble(m)
            scribbles[i] = (scribbles[i] | region_scribble)
                
            # Optionally use a second scribble type
            pick = np.random.rand()
            if pick < 0.5:
                pick = np.random.rand()
                if pick < 0.5:
                    region_scribble = get_thinned_scribble(m)
                else:
                    region_scribble = get_curve_scribble(m)
                scribbles[i] = (scribbles[i] | region_scribble)

            this_scribble = get_curve_scribble(m, max_srb=5, sort=False)
            scribbles[i] = scribbles[i] | this_scribble

        if i==1:

            count=1
            num_bg_scrb = np.random.randint(2, 6)
            while(num_bg_scrb>0):
                region_scribble = get_curve_scribble(m, max_srb=2, sort=False)
                scribbles[i] = np.maximum(scribbles[i], region_scribble.astype(np.uint8)*count)
                count+=1 
                num_bg_scrb-=1
    
    return scribbles[0].astype(np.uint8), scribbles[1].astype(np.uint8)

def get_scribble_eval(gt, bg = False):
    
    gt = gt > 128
    # we just draw scribbles referring only to the GT 

    if bg:
        bg_scribbles = []
        m = (~gt).astype(np.uint8)
        if np.argwhere(m).shape[0] == 0:
            bg_scribbles = [torch.from_numpy(m)]
            return torch.stack(bg_scribbles,0)
        num_bg_scrb = 1
        while(num_bg_scrb>0):
            
            region_scribble = get_curve_scribble(m, max_srb=2, sort=False)
            
            bg_scribbles.append(torch.from_numpy(region_scribble.astype(np.uint8)))
            num_bg_scrb-=1
        
        bg_scribbles = torch.stack(bg_scribbles,0)
        return bg_scribbles
    else:
        fg_scribbles = np.zeros_like(gt)
        m = gt.astype(np.uint8)
        if np.argwhere(m).shape[0] == 0:
            return m
        
        region_scribble = get_curve_scribble(m)
        fg_scribbles = (fg_scribbles | region_scribble)

        return fg_scribbles.astype(np.uint8)

def get_iterative_eval(pred_mask, gt_mask, bg_mask, device):

    # pred_mask = pred_mask*255
    # gt_mask = gt_mask*255
    
    # pred_mask = pred_mask > 128
    # gt_mask = gt_mask > 128

    pred_mask = (pred_mask*255) > 128
    gt_mask = (gt_mask*255) > 128

    # False positive and false negative
    # fp = (pred_mask & ~gt_mask).astype(np.uint8)
    # fn = (~pred_mask & gt_mask).astype(np.uint8)

    # torch functionalities
    fp = torch.logical_and(pred_mask, torch.logical_not(gt_mask)).to(dtype=torch.uint8)
    fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).to(dtype=torch.uint8)

    # error_list = []
    is_fg = True
    if torch.sum(fn) > torch.sum(fp):
        error_list= [fn]
    else:
        fp = torch.logical_and(fp, bg_mask).to(dtype=torch.uint8)
        error_list = [fp]
        is_fg= False
    # opening_size = np.random.randint(5, 20)
    # fp = cv2.morphologyEx(fp, cv2.MORPH_OPEN, disk_kernel(opening_size))
    # fn = cv2.morphologyEx(fn, cv2.MORPH_OPEN, disk_kernel(opening_size))

    # processing tensors
    scribbles = []
    for m in error_list:
        if torch.nonzero(m).shape[0] == 0:
            scribbles.append(m)
        else:
            region_scribble = get_curve_scribble(m.cpu().numpy(),max_srb=2)
            scribbles.append(torch.from_numpy(region_scribble).to(device))
    return torch.stack(scribbles,0), is_fg
# if __name__ == '__main__':
#     import sys
#     mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
#     gt = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

#     fp_scibble, fn_scibble = get_scribble_gt(mask)

#     cv2.imwrite('fromzero_p.png', fp_scibble*255)
#     cv2.imwrite('fromzero_n.png', fn_scibble*255)