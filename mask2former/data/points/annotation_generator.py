import numpy as np
import random
import cv2
from PIL import Image
import torch

def get_sampled_locations(sample_locations, img_area, n_pts, D=40):
    d_step = int(D / 2)
    img = np.copy(img_area)
    pts = []
    if len(list(zip(sample_locations[0], sample_locations[
        1]))) == 1:  # bug fix: there are instances with only one-pixel; for example davis-train-miami-surf set
        return [[sample_locations[0][0], sample_locations[1][0]]]
    for click in range(n_pts):
        pixel_samples = list(zip(sample_locations[0], sample_locations[1]))
        if len(pixel_samples) > 1:
            [y, x] = random.sample(pixel_samples, 1)[0]
            pts.append([y, x])

            x_min = max(0, x - d_step)
            x_max = min(x + d_step, img.shape[1])
            y_min = max(0, y - d_step)
            y_max = min(y + d_step, img.shape[0])
            img[y_min:y_max, x_min:x_max] = 0

            sample_locations = np.where(img == 1)

    return pts


def generate_point_masks(masks, max_n_pts, void_label=255, sampling="fixed"):
    """
    :param masks: numpy array of shape N x I x H x W
    :param max_n_pts: maximum number of points
    """
    masks = masks.astype(np.uint8)
    masks[masks == void_label] = 0

    N, I, H, W = masks.shape
    num_instances = N * I
    masks = np.reshape(masks, (num_instances, H, W))
    n_pts = np.random.choice(np.arange(1, max_n_pts), size=num_instances) if sampling == "random" \
        else np.repeat([max_n_pts], repeats=num_instances)

    point_masks = []
    for num_ins, (_m, _pt_count) in enumerate(zip(masks, n_pts)):
        sample_locations = np.where(_m == 1)
        _pm = np.zeros_like(_m)
        if len(sample_locations[0]) > 0:
            sampled_points = get_sampled_locations(sample_locations, _m, _pt_count)
            sampled_points = np.stack(np.array(sampled_points), axis=0)  # n x 2
            _pm[sampled_points[:, 0], sampled_points[:, 1]] = 1
            point_masks.append(_pm)
        else:
            point_masks.append(_pm)

    point_masks = np.reshape(
        np.stack(point_masks, axis=0),
        (N, I, H, W)
    )

    return point_masks

def point_candidates_dt(mask, max_num_pts=3, k=1.7):
    mask = mask.astype(np.uint8)
    # masks[masks == void_label] = 0

    padded_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    candidates = np.argwhere(dt > (dt.max()/k))
    num_pts = np.random.randint(1,max_num_pts+1)
    num_pts = min(candidates.shape[0], num_pts)
    # print(candidates.shape[0], num_pts)
    try:
        indices = random.sample(range(candidates.shape[0]),num_pts)
    except ValueError:
        return candidates[0]
    
    return candidates[indices]

def point_candidates_dt_determinstic(mask, max_num_pts=1, k=1.7):
    mask = mask.astype(np.uint8)
    # masks[masks == void_label] = 0

    padded_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    candidates = np.argwhere(dt == (dt.max()))
    num_pts = np.random.randint(1,max_num_pts+1)
    num_pts = min(candidates.shape[0], num_pts)
    # print(candidates.shape[0], num_pts)
    try:
        indices = random.sample(range(candidates.shape[0]),num_pts)
    except ValueError:
        return candidates[0]
    
    return candidates[indices]

def create_circular_mask(h, w, centers=None, radius=None):

    # if center is None: # use the middle of the image
    #     center = (int(w/2), int(h/2))
    assert centers is not None
    assert radius is not None

    mask=np.zeros((h,w), dtype=bool) 
    for center in centers:

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)

        mask = mask | (dist_from_center <= radius)
    return mask.astype(np.uint8)

def generate_point_to_blob_masks(masks, max_num_points=3, radius_size=8, all_masks=None):

    """
    :param masks: numpy array of shape N x I x H x W
    :param patch_size: size of patch (int)
    """
    # assert all_masks is not None
    masks = np.asarray(masks).astype(np.uint8)
    all_masks = np.asarray(all_masks).astype(np.uint8)
    # masks[masks == void_label] = 0
    
    N, I, H, W = masks.shape
    num_instances = N * I
    masks = np.reshape(masks, (num_instances, H, W))
    point_to_blob_masks = []
    # full_mask = np.zeros((H,W))
    for num_ins, (_m) in enumerate(masks):
        sample_locations = point_candidates_dt(_m, max_num_pts=max_num_points)
        _pm = np.zeros_like(_m)
        all_masks = np.logical_or(all_masks, _m)
        if sample_locations.shape[0] > 0:
            _pm = create_circular_mask(H, W, centers=sample_locations, radius=radius_size)
            point_to_blob_masks.append(_pm)
        else:
            point_to_blob_masks.append(_pm)

    
    # full_bg_mask = np.logical_not(all_masks).astype(int)
    bg = []
    full_bg_mask = (~all_masks).astype(np.uint8)
    _bg = np.zeros_like(full_bg_mask)
    pick = np.random.rand()
    if pick > 0.5:
        sample_locations = point_candidates_dt(full_bg_mask, max_num_pts=max_num_points+1)
    else:
        sample_locations = np.argwhere(full_bg_mask)
        num_pts = np.random.randint(2,max_num_points+3)
        num_pts = min(num_pts, sample_locations.shape[0])
        # try:
        indices = random.sample(range(sample_locations.shape[0]),num_pts)
        # except ValueError:
        #     indices = []
        sample_locations = sample_locations[indices]
    if sample_locations.shape[0] > 0:
        for loc in sample_locations:
            _bg = create_circular_mask(H, W, centers=[loc], radius=radius_size)
            bg.append(_bg)
    else:
        bg.append(_bg)

    masks = np.reshape(
        np.stack(point_to_blob_masks, axis=0),
        (N, I, H, W)
    ) 

    bg = torch.from_numpy(np.stack(bg, axis=0)).to(dtype=torch.uint8)
    # print(np.unique(masks))
    return torch.from_numpy(masks).to(dtype=torch.uint8), bg
    # raise NotImplementedError("Not implemented yet!!!")

def gen_multi_points_per_mask(masks, max_num_points=3, radius_size=8, all_masks=None):

    """
    :param masks: numpy array of shape I x H x W
    :param patch_size: size of patch (int)
    """
    # assert all_masks is not None
    masks = np.asarray(masks).astype(np.uint8)
    all_masks = np.asarray(all_masks) #.astype(np.uint8)
    # masks[masks == void_label] = 0
    
    I, H, W = masks.shape
    num_scrbs_per_mask = [0]*I
    point_to_blob_masks = []
    for i, (_m) in enumerate(masks):
        sample_locations = point_candidates_dt(_m, max_num_pts=max_num_points)
        _pm = np.zeros_like(_m)
        # all_masks = np.logical_or(all_masks, _m)
        if sample_locations.shape[0] > 0:
            # all_points_per_mask = np.zeros_like(_m)
            points_per_mask = []
            for loc in sample_locations:
                _pm = create_circular_mask(H, W, centers=[loc], radius=radius_size)
                # all_points_per_mask = np.logical_or(all_points_per_mask, _pm)
                points_per_mask.append(_pm)
                num_scrbs_per_mask[i]+=1
            point_to_blob_masks.append(torch.from_numpy(np.stack(points_per_mask, axis=0)).to(torch.float))
            # point_to_blob_masks.insert(0,all_points_per_mask)
            # num_scrbs_per_mask[i]+=1
        else:
            return None
            print("no points")
            point_to_blob_masks.append(_pm)
            num_scrbs_per_mask[i]+=1

    # full_bg_mask = np.logical_not(all_masks).astype(int)
    bg = []
    full_bg_mask = (~all_masks).astype(np.uint8)
    _bg = np.zeros_like(full_bg_mask)
    pick = np.random.rand()
    if pick > 0.5:
        sample_locations = point_candidates_dt(full_bg_mask, max_num_pts=max_num_points+1)
    else:
        sample_locations = np.argwhere(full_bg_mask)
        num_pts = np.random.randint(2,max_num_points+3)
        num_pts = min(num_pts, sample_locations.shape[0])
        # try:
        indices = random.sample(range(sample_locations.shape[0]),num_pts)

        sample_locations = sample_locations[indices]
    if sample_locations.shape[0] > 0:
        # all_points_for_bg =  np.zeros_like(full_bg_mask)
        points_for_bg = []
        for loc in sample_locations:
            _bg = create_circular_mask(H, W, centers=[loc], radius=radius_size)
            points_for_bg.append(_bg)
            # all_points_for_bg = np.logical_or(all_points_for_bg, _bg)
        bg.append(torch.from_numpy(np.stack(points_for_bg, axis=0)).to(torch.float))
        # bg.insert(0,all_points_for_bg)
    else:
        return None
        bg.append(_bg)

    # point_to_blob_masks = np.stack(point_to_blob_masks, axis=0)
    # point_to_blob_masks = torch.from_numpy(point_to_blob_masks).to(dtype=torch.uint8)

    # bg = torch.from_numpy(np.stack(bg, axis=0)).to(dtype=torch.uint8)
    # print(np.unique(masks))
    return point_to_blob_masks, bg, num_scrbs_per_mask
    # raise NotImplementedError("Not implemented yet!!!")

def generate_point_to_blob_masks_eval(masks, max_num_points=3, radius_size=8, all_masks=None):

    """
    :param masks: numpy array of shape N x I x H x W
    :param patch_size: size of patch (int)
    """
    # assert all_masks is not None
    masks = np.asarray(masks).astype(np.uint8)
    all_masks = np.asarray(all_masks).astype(np.uint8)
    # masks[masks == void_label] = 0
    
    N, I, H, W = masks.shape
    num_instances = N * I
    masks = np.reshape(masks, (num_instances, H, W))
    point_to_blob_masks = []
    # full_mask = np.zeros((H,W))
    for num_ins, (_m) in enumerate(masks):
        sample_locations = point_candidates_dt(_m, max_num_pts=max_num_points)
        _pm = np.zeros_like(_m)
        all_masks = np.logical_or(all_masks, _m)
        if sample_locations.shape[0] > 0:
            _pm = create_circular_mask(H, W, centers=sample_locations, radius=radius_size)
            point_to_blob_masks.append(_pm)
        else:
            point_to_blob_masks.append(_pm)

    
    # full_bg_mask = np.logical_not(all_masks).astype(int)
    bg = []
    full_bg_mask = (~all_masks).astype(np.uint8)
    _bg = np.zeros_like(full_bg_mask)
    pick = np.random.rand()
    if pick > 0.5:
        sample_locations = point_candidates_dt(full_bg_mask, max_num_pts=max_num_points)
    else:
        sample_locations = np.argwhere(full_bg_mask)
        num_pts = np.random.randint(1,max_num_points+1)
        num_pts = min(num_pts, sample_locations.shape[0])
        # try:
        indices = random.sample(range(sample_locations.shape[0]),num_pts)
        # except ValueError:
        #     indices = []
        sample_locations = sample_locations[indices]
    if sample_locations.shape[0] > 0:
        for loc in sample_locations:
            _bg = create_circular_mask(H, W, centers=[loc], radius=radius_size)
            bg.append(_bg)
    else:
        bg.append(_bg)

    masks = np.reshape(
        np.stack(point_to_blob_masks, axis=0),
        (N, I, H, W)
    )

    bg = torch.from_numpy(np.stack(bg, axis=0)).to(dtype=torch.uint8)
    # print(np.unique(masks))
    return torch.from_numpy(masks).to(dtype=torch.uint8), bg
    # raise NotImplementedError("Not implemented yet!!!")


def generate_point_to_blob_masks_eval_deterministic(gt_masks, max_num_points=1, radius_size=8, all_masks=None):

    """
    :param masks: numpy array of shape I x H x W
    :param patch_size: size of patch (int)
    """
    gt_masks = np.asarray(gt_masks).astype(np.uint8)
    
    I, H, W = gt_masks.shape
    num_scrbs_per_mask = [0]*I
    point_to_blob_masks = []
    for i, (_m) in enumerate(gt_masks):
        sample_locations = point_candidates_dt_determinstic(_m, max_num_pts=max_num_points)
        _pm = np.zeros_like(_m)
        # all_masks = np.logical_or(all_masks, _m)
        if sample_locations.shape[0] > 0:
            # all_points_per_mask = np.zeros_like(_m)
            points_per_mask = []
            for loc in sample_locations:
                _pm = create_circular_mask(H, W, centers=[loc], radius=radius_size)
                # all_points_per_mask = np.logical_or(all_points_per_mask, _pm)
                points_per_mask.append(_pm)
                num_scrbs_per_mask[i]+=1
            point_to_blob_masks.append(torch.from_numpy(np.stack(points_per_mask, axis=0)).to(torch.float))

    return point_to_blob_masks, num_scrbs_per_mask

def get_iterative_points(pred_mask, gt_mask, device):

    pred_mask = (pred_mask*255) > 128
    gt_mask = (gt_mask*255) > 128

    # torch functionalities
    fp = torch.logical_and(pred_mask, torch.logical_not(gt_mask)).to(dtype=torch.uint8)
    fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).to(dtype=torch.uint8)

    H, W = pred_mask.shape
   
    # processing tensors
    scribbles = []
    for m in [fn, fp]:
        if torch.nonzero(m).shape[0] == 0:
            scribbles.append(m)
        else:
            m = m.cpu()
            sample_locations = point_candidates_dt(np.asarray(m), max_num_pts=2)
            pm = np.zeros_like(m)
            if sample_locations.shape[0] > 0:
                pm = create_circular_mask(H, W, centers=sample_locations, radius=8)
            scribbles.append(torch.from_numpy(pm).to(device))
    return torch.stack(scribbles,0)

def get_corrective_points(pred_mask, gt_mask, bg_mask, device, radius=8, max_num_points=2):
    
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
                pm = np.zeros_like(m)
                if sample_locations.shape[0] > 0:
                    pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
                scribbles.append(torch.from_numpy(pm).to(device,dtype = torch.uint8))
        else:
            m = m.cpu()
            sample_locations = point_candidates_dt(np.asarray(m), max_num_pts=max_num_points)
            pm = np.zeros_like(m)
            if sample_locations.shape[0] > 0:
                pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
            scribbles.append(torch.from_numpy(pm).to(device,dtype = torch.uint8))
    return torch.stack(scribbles,0), is_fg

def get_corrective_points_determinstic(pred_mask, gt_mask, bg_mask, device, fg_points=None, radius=8):
    
    # pred_mask = (pred_mask*255) > 128
    # gt_mask = (gt_mask*255) > 128
    pred_mask = pred_mask>0.5
    gt_mask = gt_mask > 0.5

    # torch functionalities
    
    fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).to(dtype=torch.uint8)
    if fg_points is not None:
        fn = torch.logical_and(fn, ~(fg_points.to(dtype=torch.bool))).to(dtype=torch.uint8)

    fp = torch.logical_and(pred_mask, torch.logical_not(gt_mask)).to(dtype=torch.uint8)
    fp = torch.logical_and(fp, bg_mask).to(dtype=torch.uint8)


    padded_mask_fn = np.pad(np.asarray(fn), ((1, 1), (1, 1)), 'constant')
    fn_mask_dt = cv2.distanceTransform(padded_mask_fn.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    

    padded_mask_fp = np.pad(np.asarray(fp), ((1, 1), (1, 1)), 'constant')
    fp_mask_dt = cv2.distanceTransform(padded_mask_fp.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    # fn_mask_dt = cv2.distanceTransform(np.asarray(fn).astype(np.uint8), cv2.DIST_L2, 0)
    # fp_mask_dt = cv2.distanceTransform(np.asarray(fp).astype(np.uint8), cv2.DIST_L2, 0)

    H, W = pred_mask.shape
    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    is_fg = True
    if fn_max_dist > fp_max_dist:
        error_list = [fn]
    else:
        # fp = torch.logical_and(fp, bg_mask).to(dtype=torch.uint8)
        error_list = [fp]
        is_fg=False

    # is_fg = True
    # if torch.sum(fn) > torch.sum(fp):
    #     error_list = [fn]
    # else:
    #     fp = torch.logical_and(fp, bg_mask).to(dtype=torch.uint8)
    #     error_list = [fp]
    #     is_fg=False
    # H, W = pred_mask.shape
   
    # processing tensors
    scribbles = []
    for m in error_list:
        if torch.nonzero(m).shape[0] == 0:
            # print("here")
            scribbles.append(m)
        else:
            m = m.cpu()
            sample_locations = point_candidates_dt_determinstic(np.asarray(m), max_num_pts=1)
            pm = np.zeros_like(m)
            if sample_locations.shape[0] > 0:
                pm = create_circular_mask(H, W, centers=sample_locations, radius=radius)
            scribbles.append(torch.from_numpy(pm).to(device,dtype = torch.uint8))
    return torch.stack(scribbles,0), is_fg


def get_next_click(pred_mask, gt_mask, bg_mask, device):

    pred_mask = pred_mask>0.5
    gt_mask = gt_mask > 0.5

    # torch functionalities
    fp = torch.logical_and(pred_mask, torch.logical_not(gt_mask)).to(dtype=torch.uint8)

    # fp = torch.logical_and(fp, bg_mask).to(dtype=torch.uint8)
    fn = torch.logical_and(torch.logical_not(pred_mask), gt_mask).to(dtype=torch.uint8)

    # fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
    # fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)
    # padding = True
    # if padding:
    #     fn = np.pad(fn, ((1, 1), (1, 1)), 'constant')
    #     fp = np.pad(fp, ((1, 1), (1, 1)), 'constant')

    fn_mask_dt = cv2.distanceTransform(np.asarray(fn).astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(np.asarray(fp).astype(np.uint8), cv2.DIST_L2, 0)

    # if padding:
    #     fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
    #     fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    # fn_mask_dt = fn_mask_dt * self.not_clicked_map
    # fp_mask_dt = fp_mask_dt * self.not_clicked_map
    H, W = pred_mask.shape
    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    is_fg = fn_max_dist > fp_max_dist
    scribbles = []
    if is_fg:
        sample_locations = np.where(fn_mask_dt == fn_max_dist)
        sample_locations = [[sample_locations[0][0],sample_locations[1][0]]]
        print(sample_locations)
        pm = create_circular_mask(H, W, centers=sample_locations, radius=8)
          # coords is [y, x]
    else:
        sample_locations = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]
        sample_locations = [[sample_locations[0][0],sample_locations[1][0]]]
        print(sample_locations)
        pm = create_circular_mask(H, W, centers=sample_locations, radius=8)
    scribbles.append(torch.from_numpy(pm).to(device,dtype = torch.uint8))
    return torch.stack(scribbles,0), is_fg
