import torch
import cv2
import os
import torchvision
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
import torchvision.transforms.functional as F

def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels


def save_gt_segm(batch, indx = 0, display=True, data_name = "coco_2017_val",text=None):
    indx = min(indx, len(batch))
    img = batch[indx]
    inst = img["instances"]
    image = np.asarray(img["image"].permute(1,2,0))
    # masks = torch.stack([inst.gt_masks, img["pos_scrb"]], 0)
    # print(image.shape)
    # print(f"num of instnaces:{inst.gt_masks.shape[0]}")
    # print(f"num of fg scribbles:{img['fg_scrbs'].shape[0]}")
    # print(f"num of bg scribbles:{img['bg_scrbs'].shape[0]}")
    # print(f"Total scibbles: {img['scrbs_count']}")
    if img['fg_scrbs'] is not None:
        for scrb in img["fg_scrbs"]:
            color = [np.random.randint(0, 255), np.random.randint(0, 1), np.random.randint(0, 255)]
            image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
    if img['bg_scrbs'] is not None:
        for scrb in img["bg_scrbs"]:
            color = [np.random.randint(0, 255), np.random.randint(0, 1), np.random.randint(0, 255)]
            image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
    # print(image[:, :, ::-1].shape)
    metadata=MetadataCatalog.get(data_name)
    visualizer = Visualizer(image, metadata)
    # labels = _create_text_labels(inst.gt_classes, scores=None, class_names = metadata.get("thing_classes", None))
    labels = None
    vis = visualizer.overlay_instances(masks = inst.gt_masks, boxes = inst.gt_boxes, labels=labels)

    img_write = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)
    if not display:
        save_dir = os.path.join(os.getcwd(),'all_data/interactive_output_multiscale/inference/compare',f'gt_segm__{text}')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"{img['image_id']}.jpg"), img_write)
    else:
        cv2.imshow("img_window",img_write)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    del visualizer

@torch.no_grad()
def save_pred_segm(batch, model, indx = 0, display=True, data_name = "coco_2017_val", text=None):
    preds = model(batch)[0]
    pred_masks = preds[0]['instances'].pred_masks.to(device ='cpu')
    
    # print(pred_masks.shape)
    img = batch[0]
    # image = img["image"]
    image = np.asarray(img["image"].permute(1,2,0))
    # print(image.shape, img['fg_scrbs'].shape)
    # image = cv2.resize(image, pred_masks[0].shape[::-1])
    # trans = torchvision.transforms.Resize(image.shape[:2])
    pred_masks = F.resize(pred_masks.to(dtype=torch.uint8), image.shape[:2])
    # pred_masks = (pred_masks*255).to(dtype=torch.uint8)
    # pred_masks = trans(pred_masks)
    print(pred_masks.shape)
    fg_scrbs = img['fg_scrbs']
    bg_scrbs = img['bg_scrbs']
    # print(fg_scrbs.shape, image.shape)
    for scrb in fg_scrbs:
        color = [np.random.randint(0, 255), np.random.randint(0, 1), np.random.randint(0, 255)]
        image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
    if bg_scrbs is not None:
        for scrb in bg_scrbs:
            color = [np.random.randint(0, 255), np.random.randint(0, 1), np.random.randint(0, 255)]
            image[scrb>0.5, :] = np.array(color, dtype=np.uint8)
    # print(image[:, :, ::-1].shape)

    inst = preds[0]['instances'].to(device ='cpu')

    metadata=MetadataCatalog.get(data_name)
    visualizer = Visualizer(image, metadata)
    # labels = _create_text_labels(inst.pred_classes, scores=None, class_names = metadata.get("thing_classes", None))
   
    # vis = visualizer.overlay_instances(masks = pred_masks)
    # vis = visualizer.draw_instance_predictions(preds[0]['instances'].to(device ='cpu'))
    
    vis = visualizer.overlay_instances(masks = pred_masks, labels=None)

    # save_dir = os.path.join(os.getcwd(),'interactive_output_multiscale/inference/compare/',f'pred_segm_{text}')
    # os.makedirs(save_dir, exist_ok=True)

    img_write = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)
    # cv2.imwrite(os.path.join(save_dir, f"{img['image_id']}.jpg"), img_write)

    if not display:
        save_dir = os.path.join(os.getcwd(),'all_data/iterative_train_scratch/inference/compare',f'final_model_{text}')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"{img['image_id']}.jpg"), img_write)
    else:
        cv2.imshow("img_window",img_write)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    del visualizer