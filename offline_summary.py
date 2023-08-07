#from mask2former.data.datasets.register_coco_mval_multi_insts import *
from detectron2.data import DatasetCatalog, MetadataCatalog
import debugpy
import pickle
from prettytable import PrettyTable
import os
import csv
import argparse

import torch

def get_statistics(summary_stats, save_stats_path=None):
    # with open(pickle_path, 'rb') as handle:
    #     summary_stats= pickle.load(handle)
    
    model_name =  summary_stats["model"] 
    dataset_name = summary_stats["dataset"]
    iou_threshold = summary_stats["iou_threshold"]
    # clicked_objects_per_interaction = summary_stats["clicked_objects_per_interaction"]
    ious_objects_per_interaction = summary_stats["ious_objects_per_interaction"]
    # object_areas_per_image = summary_stats['object_areas_per_image']
    # fg_click_coords_per_image = summary_stats['fg_click_coords_per_image']  
    # bg_click_coords_per_image = summary_stats['bg_click_coords_per_image']  
    

    NFO = 0
    NFI = 0
    NCI_all = 0.0
    NCI_suc = 0.0
    Avg_IOU = 0.0
    total_images = len(list(ious_objects_per_interaction.keys()))
    total_num_instances = 0
    for _image_id in ious_objects_per_interaction.keys():
        final_ious = ious_objects_per_interaction[_image_id][-1]

        total_interactions_per_image = len(ious_objects_per_interaction[_image_id]) + len(final_ious)-1

        Avg_IOU += sum(final_ious)/len(final_ious)
        NCI_all += total_interactions_per_image/len(final_ious)
        total_num_instances += len(final_ious)

        _is_failed_image = False
        _suc = 0
        for i, iou in enumerate(final_ious):
            if iou<iou_threshold:
                _is_failed_image = True
                NFO +=1
            else:
                _suc+=1
        if _suc!=0:
            NCI_suc += total_interactions_per_image/_suc
        else:
            NCI_suc += total_interactions_per_image
        if _is_failed_image:
            NFI+=1

    NCI_all/=total_images
    NCI_suc/=total_images
    Avg_IOU/=total_images
                
    row = [model_name, NCI_all, NCI_suc, NFI, NFO, Avg_IOU, iou_threshold, 10, total_num_instances]

    table = PrettyTable()
    table.field_names = ["dataset", "NCI_all", "NCI_suc", "NFI", "NFO", "Avg_IOU", "#samples"]
    table.add_row([dataset_name, NCI_all, NCI_suc, NFI, NFO, Avg_IOU, total_num_instances])

    print(table)
    
    if save_stats_path is not None:
        save_stats_path = os.path.join(save_stats_path, f'{dataset_name}.txt')
    else:
        save_stats_path = os.path.join("./output/iccv/eval", f'{dataset_name}.txt')
        os.makedirs("./output/iccv/eval", exist_ok=True)
    if not os.path.exists(save_stats_path):
        # print("No File")
        header = ['model', "NCI_all", "NCI_suc", "NFI", "NFO", "Avg_IOU", 'IOU_thres',"max_num_iters", "num_inst"]
        with open(save_stats_path, 'w') as f:
            writer = csv.writer(f, delimiter= "\t")
            writer.writerow(header)
            # writer.writerow(row)

    with open(save_stats_path, 'a') as f:
        writer = csv.writer(f, delimiter= "\t")
        writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser("Evaluation summary from pickle path")
    parser.add_argument("--pickle-path", default="", type=str, help="Path to the pickle file")
    # parser.add_argument("--ablation", default="", type=str, help="ablation or not")
    parser.add_argument("--ablation", action="store_true", help="perform evaluation only")
    return parser.parse_args()

def main():
    args = parse_args()

    pickle_path = args.pickle_path
    with open(pickle_path, 'rb') as handle:
        summary_stats= pickle.load(handle)
    if args.ablation:
        get_statistics(summary_stats, ablation=True)
    else:
        get_statistics(summary_stats, ablation=False)


if __name__ == "__main__":
    main()
