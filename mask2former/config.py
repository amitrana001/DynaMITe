# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    cfg.INPUT.MIN_AREA_FOR_OBJECT = 500.0
    cfg.INPUT.MIN_AREA_FOR_MASK = 400.0
    
    cfg.INPUT.DISTRACTOR_BG  = False
    cfg.INPUT.DISTRACTOR_OBJECTS = False
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.QUERY_INITIALIZER = "multi_scale" # OR single_scale
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


    #______________HODOR______________
    cfg.MODEL.HODOR = CN()
    cfg.MODEL.HODOR.ENCODER_LAYERS = 5
    cfg.MODEL.HODOR.DECODER_LAYERS = 5
    cfg.MODEL.HODOR.NUM_ATTN_HEADS = 8
    cfg.MODEL.HODOR.POS_ENCODINGS_IN_FMQ = False
    cfg.MODEL.HODOR.POS_ENCODINGS_IN_FQM = True
    cfg.MODEL.HODOR.TRANSFORMER_PRE_NORMALIZE = False

    #__________________Iterative_training________________

    cfg.ITERATIVE = CN()
    cfg.ITERATIVE.TRAIN = CN()
    cfg.ITERATIVE.TEST = CN()
    cfg.ITERATIVE.TRAIN.ACCUMULATE_INTERACTION_LOSS = False   #"loss_after_interaction" # "accumulate_loss"
    cfg.ITERATIVE.TRAIN.MAX_NUM_INTERACTIONS = 10
    cfg.ITERATIVE.TRAIN.RANDOM_INSTANCES_INTERACTION = False
    cfg.ITERATIVE.TRAIN.SINGLE_INSTANCE_INTERACTION = False
    cfg.ITERATIVE.TRAIN.INTERACTION_RESET_PROBABILITY = 0.2
    cfg.ITERATIVE.TRAIN.BACKPROP_CLASSIFICATION_SCORE = True
    cfg.ITERATIVE.TRAIN.CLASS_AGNOSTIC = False
    cfg.ITERATIVE.TRAIN.CONCAT_COORD_MASK_FEATURES = False
    cfg.ITERATIVE.TRAIN.CONCAT_COORD_IMAGE_FEATURES = False
    cfg.ITERATIVE.TRAIN.RANDOM_BG_QUERIES = False
    cfg.ITERATIVE.TRAIN.USE_POS_COORDS = False
    cfg.ITERATIVE.TRAIN.USE_POINTS = True
    cfg.ITERATIVE.TRAIN.USE_SCRIBBLES = False
    cfg.ITERATIVE.TRAIN.USE_CNN_BLOCK = False
    cfg.ITERATIVE.TRAIN.NUM_STATIC_BG_QUERIES = 9
    cfg.ITERATIVE.TRAIN.MIXED_TRAINING = False
    cfg.ITERATIVE.TRAIN.PROBABILITY_POINTS = 0.5
    cfg.ITERATIVE.TRAIN.USE_MULTICLASS_SOFTMAX = False

    cfg.ITERATIVE.TEST.INTERACTIVE_EVALAUTION = False
    cfg.ITERATIVE.TEST.MAX_NUM_INTERACTIONS = 20
    cfg.ITERATIVE.TEST.IOU_THRESHOLD = 0.85

    # ______________________Reverse Cross Attn____________
    cfg.REVERSE_CROSS_ATTN =CN()
    cfg.REVERSE_CROSS_ATTN.USE_REVERSE_CROSS_ATTN = False
    cfg.REVERSE_CROSS_ATTN.NUM_LAYERS = 5
    cfg.REVERSE_CROSS_ATTN.SCALE_FACTOR = 1.0
    cfg.REVERSE_CROSS_ATTN.USE_ATTN_MASK = False