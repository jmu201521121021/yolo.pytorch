
def merge_config(args, cfg):
    cfg.DATASET.DATA_ROOT = args.data_root
    cfg.DATASET.DATASET_NAME = args.data_name
    cfg.DATASET.TRAIN_TRANSFORM = args.train_transform
    cfg.DATASET.TEST_TRANSFORM = args.test_transform
    cfg.MODEL.BACKBONE.NAME = args.backbone_name
    if "model_name" in args:
        cfg.SOLVER.MODEL_NAME = args.model_name
    cfg.SOLVER.BATCH_SIZE = args.batch_size
    cfg.SOLVER.SAVE_MODEL_FREQ = args.save_model_freq
    cfg.SOLVER.SAVE_MODEL_DIR = args.save_model_dir
    cfg.SOLVER.START_EPOCH = args.start_epoch
    cfg.SOLVER.MAX_EPOCH = args.max_epoch

    cfg.SOLVER.PRINT_LOG_FREQ = args.print_log_freq
    cfg.SOLVER.TEST_FREQ = args.test_freq

    cfg.SOLVER.LR = args.lr
    cfg.SOLVER.MOMENTUM = args.momentum
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    cfg.SOLVER.DECAY_EPOCH = args.decay_epoch

    cfg.SOLVER.GPU_IDS = args.gpu_ids
    cfg.SOLVER.PRETRAINED =args.pretrained
    cfg.MODEL.DARKNETS.NUM_CLASSES = args.num_classes

    return  cfg