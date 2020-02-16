from easydict import EasyDict as edict

def get_default_config():
    CFG = edict()
    CFG.MODEL = edict()
    # backbone
    CFG.MODEL.BACKBONE = edict()
    # FPN
    CFG.MODEL.FPN = edict()
    # YOLOV3
    CFG.MODEL.YOLOV3 = edict()
    #anchor generator
    CFG.MODEL.ANCHOR_GENERATOR = edict()

    CFG.MODEL.BACKBONE.FREEZE_AT = 0
    CFG.MODEL.PIXEL_MEAN = 3
    CFG.MODEL.BACKBONE.NAME = "build_darknet_backbone"
    
    # darknet config
    CFG.MODEL.DARKNETS=edict()

    CFG.MODEL.DARKNETS.DEPTH = 53
    CFG.MODEL.DARKNETS.OUT_FEATURES = ["res6"] # or ["res4", "res5", "res6"](train yolov3)
    CFG.MODEL.DARKNETS.NUM_GROUPS = 1
    CFG.MODEL.DARKNETS.WIDTH_PER_GROUP = 32
    CFG.MODEL.DARKNETS.RES2_OUT_CHANNELS=128
    CFG.MODEL.DARKNETS.NORM = "BN"
    CFG.MODEL.DARKNETS.ACTIVATE = "LeakReLU"
    CFG.MODEL.DARKNETS.ACTIVATE_ALPHA = 0.1
    CFG.MODEL.DARKNETS.NUM_CLASSES = 1000 # or None(train yolov3)

    # stem
    CFG.MODEL.DARKNETS.STEM_OUT_CHANNELS = 64

    # FPN
    CFG.MODEL.FPN.IN_FEATURES = ["res4", "res5", "res6"]
    CFG.MODEL.FPN.OUT_CHANNELS = [128, 256, 512]
    CFG.MODEL.FPN.FUSE_TYPE = "concat" # or 'avg' or 'sum
    CFG.MODEL.FPN.NORM = "BN"

    # meta_arch
    CFG.MODEL.META_ARCHITECTURE = "Yolov3"
    CFG.MODEL.YOLOV3.NUM_CLASSES = 20 # default voc dataset
    CFG.MODEL.YOLOV3.IN_FEATURES = ["p3", "p4", "p5"]

    #anchor generator
    CFG.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
    CFG.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0
    CFG.MODEL.ANCHOR_GENERATOR.SIZES = (((10,13), (16,30), (33,23)),((30,61),  (62,45),  (59,119)),((116,90),  (156,198),  (373,326)))

    return CFG



