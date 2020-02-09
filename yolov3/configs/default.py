from easydict import EasyDict as edict

def get_default_config():
    CFG = edict()
    CFG.MODEL = edict()
    # backbone
    CFG.MODEL.BACKBONE = edict()
    CFG.MODEL.BACKBONE.FREEZE_AT = 2
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
    CFG.MODEL.DARKNETS.NUM_CLASSES = 1000 # or None(train yolov3)
    # stem
    CFG.MODEL.DARKNETS.STEM_OUT_CHANNELS = 64

    return CFG



