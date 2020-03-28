import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from torch.onnx import OperatorExportTypes
from yolov3.modeling import build_backbone

from yolov3.configs.default import get_default_config
from yolov3.layers import ShapeSpec

def mobilenetv2():
    input = cv2.imread("../../../../../home/lin/mnist/train/6/13.jpg")
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = cv2.resize(input, (32, 32))

    # numpy image(H x W x C) to torch image(C x H x W)
    input = np.transpose(input, (2, 0, 1)).astype(np.float32)

    # normalize
    input = input/255.0

    input = Variable(torch.from_numpy(input))
    # add one dimension in 0
    input = input.unsqueeze(0)
    # print(input.shape)

    cfg = get_default_config()
    cfg.MODEL.BACKBONE.NAME = "build_mobilenetv2_backbone"
    # height = imgHeight, width = imgWeight
    input_shape = ShapeSpec(channels=3, height=32, width=32, stride=32)
    # build backbone
    net = build_backbone(cfg, input_shape)
    # load trained model
    net.load_state_dict(torch.load("../../tools/weights/build_mobilenetv2_backbone_epoch_63.pth"))
    # disable BathNormalization and Dropout
    net.eval()
    # test model
    tensor_out = net(input)
    # softmax
    model_test_out = F.softmax(tensor_out["linear"], dim=1)
    print("the model result is {}".format(model_test_out))

    # pytorch -> onnx
    torch.onnx.export(net, #model being run
                     input,  #model input (or a tuple for multiple inputs)
                     "./weights/mobilenetv2.onnx", # where to save the model (can be a file or file-like object)
                     operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

    import onnx
    import onnxruntime

    # load onnx model
    onnx_model = onnx.load("./weights/mobilenetv2.onnx")

    # check onnx model
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("./weights/mobilenetv2.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # softmax
    tensor_ort_out = torch.from_numpy(ort_outs[0])
    onnx_test_out = F.softmax(tensor_ort_out, dim=1)

    print("the onnx result is {}".format(onnx_test_out))

    # compare onnx Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(tensor_out["linear"]), ort_outs[0], rtol=1e-01, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    mobilenetv2()