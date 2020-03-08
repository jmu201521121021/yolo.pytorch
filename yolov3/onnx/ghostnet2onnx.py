import numpy as np
import torch
from torch.onnx import OperatorExportTypes
from yolov3.modeling import build_backbone

from yolov3.configs.default import get_default_config
from yolov3.layers import  ShapeSpec

def ghostnet():

    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    cfg = get_default_config()
    cfg.MODEL.BACKBONE.NAME = "build_ghostnet_backbone"
    input_shape = ShapeSpec(channels=3, height=256, width=256, stride=32)
    net = build_backbone(cfg, input_shape)
    # print(net)
    # net.load_state_dict(torch.load("./weights/darknet53.pth"))

    net.eval()
    tensor_out = net(x)

    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./weights/ghosnet.onnx",  # where to save the model (can be a file or file-like object)
                      operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                      # verbose=True,  # NOTE: uncomment this for debugging
                      # export_params=True,
                      )
    # print(tensor_out["res3"].size())
    import onnx
    import onnxruntime
    onnx_model = onnx.load("./weights/ghosnet.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("./weights/ghosnet.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(tensor_out["linear"]), ort_outs[0], rtol=1e-01, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
if __name__ == "__main__":
    ghostnet()