import torch
import torch.nn as nn
import onnx
import onnx2pytorch
import cv2
import numpy as np


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


voc07_mean = (0.406, 0.456, 0.485)  # np.mean(train_set.train_data, axis=(0,1,2))/255
voc07_std = (0.225, 0.224, 0.229)  # np.std(train_set.train_data, axis=(0,1,2))/255
mu = torch.tensor(voc07_mean[::-1])  # [0, 1, 2] -> [2, 1, 0]
std = torch.tensor(voc07_std[::-1])  # [0, 1, 2] -> [2, 1, 0]
def normalize(X):
    return (X - mu)/std


class ONNXYOLOModel(nn.Module):
    def __init__(self, onnx_model_path: str, input_size: int = 52, device: str = 'cpu'):
        super(ONNXYOLOModel, self).__init__()

        self.stride = 4
        self.num_anchors = 5

        self.onnx_model = onnx.load(onnx_model_path)
        self.model = onnx2pytorch.ConvertModel(self.onnx_model, experimental=True)
        self.model.eval()
        self.input_size = input_size
        self.device = device

        self.anchor_size = torch.tensor(
            [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]])
        self.anchor_boxes = self.create_grid(input_size)
        mu_full = mu.repeat_interleave(input_size * input_size)
        std_full = std.repeat_interleave(input_size * input_size)
        self.register_buffer("mu", mu_full)
        self.register_buffer("std", std_full)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / self.std
        x = x.reshape(x.shape[0], 3, self.input_size, self.input_size)
        return self.model(x)


    def create_grid(self, input_size):
        w, h = input_size, input_size
        
        fmp_w, fmp_h = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2]
        grid_xy = grid_xy[:, None, :].repeat(1, self.num_anchors, 1)

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

        # [HW, KA, 4] -> [M, 4]
        anchor_boxes = torch.cat([grid_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

        return anchor_boxes


    def decode_boxes(self, anchors, txtytwth_pred):
        """ Input: \n
                txtytwth_pred : [B, H*W*KA, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*KA, 4] \n
        """
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + anchors[..., :2]
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * anchors[..., 2:]

        # [B, H*W*KA, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1) * self.stride

        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[..., :2] = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred[..., 2:] = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        
        return x1y1x2y2_pred


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 1)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def torch_to_cv2(torch_img):
    # Convert a PyTorch tensor to a NumPy array and then to a CV2 image
    img = torch_img.permute(1, 2, 0).cpu().numpy()
    if img.dtype == np.float32 or img.max() <= 1.0:
        # If the image is in float format, scale it to 0-255
        img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
