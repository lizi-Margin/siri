import cv2
import torch
import numpy as np
from typing import List, Union
from ultralytics import YOLO
from siri.global_config import GlobalConfig as cfg


def to_int(obj:tuple) -> tuple:
    obj = list(obj)
    for i in range(len(obj)):
        obj[i] = int(obj[i])
    return tuple(obj)


def crop(image_origin):
    height, width = image_origin.shape[:2]

    if width > height:
        offset = (width - height) // 2
        image_cropped = image_origin[:, offset:offset + height]
    else:
        offset = (height - width) // 2
        image_cropped = image_origin[offset:offset + width, :]

    return image_cropped


def pad(image_origin, to_sz_wh=None):
    if to_sz_wh is not None:
        to_size_w, to_size_h = to_sz_wh
        height, width = image_origin.shape[:2]

        if to_size_h < height and to_size_w < width:
            if to_size_h/height < to_size_w/width:
                image_origin = cv2.resize(image_origin, to_int((width * to_size_h/height, to_size_h,)))
            else:
                image_origin = cv2.resize(image_origin, to_int((to_size_w, height * to_size_w/width,)))
        elif to_size_h >= height and to_size_w >= width:
            if to_size_h/height < to_size_w/width:
                image_origin = cv2.resize(image_origin, to_int((width * to_size_h/height, to_size_h,)))
            else:
                image_origin = cv2.resize(image_origin, to_int((to_size_w, height * to_size_w/width,)))
        else:
            raise NotImplementedError
        # print(to_sz_wh, " ", width, " ", height)
    else:
        to_size_w = max(width, height);to_size_h = to_size_w

    height, width = image_origin.shape[:2]

    padded_image = np.zeros((to_size_h, to_size_w, 3), dtype=image_origin.dtype)

    if to_size_h > height:
        y_offset = (to_size_h - height) // 2
        padded_image[y_offset:y_offset + height, :] = image_origin
    else:
        x_offset = (to_size_w - width) // 2
        padded_image[:, x_offset:x_offset + width] = image_origin

    return padded_image


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scaleFill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):

        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    @staticmethod
    def _update_labels(labels, ratio, padw, padh):
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


def pre_transform(im, sz_wh):
    sz_wh = 640, 640
    letterbox = LetterBox(
        tuple(reversed(sz_wh)),
        auto=False,
        scaleFill=False,
        scaleup=False,
        center=True
    )
    return [letterbox(image=x) for x in im]


def preprocess(im: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor: # to('cuda')
    assert len(im[0].shape) == 3 and im[0].shape[-1] == 3, "im shape should be (n, h, w, 3)"
    
    im = np.stack(pre_transform(im, cfg.sz_wh))
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    # im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im)

    im = im.to('cuda')
    im = im.half() if cfg.half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def postprocess(im: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(im, list):
        ret = []
        for image in im:
            assert isinstance(image, np.ndarray)
            ret.append(postprocess(image))
        return ret
    elif isinstance(im, np.ndarray):
        assert len(im.shape) == 3
        return im[..., ::-1]
    else:
        assert False


# def plot_image(image, post_process=True):
#     assert len(image.shape) == 3
#     if post_process:
#         if isinstance(image, torch.Tensor):
#             image.int()
#             image = image.cpu().numpy()
#             # image = image[::-1].transpose((1, 2, 0))
#         elif isinstance(image, np.ndarray):
#             # print(image.shape)
#             image.astype(np.uint8)
#             # image = image[..., ::-1]
#             # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     assert isinstance(image, np.ndarray)
#     cv2.imshow('Image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def plot_yolo_results(results):
#     for index, result in enumerate(results):
#         result_image = result.plot()
#         result_image = result_image[..., ::-1]
#         plot_image(result_image)
#         # cv2.imwrite(f"output_{index}.jpg", result_image)


