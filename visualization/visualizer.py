# Copyright (c) Facebook, Inc. and its affiliates.
import colorsys
import logging
import math
import numpy as np
from enum import Enum, unique
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from detectron2.utils.visualizer import ColorMode, GenericMask, random_color, _OFF_WHITE
from detectron2.utils.visualizer import Visualizer as Visualizer_d2
from detectron2.structures import Instances

import matplotlib.colors as mcolors

colors = [
    'deepskyblue',
    'lawngreen',
    'orange',
    'tomato',
    'violet',

]


class Visualizer(Visualizer_d2):
    # pred_masks: NxHxW
    def draw_masks(self, pred_masks, random_color=False):
        for i in range(len(pred_masks)):
            pred_mask = pred_masks[i]
            if not random_color:
                color = colors[i]
            else:
                color = None
            self.draw_binary_mask(
                pred_mask,
                color=color,
                # edge_color=mplc.to_rgb(color) + (1,),
                edge_color=_OFF_WHITE,
                alpha=0.5,
                # area_threshold=area_threshold,
            )
        return self.output

    def draw_instance_predictions(self, predictions):
        assert predictions.has("pred_masks")
        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        self.overlay_instances(
            masks=masks,
            boxes=None,
            labels=None,
            keypoints=None,
            assigned_colors=colors,
            alpha=0.5,
        )
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            # linewidth=0,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1)
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def draw_binary_mask(
            self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5, area_threshold=10
    ):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component smaller than this area will not be shown.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes:
            # draw polygons for regular masks
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
        else:
            # TODO: Use Path/PathPatch to draw vector graphics:
            # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            has_valid_segment = True
            self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=0)

        if text is not None and has_valid_segment:
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            self._draw_text_in_mask(binary_mask, text, lighter_color)
        return self.output