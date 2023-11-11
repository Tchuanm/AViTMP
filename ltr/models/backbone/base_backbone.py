"""
\file vot.py

@brief Python utility functions for VOT toolkit integration

@author Luka Cehovin, Alessio Dore

@date 2023

"""

import os
import collections
import numpy as np

try:
    import trax
except ImportError:
    raise Exception('TraX support not found. Please add trax module to Python path.')

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])
Empty = collections.namedtuple('Empty', [])

class VOT(object):
    """ Base class for VOT toolkit integration in Python.
        This class is only a wrapper around the TraX protocol and can be used for single or multi-object tracking.
        The wrapper assumes that the experiment will provide new objects onlf at the first frame and will fail otherwise."""
    def __init__(self, region_format, channels=None, multiobject: bool = None):
        """ Constructor for the VOT wrapper.

        Args:
            region_format: Region format options
            channels: Channels that are supported by the tracker
            multiobject: Whether to use multi-object tracking
        """
        assert(region_format in [trax.Region.RECTANGLE, trax.Region.POLYGON, trax.Region.MASK])

        if multiobject is None:
            multiobject = os.environ.get('VOT_MULTI_OBJECT', '0') == '1'

        if channels is None:
            channels = ['color']
        elif channels == 'rgbd':
            channels = ['color', 'depth']
        elif channels == 'rgbt':
            channels = ['color', 'ir']
        elif channels == 'ir':
            channels = ['ir']
        else:
            raise Exception('Illegal configuration {}.'.format(channels))

        self._trax = trax.Server([region_format], [trax.Image.PATH], channels, metadata=dict(vot="python"), multiobject=multiobject)

        request = self._trax.wait()
        print('request.type: ',request.type)
        assert(request.type == 'initialize')

        self._objects = []

        assert len(request.objects) > 0 and (multiobject or len(request.objects) == 1)

        for object, _ in request.objects:
            if isinstance(object, trax.Polygon):
                self._objects.append(Polygon([Point(x[0], x[1]) for x in object]))
            elif isinstance(object, trax.Mask):
                self._objects.append(object.array(True))
            else:
                self._objects.append(Rectangle(*object.bounds()))

        self._image = [x.path() for k, x in request.image.items()]
        if len(self._image) == 1:
            self._image = self._image[0]

        self._multiobject = multiobject

        self._trax.status(request.objects)

    def region(self):
        """
        Returns initialization region for the first frame in single object tracking mode.

        Returns:
            initialization region
        """

        assert not self._multiobject

        return self._objects[0]

    def objects(self):
        """
        Returns initialization regions for the first frame in multi object tracking mode.

        Returns:
            initialization regions for all objects
        """

        return self._objects

    def report(self, status, confidence = None):
        """
        Report the tracking results to the client

        Arguments:
            status: region for the frame or a list of regions in case of multi object tracking
            confidence: confidence for the object detection, used only in single object tracking mode
        """

        def convert(region):
            """ Convert region to TraX format """
            # If region is None, return empty region
            if region is None: return trax.Rectangle.create(0, 0, 0, 0)
            print('region: ',region)
            assert isinstance(region, (Empty, Rectangle, Polygon, np.ndarray))
            if isinstance(region, Empty):
                return trax.Rectangle.create(0, 0, 0, 0)
            elif isinstance(region, Polygon):
                return trax.Polygon.create([(x.x, x.y) for x in region.points])
            elif isinstance(region, np.ndarray):
                return trax.Mask.create(region)
            else:
                return trax.Rectangle.create(region.x, region.y, region.width, region.height)

        if not self._multiobject:
            status = convert(status)
        else:
            assert isinstance(status, (list, tuple))
            status = [(convert(x), {}) for x in status]

        properties = {}

        if not confidence is None and not self._multiobject:
            properties['confidence'] = confidence

        self._trax.status(status, properties)

    def frame(self):
        """
        Get a frame (image path) from client

        Returns:
            absolute path of the image
        """
        if hasattr(self, "_image"):
            image = self._image
            del self._image
            return image

        request = self._trax.wait()

        # Only the first frame can declare new objects for now
        assert request.objects is None or len(request.objects) == 0

        if request.type == 'frame':
            image = [x.path() for k, x in request.image.items()]
            if len(image) == 1:
                return image[0]
            return image
        else:
            return None

    def quit(self):
        """ Quit the tracker"""
        if hasattr(self, '_trax'):
            self._trax.quit()

    def __del__(self):
        """ Destructor for the tracker, calls quit. """
        self.quit()

class VOTManager(object):
    """ VOT Manager is provides a simple interface for running multiple single object trackers in parallel. Trackers should implement a factory interface. """

    def __init__(self, factory, region_format, channels=None):
        """ Constructor for the manager. 
        The factory should be a callable that accepts two arguments: image and region and returns a callable that accepts a single argument (image) and returns a region.

        Args:
            factory: Factory function for creating trackers
            region_format: Region format options
            channels: Channels that are supported by the tracker
        """
        self._handle = VOT(region_format, channels, multiobject=True)
        self._factory = factory

    def run(self):
        """ Run the tracker, the tracking loop is implemented in this function, so it will block until the client terminates the connection."""
        objects = self._handle.objects()

        # Process the first frame
        image = self._handle.frame()
        if not image:
            return

        trackers = [self._factory(image, object) for object in objects]
        # print('trackers:', trackers)
        while True:

            image = self._handle.frame()
            if not image:
                break

            status = [tracker(image) for tracker in trackers]
            # print('status:', status)
            self._handle.report(status)
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ltr.models.backbone.patch_embed import PatchEmbed
# from lib.models.ostrack.utils import combine_tokens, recover_tokens
from collections import OrderedDict
# from ltr.models.transformer.transformer import TransformerDecoder, TransformerDecoderLayer


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.pos_embed_x = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False
        # self.FC = nn.Linear(768, 256)
        # self.norm2 = nn.LayerNorm(256)

    def finetune_track(self, patch_start_index=1):

        search_size = to_2tuple(288)       # 256
        new_patch_size = 16      # 16

        self.return_inter = False      #  False
        self.return_stage = []     # []
        self.add_sep_seg = False


        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, x, z, outlayer):
        outputs = OrderedDict()
        x = self.patch_embed(x)
        x += self.pos_embed_x
        z = self.patch_embed(z)
        z += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
        B, HW, C = z.shape
        x = x.reshape(B, -1, C)
        x = torch.cat((x, z), dim=1)
        x = self.pos_drop(x)
        outputs['input_embeding'] = x
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i != len(self.blocks)-1:
                outputs[i] = x
        x = self.norm(x)
        outputs[len(self.blocks)-1] = x 
        outputs['pos_embed_x'] = self.pos_embed_x
        return outputs

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, z, outlayer):
        outputs = self.forward_features(x, z, outlayer)
        return outputs
