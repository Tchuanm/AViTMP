import math
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor
import os

import ltr.models.transformer.transformer as trans

import ltr.models.transformer.filter_predictor as fp
import ltr.models.transformer.heads as heads


class AViTMPnet(nn.Module):
    """The AViTMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):
        """Runs the AViTMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]), test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        feat_head, pos_train = self.get_backbone_head_feat(train_feat)   #  bs 1024 18 18
        test_feat_head, train_feat_head = [], []
        for key, _ in feat_head.items():
            test_feat_head.append(feat_head[key][:, -pos_train.shape[1]:, ...])
            train_feat_head.append(feat_head[key][:, :-pos_train.shape[1], ...])

        pos_test = pos_train

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, pos_train, pos_test, train_bb, *args, **kwargs)

        return test_scores, bbox_preds 

    def get_backbone_head_feat(self, backbone_feat):
        # feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        pos = backbone_feat['pos_embed_x']
        del backbone_feat['pos_embed_x']
        if len(self.head_layer) == 1:
            return backbone_feat[self.head_layer[0]], pos
        return backbone_feat, pos

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im12, im3, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im12, im3, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def avitmpnet_b(filter_size=4, head_layer=['input_embeding', 'layer0', 'layer1', 'layer2', 'layer3'],
              backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, out_feature_dim=768, frozen_backbone_layers=3, nhead=8, num_encoder_layers=6,
              num_decoder_layers=6, feature_sz=18, use_test_frame_encoding=True, remove=0):
    # Backbone
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    pretrained = os.path.join(pretrained_path, 'mae_pretrain_vit_base.pth')
    backbone_net = backbones.vit_base_patch16_224(pretrained=pretrained, frozen_blocks=frozen_backbone_layers, remove=remove)
    backbone_net.finetune_track(patch_start_index=1)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))
    feature_dim = 768       # backbone_output_dim
    dim_feedforward = 4 * out_feature_dim
    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim, input_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    # classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)
    classifier = heads.ConvFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # AViTMP network
    net = AViTMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


@model_constructor
def avitmpnet_l(filter_size=1, head_layer=['input_embeding', 'layer0', 'layer1', 'layer2', 'layer3'],
               backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
               final_conv=True, out_feature_dim=1024, frozen_backbone_layers=3, nhead=8, num_encoder_layers=6,
               num_decoder_layers=6, feature_sz=18, use_test_frame_encoding=True, remove=0):
    # Backbone
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    pretrained = os.path.join(pretrained_path, 'mae_pretrain_vit_large.pth')
    backbone_net = backbones.vit_large_patch16(pretrained, frozen_blocks=frozen_backbone_layers, remove=remove)
    backbone_net.finetune_track(patch_start_index=1)
    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    in_feature_dim = 1024       # backbone_output_dim
    dim_feedforward = 4 * out_feature_dim
    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=in_feature_dim, input_dim=in_feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    # classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)
    classifier = heads.ConvFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # AViTMP network
    net = AViTMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net
