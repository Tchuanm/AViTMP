import torch
import torch.nn as nn
import ltr.models.layers.filter as filter_layer


def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(1, outplanes),
        nn.ReLU(inplace=True),
    ]
    return layers


class Head(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

    def forward(self, train_feats, test_feats, pos_train, pos_test, train_bb, *args, **kwargs):
        assert train_bb.dim() == 3
        num_sequences = train_bb.shape[1]
        for i, (train_feat, test_feat) in enumerate(zip(train_feats, test_feats)):
            if train_feat.dim() == 5:
                train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
            if test_feat.dim() == 5:
                test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])
            # Extract features
            train_feat = train_feat.reshape(-1, num_sequences, *test_feat.shape[-2:])
            test_feat = test_feat.reshape(-1, num_sequences, *test_feat.shape[-2:])
            train_feat = self.extract_head_feat(train_feat, num_sequences)
            test_feat = self.extract_head_feat(test_feat, num_sequences)
            train_feats[i] = train_feat
            test_feats[i] = test_feat

        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feats, test_feats, pos_train, pos_test, *args, **kwargs)

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        # if num_sequences is None:
        #     return self.feature_extractor(feat)
        N,B,HW,C = feat.size()
        input = feat.reshape(N*B, C, 18, 18)
        output = self.feature_extractor(input)
        return output.flatten(-2).reshape(N,B,HW, output.shape[-3])

    def get_filter_and_features(self, train_feat, test_feat, pos_train, pos_test, train_label, *args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, pos_train, pos_test, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, pos_train, pos_test, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights

        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, pos_train, pos_test, train_label, num_gth_frames,  *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, pos_train, pos_test, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


class LinearFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)  # input : 1/bs/768/18/18, 2/768/1/1/, output: 1/2/18/18


class ConvFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        hidden_channel = num_channels // 4
        self.project_filter = project_filter

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, hidden_channel))
        layers.extend(conv_layer(hidden_channel, hidden_channel))
        layers.extend(conv_layer(hidden_channel, hidden_channel))
        layers.extend(conv_layer(hidden_channel, hidden_channel))
        self.tower_cls = nn.Sequential(*layers)

        self.cls_layer = nn.Conv2d(hidden_channel, 1, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
            # filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        feats_att = attention.unsqueeze(2)*feat # (nf, ns, c, h, w)

        feats_tower = self.tower_cls(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        fg_bg = torch.exp(self.cls_layer(feats_tower)).unsqueeze(0).reshape(nf, ns, h, w)
        return fg_bg


class DenseBoxRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter
        hidden_channel = num_channels // 4

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, hidden_channel))
        layers.extend(conv_layer(hidden_channel, hidden_channel))
        layers.extend(conv_layer(hidden_channel, hidden_channel))
        layers.extend(conv_layer(hidden_channel, hidden_channel))
        self.tower = nn.Sequential(*layers)

        self.bbreg_layer = nn.Conv2d(hidden_channel, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        feats_att = attention.unsqueeze(2)*feat # (nf, ns, c, h, w)

        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0) # (nf*ns, 4, h, w)
        return ltrb
