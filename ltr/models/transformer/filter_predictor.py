import torch
import torch.nn as nn
from ltr.models.transformer.position_encoding import PositionEmbeddingSine


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FilterPredictor(nn.Module):
    def __init__(self, transformer, feature_sz, use_test_frame_encoding=True):
        super().__init__()
        self.transformer = transformer
        self.feature_sz = feature_sz
        self.use_test_frame_encoding = use_test_frame_encoding

        self.box_encoding = MLP([4, self.transformer.d_model//4, self.transformer.d_model, self.transformer.d_model])

        self.query_embed_fg = nn.Embedding(1, self.transformer.d_model)

        if self.use_test_frame_encoding:
            self.query_embed_test = nn.Embedding(1, self.transformer.d_model)

        self.query_embed_fg_decoder = self.query_embed_fg

    def forward(self, train_feat, test_feat, pos_train, pos_test, train_label, train_ltrb_target,  *args, **kwargs):
        return self.predict_filter(train_feat, test_feat, pos_train, pos_test, train_label, train_ltrb_target, pos_train, pos_test, *args, **kwargs)

    def predict_filter(self, train_feats, test_feats, pos_train, pos_test, train_label, train_ltrb_target, *args, **kwargs):
        #train_label size guess: Nf_tr, Ns, H, W.

        if train_ltrb_target.dim() == 3:
                train_ltrb_target = train_ltrb_target.unsqueeze(1)
        train_label_seq = train_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2) # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2) # Ns,4,Nf_tr*H*W
        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2,0,1) # Nf_tr*H*H,Ns,C
        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq

        # feats = []
        for i, (train_feat, test_feat) in enumerate(zip(train_feats, test_feats)):
            if train_feat.dim() == 3:
                train_feat = train_feat.unsqueeze(1)
            if test_feat.dim() == 3:
                test_feat = test_feat.unsqueeze(1)

            test_feat = test_feat.transpose(-1, -2)
            train_feat = train_feat.transpose(-1, -2)
            hw = test_feat.shape[-1]

            test_feat_seq = test_feat.permute(1, 2, 0, 3).flatten(2).permute(2, 0, 1) # Nf_te*H*W, Ns, C
            train_feat_seq = train_feat.permute(1, 2, 0, 3).flatten(2).permute(2, 0, 1) # Nf_tr*H*W, Ns, C

            if self.use_test_frame_encoding:
                test_token = self.query_embed_test.weight.reshape(1, 1, -1)
                test_label_enc = torch.ones_like(test_feat_seq) * test_token
                feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
                train_feats[i] = feat
            else:
                test_label_enc = torch.zeros_like(test_feat_seq)
                feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
                train_feats[i] = feat

        output_embed, enc_mem = self.transformer(train_feats, mask=None, query_embed=self.query_embed_fg_decoder.weight, pos_embed=None)

        enc_opt = enc_mem[-hw:].transpose(0, 1)
        dec_opt = output_embed.squeeze(0).transpose(1, 2)

        return dec_opt.reshape(test_feat.shape[1], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(1, test_feat.shape[1], -1, self.feature_sz, self.feature_sz)

    def predict_cls_bbreg_filters_parallel(self, train_feats, test_feats, pos_train, pos_test, train_label, num_gth_frames, train_ltrb_target, *args, **kwargs):
        # train_label size guess: Nf_tr, Ns, H, W.
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)
        train_label = torch.cat([train_label, train_label], dim=1)
        train_ltrb_target = torch.cat([train_ltrb_target, train_ltrb_target], dim=1)        # 1 4 18 18
        train_label_seq = train_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2) # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2) # Ns,4,Nf_tr*H*W
        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2,0,1) # Nf_tr*H*H,Ns,C
        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq
        # feats = []
        for i, (train_feat, test_feat) in enumerate(zip(train_feats, test_feats)):
            if train_feat.dim() == 3:
                train_feat = train_feat.unsqueeze(1)
            if test_feat.dim() == 3:
                test_feat = test_feat.unsqueeze(1)

            test_feat = test_feat.transpose(-1, -2)
            train_feat = train_feat.transpose(-1, -2)
            hw = test_feat.shape[-1]

            train_feat = torch.cat([train_feat, train_feat], dim=1)
            test_feat = torch.cat([test_feat, test_feat], dim=1)

            test_feat_seq = test_feat.permute(1, 2, 0, 3).flatten(2).permute(2, 0, 1) # Nf_te*H*W, Ns, C
            train_feat_seq = train_feat.permute(1, 2, 0, 3).flatten(2).permute(2, 0, 1) # Nf_tr*H*W, Ns, C

            if self.use_test_frame_encoding:
                test_token = self.query_embed_test.weight.reshape(1, 1, -1)
                test_label_enc = torch.ones_like(test_feat_seq) * test_token
                feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
                train_feats[i] = feat
            else:
                test_label_enc = torch.zeros_like(test_feat_seq)
                feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq+test_label_enc], dim=0)
                train_feats[i] = feat

        src_key_padding_mask = torch.zeros(feat.shape[1], feat.shape[0]).bool()
        src_key_padding_mask[1, num_gth_frames*hw:-hw] = 1.
        src_key_padding_mask = src_key_padding_mask.bool().to(feat.device)

        output_embed, enc_mem = self.transformer(train_feats, mask=src_key_padding_mask,
                                                 query_embed=self.query_embed_fg_decoder.weight,
                                                 pos_embed=None)         #  , train_label=train_label

        enc_opt = enc_mem[-hw:].transpose(0, 1).permute(0, 2, 1).reshape(1, test_feat.shape[1], -1, self.feature_sz, self.feature_sz)
        dec_opt = output_embed.squeeze(0).transpose(1, 2).reshape(test_feat.shape[1], -1, 1, 1)

        cls_enc_opt = enc_opt[:, 0].unsqueeze(1)
        bbreg_enc_opt = enc_opt[:, 1].unsqueeze(1)
        cls_dec_opt = dec_opt[0].unsqueeze(0)
        bbreg_dec_opt = dec_opt[1].unsqueeze(0)

        return cls_dec_opt, bbreg_dec_opt, cls_enc_opt, bbreg_enc_opt
