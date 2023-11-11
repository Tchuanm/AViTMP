from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
from ltr.models.layers import activation
from pytracking.features.lame import LAME
import cv2 as cv

import numpy as np
from collections import defaultdict


class AViTMP(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # self.lame = LAME()
        self.pre_scores = []
        self.cls_test_feat_enc = []
        self.search_area_scale = 5
        self.search_area_weight = 1
        self.scale_threshold = 16 
        self.max_score_threshold = 1.00
        self.update_time = 0
        # Initialize network
        self.initialize_features()
        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])
        self.sz_scale0 = state[3] * state[2]

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz
        tfs = self.params.get('train_feature_size', 18)
        stride = self.params.get('feature_stride', 16)
        self.train_img_sample_sz = torch.Tensor([tfs*stride, tfs*stride])

        # Set search area
        search_area = torch.prod(self.target_sz * self.search_area_scale).item()
        self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        self.logging_dict = defaultdict(list)

        self.target_scales = []
        self.target_not_found_counter = 0

        self.cls_weights_avg = None
        self.bad_clf = 0
        self.pre_pos = None         # self.pos2
        self.pre_target_sz = None
        self.pre_target_scale = None

        out = {'time': time.time() - tic}
        return out

    def clip_bbox_to_image_area(self, bbox, image, minwidth=10, minheight=10):
        H, W = image.shape[:2]
        x1 = max(0, min(bbox[0], W - minwidth))
        y1 = max(0, min(bbox[1], H - minheight))
        x2 = max(x1 + minwidth, min(bbox[0] + bbox[2], W))
        y2 = max(y1 + minheight, min(bbox[1] + bbox[3], H))
        return torch.Tensor([x1, y1, x2 - x1, y2 - y1])

    def encode_bbox(self, bbox):
        stride = self.params.get('feature_stride')
        output_sz = self.params.get('image_sample_size')

        shifts_x = torch.arange(
            0, output_sz, step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shifts_y = torch.arange(
            0, output_sz, step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        xs, ys = locations[:, 0], locations[:, 1]

        xyxy = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2],
                            bbox[:, 1] + bbox[:, 3]], dim=1)

        l = xs[:, None] - xyxy[:, 0][None]
        t = ys[:, None] - xyxy[:, 1][None]
        r = xyxy[:, 2][None] - xs[:, None]
        b = xyxy[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2).reshape(-1, 4)

        reg_targets_per_im = reg_targets_per_im / output_sz

        sz = output_sz // stride
        nb = bbox.shape[0]
        reg_targets_per_im = reg_targets_per_im.reshape(sz, sz, nb, 4).permute(2, 3, 0, 1)

        return reg_targets_per_im

    def track2(self, image, info: dict = None) -> dict:
        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos2(self.pos2, self.target_scale2),
                                                                      self.target_scale2 * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        feat_head, posx = self.get_backbone_head_feat(backbone_feat)
        test_feat_head, train_feat_head = [], []
        for key, _ in feat_head.items():
            test_feat_head.append(feat_head[key][:, -posx.shape[1]:, ...])
            train_feat_head.append(feat_head[key][:, :-posx.shape[1], ...])

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw, bbox_preds, _ = self.classify_target(train_feat_head, test_feat_head)   # 1 1 18 18,  1 1 4 18 18

        _, scale_ind, s, flag, score_loc, _, _, _ = self.localize_target(scores_raw, sample_pos, sample_scales, back2=True, max_id=None, lame_score=None)
        bbox_raw = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc, scores_raw)
        bbox = self.clip_bbox_to_image_area(bbox_raw, image)

        pos = bbox[:2].flip(0) + bbox[2:].flip(0)/2  # [y + h/2, x + w/2]
        target_sz = bbox[2:].flip(0)

        new_state = torch.cat((pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, target_sz[[1, 0]]))

        score_map = s[scale_ind, ...]
        max_score = score_map.max().cpu().item()
        
        if self.params.get('output_not_found_box', False):
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()
        # print('output_state:', output_state)
        if flag == "normal":
            flag = 3
        elif flag == "hard_negative":
            flag = 2
        elif flag == 'uncertain':
            flag = 1
        elif flag == 'not_found':
            flag = 0
        else:
            flag = None
        out = {'target_bbox': output_state,
               'output_state_2nd': None,
               'object_presence_score': max_score,
               'flag': flag}        # not_found  uncertain hard_negative  normal

        if self.visdom is not None:
            self.visualize_raw_results(score_map)

        return out

    def track_gt(self, image, info: dict = None) -> dict:
        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #
        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos2(self.pos, self.target_scale),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        feat_head, posx = self.get_backbone_head_feat(backbone_feat)
        test_feat_head, train_feat_head = [], []
        for key, _ in feat_head.items():
            test_feat_head.append(feat_head[key][:, -posx.shape[1]:, ...])
            train_feat_head.append(feat_head[key][:, :-posx.shape[1], ...])

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw, bbox_preds, _ = self.classify_target(train_feat_head, test_feat_head)   # 1 1 18 18,  1 1 4 18 18

        _, scale_ind, s, flag, score_loc, _, _, _ = self.localize_target(scores_raw, sample_pos, sample_scales, back2=False, max_id=None, lame_score=None)
        bbox_raw = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc, scores_raw)
        bbox = self.clip_bbox_to_image_area(bbox_raw, image)

        pos = bbox[:2].flip(0) + bbox[2:].flip(0)/2  # [y + h/2, x + w/2]
        target_sz = bbox[2:].flip(0)
        new_state = torch.cat((pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, target_sz[[1, 0]]))

        score_map = s[scale_ind, ...]
        max_score = score_map.max().cpu().item()

        if self.params.get('output_not_found_box', False):
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        if flag == "normal":
            flag = 3
        elif flag == "hard_negative":
            flag = 2
        elif flag == 'uncertain':
            flag = 1
        elif flag == 'not_found':
            flag = 0
        else:
            flag = None
        out = {'target_bbox': output_state,
               'output_state_2nd': None,
               'object_presence_score': max_score,
               'flag': flag}        # not_found  uncertain hard_negative  normal

        return out

    def track(self, image, info: dict = None) -> dict:

        self.debug_info = {}
        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num
        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        feat_head, pos = self.get_backbone_head_feat(backbone_feat)
        test_feat_head, train_feat_head = [], []
        for key, _ in feat_head.items():
            test_feat_head.append(feat_head[key][:, -pos.shape[1]:, ...])
            train_feat_head.append(feat_head[key][:, :-pos.shape[1], ...])

        self.pos_embed = pos
        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw, bbox_preds, cls_test_feat_enc = self.classify_target(train_feat_head, test_feat_head)   # 1 1 18 18,  1 1 4 18 18

        # if self.frame_num > 5:
        #     self.pre_scores.pop(0)
        #     self.cls_test_feat_enc.pop(0)
        # self.pre_scores.append(scores_raw)
        # self.cls_test_feat_enc.append(cls_test_feat_enc)
        # pre_scores = torch.cat(self.pre_scores, dim=1)
        # cls_test_feat = torch.cat(self.cls_test_feat_enc, dim=1)
        # score_lame = self.lame(pre_scores, cls_test_feat)[-1:,...]
        # max_id = score_lame[-1:,...].argmax()
        max_id = None
        score_lame = None

        translation_vec, scale_ind, s, flag, score_loc, translation_vec2, score_loc2, max_score2 = self.localize_target(scores_raw, sample_pos, sample_scales, 
                                                                                                        back2=False, max_id=max_id, lame_score=score_lame)
        bbox_raw = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc, scores_raw)
        bbox = self.clip_bbox_to_image_area(bbox_raw, image)
        
        score_map = s[scale_ind, ...]
        max_score = score_map.max().cpu().item()

        self.pos = bbox[:2].flip(0) + bbox[2:].flip(0)/2  # [y + h/2, x + w/2]
        self.target_sz = bbox[2:].flip(0)
        self.target_scale = torch.sqrt(self.target_sz.prod() / (self.search_area_weight * self.base_target_sz.prod()))
        self.target_scales.append(self.target_scale)
        if self.params.get('search_area_rescaling_at_occlusion', False):
            self.search_area_rescaling()
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain', 'hard_negative'] 
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None
        # scale change with template 0. 
        current_size = bbox[2] * bbox[3] 

        if (self.sz_scale0 / current_size) > self.scale_threshold or (self.sz_scale0 / current_size) < ( 1 / self.scale_threshold):
            self.update_t0_flag = True
        else:
            self.update_t0_flag = False
        # if False:
        if update_flag and self.params.get('update_classifier', False) and max_score > self.params.get('conf_ths', 0.5): 
            # Get train sample
            train_x = test_feat_head[-1][scale_ind:scale_ind+1, ...]
            self.update_nums += 1
            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])
            train_y = self.get_label_function(self.pos, sample_pos[scale_ind, :], sample_scales[scale_ind]).to(self.params.device)
            # update init_tokens and init_image and sample_weights of (token1 & token2)
            self.update_memory(TensorList([train_x]), train_y, target_box, learning_rate, max_score)
            if self.previous_replace_ind[0] == 0:
                self.init_image = im_patches
                self.sz_scale0 = current_size   # update size of current. 
                print('update init_token0 and img0. cls-max-score ----------------', self.frame_num, max_score)
            else:
                # x,y,w,h = new_state
                # im[:,:,int(y):int(y+h), int(x):int(x+w)] = im.mean()
                # im_patches, _ = sample_patch_multiscale(im, self.get_centered_sample_pos(),
                #                                                 self.target_scale * self.params.scale_factors,
                #                                                 self.img_sample_sz, 
                #                                                 mode=self.params.get('border_mode', 'replicate'),
                #                                                 max_scale_change=self.params.get('patch_max_scale_change', None))
                self.init_image2 = im_patches 
                # print('update init_token1 and img1. cls-max-score ----------------', self.frame_num, max_score)

        # second peak result
        if score_loc2 is not None:
            bbox_raw_2nd = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc2, scores_raw)
            bbox_2nd = self.clip_bbox_to_image_area(bbox_raw_2nd, image)    # xl,yl, w, h
            self.pos2 = bbox_2nd[:2].flip(0) + bbox_2nd[2:].flip(0)/2  # [y + h/2, x + w/2]
            self.target_sz2 = bbox_2nd[2:].flip(0)
            self.target_scale2 = torch.sqrt(self.target_sz2.prod() / self.base_target_sz.prod())
            new_state_2nd = torch.cat((self.pos2[[1, 0]] - (self.target_sz2[[1, 0]] - 1) / 2, self.target_sz2[[1, 0]]))
            output_state_2nd = new_state_2nd.tolist()
            max_score2 = max_score2.cpu().item()
        else:
            output_state_2nd = None
            max_score2 = 0

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[0,[1,0]], sample_coords[0,[3,2]] - sample_coords[0,[1,0]] - 1))

        if self.params.get('output_not_found_box', False):
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()
        if flag == "normal":
            flag = 3
        elif flag == "hard_negative":
            flag = 2
        elif flag == 'uncertain':
            flag = 1
        elif flag == 'not_found':
            flag = 0
        else:
            flag = None

        # visualtion
        # if self.update_t0_flag: 
        #     first_point = (math.ceil(output_state[0]), math.ceil(output_state[1]))
        #     last_point = (math.ceil(output_state[0] + output_state[2]), math.ceil(output_state[1] + output_state[3]))
        #     cv.putText(image, str(self.frame_num), (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        #     cv.rectangle(image, first_point, last_point, (0 , 0, 255), 2)
        #     first_point2 = (math.ceil(output_state_2nd[0]), math.ceil(output_state_2nd[1]))
        #     last_point2 = (math.ceil(output_state_2nd[0] + output_state_2nd[2]), math.ceil(output_state_2nd[1] + output_state_2nd[3]))
        #     cv.rectangle(image, first_point2, last_point2, (0 , 255, 0), 2)
        #     cv.imwrite('./frame/images.png', image)

        out = {'target_bbox': output_state,
               'output_state_2nd': output_state_2nd,
               'object_presence_score': max_score,
               'flag': flag,
               'object_presence_score2': max_score2}        # not_found  uncertain hard_negative  normal

        if self.visdom is not None:
            self.visualize_raw_results(score_map)

        return out

    def visualize_raw_results(self, score_map):
        self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
        self.logging_dict['max_score'].append(score_map.max())
        self.visdom.register(torch.tensor(self.logging_dict['max_score']), 'lineplot', 3, 'Max Score')
        self.debug_info['max_score'] = score_map.max().item()
        self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

    def direct_bbox_regression(self, bbox_preds, sample_coords, score_loc, scores_raw):
        shifts_x = torch.arange(
            0, self.img_sample_sz[0], step=16,
            dtype=torch.float32
        )
        shifts_y = torch.arange(
            0, self.img_sample_sz[1], step=16,
            dtype=torch.float32
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + 16 // 2
        xs, ys = locations[:, 0], locations[:, 1]
        s1, s2 = scores_raw.shape[2:]
        xs = xs.reshape(s1, s2)
        ys = ys.reshape(s1, s2)

        ltrb = bbox_preds.permute(0,1,3,4,2)[0,0].cpu() * self.train_img_sample_sz[[0, 1, 0, 1]]
        xs1 = xs - ltrb[:, :, 0]
        xs2 = xs + ltrb[:, :, 2]
        ys1 = ys - ltrb[:, :, 1]
        ys2 = ys + ltrb[:, :, 3]
        sl = score_loc.int()

        x1 = xs1[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[0, 1]
        y1 = ys1[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[0, 0]
        x2 = xs2[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[0, 1]
        y2 = ys2[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[0, 0]
        w = x2 - x1
        h = y2 - y1

        return torch.Tensor([x1, y1, w, h])

    def search_area_rescaling(self):
        if len(self.target_scales) > 0:
            min_scales, max_scales, max_history = 2, 30, 60
            self.target_not_found_counter += 1
            num_scales = max(min_scales, min(max_scales, self.target_not_found_counter))
            target_scales = torch.tensor(self.target_scales)[-max_history:]
            target_scales = target_scales[target_scales >= target_scales[-1]]  # only boxes that are bigger than the `not found`
            target_scales = target_scales[-num_scales:]  # look as many samples into past as not found endures.
            self.target_scale = torch.mean(target_scales) # average bigger boxes from the past

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def get_centered_sample_pos2(self, pos, target_scale):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return pos + ((self.feature_sz + self.kernel_size) % 2) * target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, train_feat_head, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            # train_samples = self.training_samples[0][:self.num_stored_samples[0], ...]
            # train_samples = train_feat_head.reshape(-1, prompt_feat.shape[1], train_feat_head.shape[-1])
            target_labels = self.target_labels[0][:self.num_stored_samples[0], ...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0], :]

            # test_feat = self.net.head.extract_head_feat(sample_x)
            # train_feat = self.net.head.extract_head_feat(train_samples)
            # prompt_feat = self.net.head.extract_head_feat(prompt_feat)
            for i in range(len(train_feat_head)):
                train_feat_head[i] = train_feat_head[i].reshape(-1, 1, *sample_x[i].shape[-2:])
                sample_x[i] = sample_x[i].reshape(-1, 1, *sample_x[i].shape[-2:])
                train_feat_head[i] = self.net.head.extract_head_feat(train_feat_head[i]).squeeze(1)
                sample_x[i] = self.net.head.extract_head_feat(sample_x[i]).squeeze(1)

            train_ltrb = self.encode_bbox(target_boxes)
            # train_ltrb[-1] = train_ltrb[-1].mean() 
            cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc = \
                self.net.head.get_filter_and_features_in_parallel(train_feat_head, sample_x, pos_train=self.pos_embed, pos_test=self.pos_embed, num_gth_frames=self.num_gth_frames,
                                                                  train_label=target_labels, train_ltrb_target=train_ltrb)

            # fuse encoder and decoder features to one feature map
            target_scores = self.net.head.classifier(cls_test_feat_enc, cls_weights)

            # compute the final prediction using the output module
            bbox_preds = self.net.head.bb_regressor(bbreg_test_feat_enc, bbreg_weights)

        return target_scores, bbox_preds, cls_test_feat_enc

    def localize_target(self, scores, sample_pos, sample_scales, back2=False, max_id=None, lame_score=None):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales, back2=back2, max_id=max_id, lame_score=lame_score)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None, max_disp

    def localize_advanced(self, scores, sample_pos, sample_scales, back2=False, max_id=None, lame_score=None):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window
            
        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]

        if max_id != scores.argmax() and max_id != None:
            lame_score = lame_score.reshape(scores.size())
            _, max_disp_lame = dcf.max2d(lame_score)
            temp_disp = (max_disp_lame + max_disp1) // 2
            if (temp_disp == max_disp_lame).all():
                max_disp1 = temp_disp
                max_score1 = scores.flatten()[max_id]
                print('TTA change max_id !!')
            else:
                print('TTA not change predict position!!')

        sample_scale = sample_scales[scale_ind]
        max_disp1 = max_disp1.float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        # Mask out target neighborhood
        if not back2:
            target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)
        else:
            target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz2 / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:   # 0.25
            return translation_vec1, scale_ind, scores_hn, 'not_found', max_disp1, translation_vec2, max_disp2, max_score2
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1, translation_vec2, max_disp2, max_score2
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1, translation_vec2, max_disp2, max_score2

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found', max_disp1, translation_vec2, max_disp2, max_score2

        if not back2:
            prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)
        else:
            prev_target_vec = (self.pos2 - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)
        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:      # if second_max > 0.8 * max
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2   # 0.8 * 16 / 2 = 6.4

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1, translation_vec2, max_disp2, max_score2
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative', max_disp2, translation_vec1, max_disp1, max_score2
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1, translation_vec2, max_disp2, max_score2

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1, translation_vec2, max_disp2, max_score2

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1, translation_vec2, max_disp2, max_score2

        return translation_vec1, scale_ind, scores_hn, 'normal', max_disp1, translation_vec2, max_disp2, max_score2

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            train_imgs = torch.cat((self.init_image, self.init_image2), dim=0)
            backbone_feat = self.net.extract_backbone_avitmp(train_imgs, im_patches)

        return backbone_feat, patch_coords, im_patches

    def get_backbone_head_feat(self, backbone_feat):
        with torch.no_grad():
            return self.net.get_backbone_head_feat(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        self.init_image = im_patches
        # self.init_image2 = im_patches
        self.init_image2 = torch.zeros_like(im_patches)
        init_img = torch.cat((self.init_image, self.init_image2), dim=0)
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone_avitmp(init_img, im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        self.target_boxes[init_target_boxes.shape[0]:,:] = init_target_boxes
        return init_target_boxes + 1

    def init_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, 1,
                                                     self.params.train_feature_size + (int(self.kernel_size[0].item()) + 1) % 2,
                                                     self.params.train_feature_size + (int(self.kernel_size[1].item()) + 1) % 2)
                                         for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.params.get('output_sigma_factor', 1/4)
        self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized img_coords
        target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)

        for target, x in zip(self.target_labels, train_x):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center_pos = self.feature_sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.feature_sz
                target[i, 0, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, sample_center, end_pad=ksz_even)
                # target[i+1, 0, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, sample_center, end_pad=ksz_even)

        return self.target_labels[0][:train_x[0].shape[0]]

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0) + 1
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])     #  / 2

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num-1] = init_sw
            sw[num-1:] = 1 - init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0], ...] = x     # init_token1
            # ts[x.shape[0]:, ...] = x     # init_token2 begin is zeros

        self.update_nums = 0

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate=None, max_score=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate, max_score)
        self.previous_replace_ind = replace_ind

        # # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1, ...] = x

        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y
            # y_memory[ind:ind+1,...] = y.mean()

        # Update bb memory
        self.target_boxes[replace_ind[0], :] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate=None, max_score=None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else 1

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)     # if s_ind=0, when sw[0]<sw[1], r_ind=0
                    r_ind = r_ind.item() + s_ind

                flag1 = (r_ind == 0) and max_score > self.max_score_threshold and self.update_t0_flag
                # flag2 = (r_ind == 0) and max_score > self.max_score_threshold and (self.update_time % 30 < 10) 
                # print('flag1,flag2:', flag1, flag2)

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                    self.update_time +=1
                elif flag1:
                    print('reset sample-weight: r_ind=0, score_threshold, scale_threshold:', self.max_score_threshold, self.scale_threshold)
                    r_ind = 0
                    sw[0] = 1 - lr
                    sw[1] = lr
                else:
                    r_ind = 1
                    sw[r_ind] = sw[r_ind] / (1 - lr)
                    self.update_time +=1
            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init-1].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init-1:].sum()
                sw[:num_init-1] = init_samp_weight / num_init-1

        # print('sample_weights, max_score, r_ind ', sw, max_score, r_ind)
        replace_ind.append(r_ind)
        return replace_ind

    def get_label_function(self, pos, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (pos - sample_pos) / (sample_scale * self.img_support_sz)

        for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5*ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x, pos_train = self.get_backbone_head_feat(init_backbone_feat)
        x = x[11][:, -x[11].shape[1]//3:, ...]
        self.pos_train = pos_train
        # x = x.reshape(x.size(0), x.size(2), self.params.train_feature_size, self.params.train_feature_size)
        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        # self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        self.feature_sz = torch.Tensor([self.params.train_feature_size, self.params.train_feature_size])
        ksz = getattr(self.net.head.filter_predictor, 'filter_size', 1)
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Get target labels for the different augmentations
        self.init_target_labels(TensorList([x]))

        self.num_gth_frames = target_boxes.shape[0]

        if hasattr(self.net.head.filter_predictor, 'num_gth_frames'):
            self.net.head.filter_predictor.num_gth_frames = self.num_gth_frames

        self.init_memory(TensorList([x]))

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
                self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')
