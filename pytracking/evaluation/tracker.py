import importlib
import os
import numpy as np
from collections import OrderedDict
from pytracking.evaluation.environment import env_settings
import time
import cv2 as cv
import copy
from pytracking.utils.visdom import Visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytracking.utils.plotting import draw_figure, overlay_mask
from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
from pathlib import Path
import torch
import math
from pytracking.evaluation.box_ops import *

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}

def box_xywh_to_xyxy(*x):
    x1, y1, w, h = x
    b = [x1, y1, x1 + w, y1 + h]
    return np.array(b)


def trackerlist(name: str, parameter_name: str, run_ids = None, display_name: str = None):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, run_id, display_name) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None, vot2020=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', self.name))
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.visdom = None
        if vot2020:
            self.vot2020 = True
            from segment_anything import SamPredictor, sam_model_registry
            sam = sam_model_registry["vit_h"](checkpoint="***/sam-hq/pretrained_checkpoint/sam_hq_vit_h.pth").to(device='cuda')
            self.predictor = SamPredictor(sam)
        else:
            self.vot2020 = False

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)

                # Show help
                help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                            'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                            'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                            'block list.'
                self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        tracker.visdom = self.visdom
        return tracker

    def run_sequence(self, seq, visualization=None, debug=None, visdom_info=None, multiobj_mode=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, 'visualization', False)
            else:
                visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)
        if visualization_ and self.visdom is None:
            self.init_visualization()

        # Get init information
        init_info = seq.init_info()
        is_single_object = not seq.multiobj_mode

        if multiobj_mode is None:
            multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default' or is_single_object:
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output = self._track_sequence(tracker, seq, init_info)

        return output

    def _track_sequence(self, tracker, seq, init_info):
        
        output = {'target_bbox': [],
                  'time': [],
                  'segmentation': [],
                  'object_presence_score': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        if tracker.params.visualization and self.visdom is None:
            self.visualize(image, init_info.get('init_bbox'))

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'clf_target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask'),
                        'object_presence_score': 1.}

        _store_outputs(out, init_default)

        segmentation = out['segmentation'] if 'segmentation' in out else None
        bboxes = [init_default['target_bbox']]
        if 'clf_target_bbox' in out:
            bboxes.append(out['clf_target_bbox'])
        if 'clf_search_area' in out:
            bboxes.append(out['clf_search_area'])
        if 'segm_search_area' in out:
            bboxes.append(out['segm_search_area'])

        if self.visdom is not None:
            tracker.visdom_draw_tracking(image, bboxes, segmentation)
        elif tracker.params.visualization:
            self.visualize(image, bboxes, segmentation)

        backtrack_times = 0
        subpeak_choose = 0
        not_subpeak_choose = 0
        self.distences = []
        self.wh_scale = []
        self.avg_conf = 0.9
        back_flag = True
        self.min_scale_thread = 0.5
        self.max_scale_thread = 1.3
        self.failure_num = 0
        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)
                    
            image = self._read_image(frame_path)
            start_time = time.time()
            info = seq.frame_info(frame_num)
            info_gt = OrderedDict()
            info['previous_output'] = prev_output
            out = tracker.track(image, info)
            if frame_num == 1:
                prev_output['target_bbox'] = init_info['init_bbox']
                prev_output['object_presence_score'] = 1.0
                self.prev_img = image
            IOU, _ = box_iou(out['target_bbox'], prev_output['target_bbox'])

            if prev_output['object_presence_score']>2:  # >0.5
                if out['flag'] < 2 or IOU<0.0: 
                    backtrack_times += 1
                    info_gt['previous_output'] = prev_output
                    out1 = tracker.track_gt(self.prev_img)
                    IOU1, _ = box_iou(out1['target_bbox'], prev_output['target_bbox'])
                    out2 = tracker.track2(self.prev_img)
                    IOU2, _ = box_iou(out2['target_bbox'], prev_output['target_bbox'])
                    if IOU2>IOU1 and out2['flag'] >= 2: 
                        print('Choose subpeak!!!!!!!!!!', prev_output['object_presence_score'], frame_num, out['object_presence_score'], out['object_presence_score2']) 
                        out['target_bbox'] = out['output_state_2nd']
                        out['object_presence_score'] = 0.5
                        subpeak_choose += 1 
                        tracker.pos = tracker.pos2         # replace search region into f_1
                        tracker.target_sz = tracker.target_sz2
                        tracker.target_scale = tracker.target_scale2
                        tracker.target_scales.pop(-1)
                        tracker.target_scales.append(tracker.target_scale)
                    elif out1['flag'] <= 1 and out2['flag'] <=1 and IOU2==0. and IOU1==0.:
                        out['target_bbox'] = [0,0,0,0]
                        self.failure_num += 1
                        # tracker.pos = tracker.pre_pos      
                        # tracker.target_sz = tracker.pre_target_sz
                        # tracker.target_scale = tracker.pre_target_scale         
                        bboxes = [[0,0,0,0]]

            if out['object_presence_score'] < 0.4 and out['object_presence_score2'] < 0.1 and out['flag']<1:
                self.failure_num += 1
                # tracker.pos = tracker.pre_pos      
                # tracker.target_sz = tracker.pre_target_sz
                # tracker.target_scale = tracker.pre_target_scale                
                bboxes = [[0,0,0,0]]
            elif out['target_bbox'] == [0,0,0,0]:
                self.failure_num += 1
                # tracker.pos = tracker.pre_pos      
                # tracker.target_sz = tracker.pre_target_sz
                # tracker.target_scale = tracker.pre_target_scale                
                bboxes = [[0,0,0,0]]
            elif out['object_presence_score'] < 0.5 and out['object_presence_score2'] < 0.2 and out['flag'] < 2:
                self.failure_num += 1
                # tracker.pos = tracker.pre_pos      
                # tracker.target_sz = tracker.pre_target_sz
                # tracker.target_scale = tracker.pre_target_scale                
                bboxes = [[0,0,0,0]]
            else:
                self.failure_num = 0
                tracker.pre_pos = tracker.pos         # self.pos2
                tracker.pre_target_sz = tracker.target_sz
                tracker.pre_target_scale = tracker.target_scale
                prev_output = OrderedDict(out)
                bboxes = [out['target_bbox']]

            # failure 5 times, search scale=2 times, 
            if self.failure_num >= 3:
                tracker.params.scale_factors = torch.ones(1) * 2
            else:
                tracker.params.scale_factors = torch.ones(1)
 
            self.prev_img = image
            self.prev_info = info
            _store_outputs(out, {'time': time.time() - start_time})

            segmentation = out['segmentation'] if 'segmentation' in out else None

            if 'clf_target_bbox' in out:
                bboxes.append(out['clf_target_bbox'])
            if 'clf_search_area' in out:
                bboxes.append(out['clf_search_area'])
            if 'segm_search_area' in out:
                bboxes.append(out['segm_search_area'])

            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, bboxes, segmentation)
            elif tracker.params.visualization:
                self.visualize(image, bboxes, segmentation)

        print('backtrack, subpeak, not choose subpeak time ', backtrack_times, subpeak_choose, not_subpeak_choose)

        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        # next two lines are needed for oxuva output format.
        output['image_shape'] = image.shape[:2]
        output['object_presence_score_threshold'] = tracker.params.get('object_presence_score_threshold', 0.55)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the video file.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': OrderedDict({1: box}), 'init_object_ids': [1, ], 'object_ids': [1, ],
                    'sequence_object_ids': [1, ]}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox'][1]]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_webcam(self, debug=None, visdom_info=None):
        """Run the tracker with the webcam.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.new_init = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'init'
                    self.new_init = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        next_object_id = 1
        sequence_object_ids = []
        prev_output = OrderedDict()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            info = OrderedDict()
            info['previous_output'] = prev_output

            if ui_control.new_init:
                ui_control.new_init = False
                init_state = ui_control.get_bb()

                info['init_object_ids'] = [next_object_id, ]
                info['init_bbox'] = OrderedDict({next_object_id: init_state})
                sequence_object_ids.append(next_object_id)

                next_object_id += 1

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)

            if len(sequence_object_ids) > 0:
                info['sequence_object_ids'] = sequence_object_ids
                out = tracker.track(frame, info)
                prev_output = OrderedDict(out)

                if 'segmentation' in out:
                    frame_disp = overlay_mask(frame_disp, out['segmentation'])

                if 'target_bbox' in out:
                    for obj_id, state in out['target_bbox'].items():
                        state = [int(s) for s in state]
                        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                     _tracker_disp_colors[obj_id], 5)

            # Put text
            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 85), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                next_object_id = 1
                sequence_object_ids = []
                prev_output = OrderedDict()

                info = OrderedDict()

                info['object_ids'] = []
                info['init_object_ids'] = []
                info['init_bbox'] = OrderedDict()
                tracker.initialize(frame, info)
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def run_vot2020(self, debug=None, visdom_info=None):
        output_segmentation = self.vot2020  # True for segment
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        if debug is None:
            visualization_ = getattr(params, 'visualization', False)
        else:
            visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)

        tracker = self.create_tracker(params)
        tracker.initialize_features()

        import pytracking.evaluation.vot2020 as vot

        def _convert_anno_to_list(vot_anno):
            vot_anno = [vot_anno[0], vot_anno[1], vot_anno[2], vot_anno[3]]
            return vot_anno

        def _convert_image_path(image_path):
            return image_path

        """Run tracker on VOT."""

        if output_segmentation:
            
            handle = vot.VOT("mask")
        else:
            handle = vot.VOT("rectangle")

        vot_anno = handle.region()

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)

        image = self._read_image(image_path)

        if output_segmentation:
            vot_anno_mask = vot.make_full_size(vot_anno, (image.shape[1], image.shape[0]))
            bbox = masks_to_bboxes(torch.from_numpy(vot_anno_mask), fmt='t').squeeze().tolist()
        else:
            bbox = _convert_anno_to_list(vot_anno)
            vot_anno_mask = None

        out = tracker.initialize(image, {'init_mask': vot_anno_mask, 'init_bbox': bbox})
        failure_num = 0

        if out is None:
            out = {}
        # init_frame 
        prev_output = OrderedDict(out)
        prev_output['target_bbox'] = bbox
        prev_output['object_presence_score'] = 1
        prev_output['flag'] = 3
        prev_img = image

        # Track
        while True:
            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = self._read_image(image_path)

            info = OrderedDict()
            info['previous_output'] = prev_output

            out = tracker.track(image, info)

            IOU, _ = box_iou(out['target_bbox'], prev_output['target_bbox'])
            if prev_output['object_presence_score']>0.5:
                if out['flag'] < 2 or IOU==0.0: 
                    out1 = tracker.track_gt(prev_img)
                    IOU1, _ = box_iou(out1['target_bbox'], prev_output['target_bbox'])
                    if IOU1 < 0.1:
                        out2 = tracker.track2(prev_img)
                        IOU2, _ = box_iou(out2['target_bbox'], prev_output['target_bbox'])
                        if IOU2>IOU1 and out2['flag']>=2: 
                            out['target_bbox'] = out['output_state_2nd']
                            out['object_presence_score'] = 0.5
                            tracker.pos = tracker.pos2         # replace search region into f_1
                            tracker.target_sz = tracker.target_sz2
                            tracker.target_scale = tracker.target_scale2
                            tracker.target_scales.pop(-1)
                            tracker.target_scales.append(tracker.target_scale2)
                            
                        elif out1['flag'] <= 1 and out2['flag'] <=1 and IOU2==0. and IOU1==0.:
                            failure_num += 1
                            out['target_bbox'] = [0,0,0,0]
                            # out['flag'] = 0

            if out['object_presence_score']<0.4 and out['object_presence_score2']<0.1 and out['flag']<=1:
                failure_num += 1
                # tracker.pos = tracker.pre_pos      
                # tracker.target_sz = tracker.pre_target_sz
                # tracker.target_scale = tracker.pre_target_scale 
                out['target_bbox'] = [0,0,0,0]
                # out['flag'] = 0
            elif out['object_presence_score'] < 0.5 and out['object_presence_score2'] < 0.2 and out['flag'] < 2:
                failure_num += 1
                out['target_bbox'] = [0,0,0,0]
            else:
                failure_num  = 0
                tracker.pre_pos = tracker.pos         # pos2
                tracker.pre_target_sz = tracker.target_sz
                tracker.pre_target_scale = tracker.target_scale
                prev_output = OrderedDict(out)

            if failure_num >= 3:
                tracker.params.scale_factors = torch.ones(1) * 2
            else:
                tracker.params.scale_factors = torch.ones(1)

            prev_img = image

            # prev_output = OrderedDict(out)   
            if output_segmentation:
                bbox = box_xywh_to_xyxy(*out['target_bbox'])
                self.predictor.set_image(image)
                # print('bbox:', bbox)
                masks, quality, _ = self.predictor.predict(box=bbox)
                # if quality.mean() < 0.80 and out['flag']<=1:     #  and flag<=1  absent
                    # pred = np.zeros_like(image[:,:,0])
                # else:
                pred = masks[0].astype(np.uint8)
            else:
                state = out['target_bbox']
                pred = vot.Rectangle(*state)
            handle.report(pred, 1.0)

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif tracker.params.visualization:
                self.visualize(image, out['target_bbox'], segmentation)


    def init_vots(self, image, region, debug=None, visdom_info=None):
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        if debug is None:
            visualization_ = getattr(params, 'visualization', False)
        else:
            visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)

        self.tracker = self.create_tracker(params)
        self.tracker.initialize_features()

        self.output_segmentation = False

        image = self._read_image(image)

        out = self.tracker.initialize(image, {'init_mask': region, 'init_bbox': region})

        if out is None:
            out = {}
        self.info = OrderedDict()
        self.info['init_bbox'] = region
        self.prev_output = OrderedDict(out)
        self.info_gt = OrderedDict()
        self.not_subpeak_choose = 0
        self.backtrack_times = 0
        self.avg_conf = 0.7
        self.wh_scale = []
        self.distences = []
        self.min_scale_thread = 0.6
        self.max_scale_thread = 1
        self.failure_num = 0


    def run_vots1(self, image, info, debug=False):
        if self.tracker.frame_num == 1:
            print('self.tracker.frame_num: ', self.tracker.frame_num)
            self.prev_output['target_bbox'] = self.info['init_bbox']
            self.prev_output['object_presence_score'] = 1
            self.prev_img = image
        print('self.prev_output[target_bbox]:', self.prev_output['target_bbox'])
        out = self.tracker.track(image, info)
        IOU, _ = box_iou(out['target_bbox'], self.prev_output['target_bbox'])
        
        ### 0. CycleTrack
        if IOU < 0.1: 
            self.backtrack_times += 1
            self.info_gt['previous_output'] = self.prev_output
            out1 = self.tracker.track_gt(self.prev_img)
            IOU1, _ = box_iou(out1['target_bbox'], self.prev_output['target_bbox'])
            out2 = self.tracker.track2(self.prev_img)
            IOU2, _ = box_iou(out2['target_bbox'], self.prev_output['target_bbox'])
            print('IOU2, IOU1: ',IOU2,IOU1)
            # flag1 = IOU2>0.5 and IOU1<0.3 and out2['object_presence_score'] > 0.3 and self.prev_output['object_presence_score']>0.5
            # flag2 = IOU2>0.7 and IOU1>0.7 and out2['object_presence_score'] > out1['object_presence_score'] and self.prev_output['object_presence_score']>0.5
            # print('flag1:', flag1) 
            # conf2_bigger = (out['object_presence_score'] + out1['object_presence_score'] < out['object_presence_score2'] + out2['object_presence_score'])
            if IOU2>IOU1 and out2['object_presence_score']>0.5 and self.prev_output['object_presence_score']>0.5: 
                pre_out_new = copy.deepcopy(out)
                pre_out_new['target_bbox'] = out['output_state_2nd']
                pre_out_new['object_presence_score'] = out['object_presence_score2']
                pre_out_new['object_presence_score2'] =  out['object_presence_score']
                pre_out_new['output_state_2nd'] = out['target_bbox']
                out = pre_out_new
                
                self.tracker.pos = self.tracker.pos2         # replace search region into f_1
                self.tracker.target_sz = self.tracker.target_sz2
                self.tracker.target_scale = self.tracker.target_scale2
                self.tracker.target_scales.pop(-1)
                self.tracker.target_scales.append(self.tracker.target_scale)

        # debug = False
        # if debug: 
        #     x, y, width, height = out['target_bbox']
        #     cv.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 4)  
        #     cv.imwrite("./visulazation/bbox/bbox_image_{}.jpg".format(self.tracker.frame_num), image)
        
        # 1. Dynamic search region
        if False:
            cent_dist, wh_scale1 = center_distence(out['target_bbox'], self.prev_output['target_bbox'])
            self.distences.append(cent_dist) # 2 is scale
            if self.tracker.frame_num > 50:
                self.wh_scale.pop(-1) 
                self.wh_scale.append(wh_scale1)
            else: 
                self.wh_scale.append(wh_scale1)
            
            avg_wh_scale = sum(self.wh_scale) / len(self.wh_scale)
            final_search_scale = wh_scale1    # * speed_change_rate
            if final_search_scale < self.min_scale_thread: 
                self.tracker.search_area_weight = 1 / math.sqrt(self.min_scale_thread)  # 
                print('< min, wh_scale1:', torch.sqrt(avg_wh_scale), final_search_scale)
            elif final_search_scale > self.max_scale_thread: 
                self.tracker.search_area_weight =  1 / math.sqrt(self.max_scale_thread) # 6
                print('> max, wh_scale1:', torch.sqrt(avg_wh_scale), final_search_scale)
            else:
                self.tracker.search_area_weight = 1 / math.sqrt(final_search_scale)

        self.prev_output = OrderedDict(out)
        if out['object_presence_score'] < 0.3 and out['object_presence_score2'] < 0.1:
            state = None
        else:
            state = out['target_bbox']
        return state


    def run_vots0_origin(self, image, info):
        if self.tracker.frame_num == 1:
            print('self.tracker.frame_num: ', self.tracker.frame_num)
            self.prev_output['target_bbox'] = self.info['init_bbox']
            self.prev_output['object_presence_score'] = 1
            self.prev_output['flag'] = 3
            self.prev_img = image

        self.info['previous_output'] = self.prev_output
        out = self.tracker.track(image, info)
        if out['object_presence_score'] < 0.2:
            return None
        ### 0. CycleTrack method 
        IOU, _ = box_iou(out['target_bbox'], self.prev_output['target_bbox'])
        if out['object_presence_score']<0.5 or IOU<0.2:
            self.backtrack_times += 1 
            # running CycleTrack
            out1 = self.tracker.track_gt(self.prev_img)
            IOU1, _ = box_iou(out1['target_bbox'], self.prev_output['target_bbox'])
            out2 = self.tracker.track2(self.prev_img)
            IOU2, _ = box_iou(out2['target_bbox'], self.prev_output['target_bbox'])

            if IOU2>IOU1 and out2['object_presence_score'] > 0.5: 
                out['target_bbox'] = out['output_state_2nd']
                out['object_presence_score'] = out['object_presence_score2']
                self.tracker.pos = self.tracker.pos2         # replace search region into f_1
                self.tracker.target_sz = self.tracker.target_sz2
                self.tracker.target_scale = self.tracker.target_scale2
                self.tracker.target_scales.pop(-1)
                self.tracker.target_scales.append(self.tracker.target_scale2)
                return out['target_bbox']

        self.prev_output = OrderedDict(out)
        if out['object_presence_score'] < 0.35 and out['object_presence_score2'] < 0.1 and out['flag'] < 1:
            state = None
        else:
            state = out['target_bbox']
        return state


    def run_test0(self, image, info):
        if self.tracker.frame_num == 1:
            print('self.tracker.frame_num: ', self.tracker.frame_num)
            self.prev_output['target_bbox'] = self.info['init_bbox']
            self.prev_output['object_presence_score'] = 1
            self.prev_output['flag'] = 3
            self.prev_img = image

        if self.failure_num >= 3:
            self.tracker.params.scale_factors = torch.ones(1) * 2
        else:
            self.tracker.params.scale_factors = torch.ones(1)

        self.info['previous_output'] = self.prev_output
        out = self.tracker.track(image, info)
        if out['object_presence_score'] < 0.1:
            self.failure_num += 1
            self.tracker.pos = self.tracker.pre_pos      
            self.tracker.target_sz = self.tracker.pre_target_sz
            self.tracker.target_scale = self.tracker.pre_target_scale 
            return None, out['flag']

        ### 0. CycleTrack method 
        if self.prev_output['object_presence_score']>0.5: 
            IOU, _ = box_iou(out['target_bbox'], self.prev_output['target_bbox'])
            if out['flag'] < 2 and out['object_presence_score']<0.5: 
                out1 = self.tracker.track_gt(self.prev_img)
                IOU1, _ = box_iou(out1['target_bbox'], self.prev_output['target_bbox'])
                if IOU1 == 0.:
                    out2 = self.tracker.track2(self.prev_img)
                    IOU2, _ = box_iou(out2['target_bbox'], self.prev_output['target_bbox'])
                    if IOU2>0 and out2['flag'] > 1: 
                        out['target_bbox'] = out['output_state_2nd']
                        # out['object_presence_score'] = 0.5
                        out['flag'] = out2['flag']
                        self.tracker.pos = self.tracker.pos2         # replace search region into f_1
                        self.tracker.target_sz = self.tracker.target_sz2
                        self.tracker.target_scale = self.tracker.target_scale2
                        self.tracker.target_scales.pop(-1)
                        self.tracker.target_scales.append(self.tracker.target_scale2)
                        return out['target_bbox'], out['flag']
                    elif out1['flag'] <= 1 and out2['flag'] <=1 and out['flag'] <=1 and IOU1==0.0 and IOU2==0.0:
                        self.failure_num += 1
                        self.tracker.pos = self.tracker.pre_pos      
                        self.tracker.target_sz = self.tracker.pre_target_sz
                        self.tracker.target_scale = self.tracker.pre_target_scale 
                        return None, out['flag']

        if out['object_presence_score']<0.4 and out['object_presence_score2']<0.1 and out['flag']<=1 and self.prev_output['flag']<=1:
            self.failure_num += 1
            self.tracker.pos = self.tracker.pre_pos      
            self.tracker.target_sz = self.tracker.pre_target_sz
            self.tracker.target_scale = self.tracker.pre_target_scale 
            return None, 0
        else:
            self.failure_num  = 0
            self.tracker.pre_pos = self.tracker.pos         # self.pos2
            self.tracker.pre_target_sz = self.tracker.target_sz
            self.tracker.pre_target_scale = self.tracker.target_scale
            self.prev_output = OrderedDict(out)
            state = out['target_bbox']
            return state, out['flag']


    def run_test1(self, image, info):
        if self.tracker.frame_num == 1:
            print('self.tracker.frame_num: ', self.tracker.frame_num)
            self.prev_output['target_bbox'] = self.info['init_bbox']
            self.prev_output['object_presence_score'] = 1
            self.prev_output['flag'] = 3
            self.prev_img = image

        if self.failure_num >= 3:
            self.tracker.params.scale_factors = torch.ones(1) * 2
        else:
            self.tracker.params.scale_factors = torch.ones(1)

        self.info['previous_output'] = self.prev_output
        out = self.tracker.track(image, info)
        IOU, _ = box_iou(out['target_bbox'], self.prev_output['target_bbox'])
        if self.prev_output['object_presence_score']>0.5:
            if out['flag'] < 2 or IOU<0.0: 
                out1 = self.tracker.track_gt(self.prev_img)
                IOU1, _ = box_iou(out1['target_bbox'], self.prev_output['target_bbox'])
                out2 = self.tracker.track2(self.prev_img)
                IOU2, _ = box_iou(out2['target_bbox'], self.prev_output['target_bbox'])
                if IOU2>IOU1 and out2['flag']>1: 
                    out['target_bbox'] = out['output_state_2nd']
                    out['object_presence_score'] = 0.5
                    self.tracker.pos = self.tracker.pos2         # replace search region into f_1
                    self.tracker.target_sz = self.tracker.target_sz2
                    self.tracker.target_scale = self.tracker.target_scale2
                    self.tracker.target_scales.pop(-1)
                    self.tracker.target_scales.append(self.tracker.target_scale2)
                    state=out['target_bbox']
                elif out1['flag'] <= 1 and out2['flag'] <=1 and IOU2==0. and IOU1==0.:
                    state=None

        self.prev_output = OrderedDict(out)
        if out['object_presence_score']<0.4 and out['object_presence_score2']<0.1 and out['flag']<=1 and self.prev_output['flag']<=1:
            state = None
        elif out['object_presence_score'] < 0.35 and out['object_presence_score2'] < 0.2 and out['flag']<=1:
            state = None
        else:
            state = out['target_bbox']

        if state == None:
            self.failure_num += 1
            self.tracker.pos = self.tracker.pre_pos      
            self.tracker.target_sz = self.tracker.pre_target_sz
            self.tracker.target_scale = self.tracker.pre_target_scale 
        else:
            self.failure_num  = 0
            self.tracker.pre_pos = self.tracker.pos         # self.pos2
            self.tracker.pre_target_sz = self.tracker.target_sz
            self.tracker.pre_target_scale = self.tracker.target_scale
            self.prev_output = OrderedDict(out) 
        return state, out['flag']


    def run_vots2023_18(self, image, info):
        if self.tracker.frame_num == 1:
            # print('self.tracker.frame_num: ', self.tracker.frame_num)
            self.prev_output['target_bbox'] = self.info['init_bbox']
            self.prev_output['object_presence_score'] = 1
            self.prev_output['flag'] = 3
            self.prev_img = image

        if self.failure_num >= 5:
            self.tracker.params.scale_factors = torch.ones(1) * 2
        else:
            self.tracker.params.scale_factors = torch.ones(1)
        self.info['previous_output'] = self.prev_output
        out = self.tracker.track(image, info)
        
        if out['object_presence_score'] < 0.35 and out['object_presence_score2'] < 0.1:
            self.failure_num += 1
            # self.tracker.pos = self.tracker.pre_pos      
            # self.tracker.target_sz = self.tracker.pre_target_sz
            # self.tracker.target_scale = self.tracker.pre_target_scale 
            return None, 0

        IOU, _ = box_iou(out['target_bbox'], self.prev_output['target_bbox'])
        if self.prev_output['object_presence_score']>0.5:
            if out['flag'] < 3 or out['object_presence_score']<0.5 or IOU==0.0: 
                out1 = self.tracker.track_gt(self.prev_img)
                IOU1, _ = box_iou(out1['target_bbox'], self.prev_output['target_bbox'])
                if IOU1 < 0.1:
                    out2 = self.tracker.track2(self.prev_img)
                    IOU2, _ = box_iou(out2['target_bbox'], self.prev_output['target_bbox'])
                    if IOU2>IOU1 and out2['flag']==3 and out1['flag'] != 3 and out1['object_presence_score'] < 0.5: 
                        out['target_bbox'] = out['output_state_2nd']
                        out['object_presence_score'] = 0.5
                        self.tracker.pos = self.tracker.pos2         # replace search region into f_1
                        self.tracker.target_sz = self.tracker.target_sz2
                        self.tracker.target_scale = self.tracker.target_scale2
                        self.tracker.target_scales.pop(-1)
                        self.tracker.target_scales.append(self.tracker.target_scale2)
                        return out['target_bbox'], 3
                    elif out1['flag'] <= 1 and out2['flag'] <=1 and IOU2==0. and IOU1==0.:
                        self.failure_num += 1
                        # self.tracker.pos = self.tracker.pre_pos      
                        # self.tracker.target_sz = self.tracker.pre_target_sz
                        # self.tracker.target_scale = self.tracker.pre_target_scale 
                        return None, 0

        if out['object_presence_score']<0.5 and out['object_presence_score2']<0.1 and out['flag']<=1:
            self.failure_num += 1
            # self.tracker.pos = self.tracker.pre_pos      
            # self.tracker.target_sz = self.tracker.pre_target_sz
            # self.tracker.target_scale = self.tracker.pre_target_scale 
            return None, 1
        else:
            self.failure_num  = 0
            # self.tracker.pre_pos = self.tracker.pos         # self.pos2
            # self.tracker.pre_target_sz = self.tracker.target_sz
            # self.tracker.pre_target_scale = self.tracker.target_scale
            self.prev_output = OrderedDict(out)
            return out['target_bbox'], out['flag']


    def run_vots2023_17(self, image, info):
        if self.tracker.frame_num == 1:
            print('self.tracker.frame_num: ', self.tracker.frame_num)
            self.prev_output['target_bbox'] = self.info['init_bbox']
            self.prev_output['object_presence_score'] = 1
            self.prev_img = image
        # scale jitter 
        if self.failure_num >= 3:
            self.tracker.params.scale_factors = torch.ones(1) * 2
        else:
            self.tracker.params.scale_factors = torch.ones(1)
        
        self.info['previous_output'] = self.prev_output
        out = self.tracker.track(image, info)
        IOU, _ = box_iou(out['target_bbox'], self.prev_output['target_bbox'])

        if self.prev_output['object_presence_score']>0.5:
            if out['flag'] <= 1 or IOU == 0.:
                out1 = self.tracker.track_gt(self.prev_img)
                IOU1, _ = box_iou(out1['target_bbox'], self.prev_output['target_bbox'])
                if IOU1 < 0.3: 
                    out2 = self.tracker.track2(self.prev_img)
                    IOU2, _ = box_iou(out2['target_bbox'], self.prev_output['target_bbox'])
                    if IOU2>IOU1 and out2['object_presence_score']>out1['object_presence_score']: 
                        out['target_bbox'] = out['output_state_2nd']
                        out['object_presence_score'] = 0.5
                        out['flag'] = 2
                        self.tracker.pos = self.tracker.pos2         # replace search region into f_1
                        self.tracker.target_sz = self.tracker.target_sz2
                        self.tracker.target_scale = self.tracker.target_scale2
                        self.tracker.target_scales.pop(-1)
                        self.tracker.target_scales.append(self.tracker.target_scale)
                    elif out2['object_presence_score'] < 0.2 and out1['object_presence_score'] < 0.2: 
                        self.failure_num += 1
                        return None, 1

        self.prev_output = OrderedDict(out)
        if out['object_presence_score'] < 0.4 and out['object_presence_score2'] < 0.1 and out['flag']<2:
            self.failure_num += 1
            return None, 1
        else:
            self.failure_num =0
            return out['target_bbox'], out['flag']


    def run_vots2023_16(self, image, info):
        if self.tracker.frame_num == 1:
            # print('self.tracker.frame_num: ', self.tracker.frame_num)
            self.prev_output['target_bbox'] = self.info['init_bbox']
            self.prev_output['object_presence_score'] = 1
            self.prev_output['flag'] = 3
            self.prev_img = image

        if self.failure_num >= 3:
            self.tracker.params.scale_factors = torch.ones(1) * 2
        else:
            self.tracker.params.scale_factors = torch.ones(1)
        self.info['previous_output'] = self.prev_output
        out = self.tracker.track(image, info)

        if out['object_presence_score'] < 0.3 and out['object_presence_score2'] < 0.1:
            self.failure_num += 1
            self.tracker.pos = self.tracker.pre_pos      
            self.tracker.target_sz = self.tracker.pre_target_sz
            self.tracker.target_scale = self.tracker.pre_target_scale 
            return None, out['flag']

        IOU, _ = box_iou(out['target_bbox'], self.prev_output['target_bbox'])
        if self.prev_output['object_presence_score']>0.5:
            if out['flag'] < 2 or out['object_presence_score']<0.5 or IOU==0.0: 
            # if out['flag'] <= 2 and out['object_presence_score']<0.7 and IOU==0.0: 
                out1 = self.tracker.track_gt(self.prev_img)
                IOU1, _ = box_iou(out1['target_bbox'], self.prev_output['target_bbox'])
                if IOU1 < 0.1:
                    out2 = self.tracker.track2(self.prev_img)
                    IOU2, _ = box_iou(out2['target_bbox'], self.prev_output['target_bbox'])
                    if IOU2>IOU1 and out2['flag']==3 and out1['flag'] != 3 and out1['object_presence_score'] < 0.5: 
                    # if IOU2>IOU1 and out2['flag']>=2 and out1['flag']<2: 
                        out['target_bbox'] = out['output_state_2nd']
                        out['object_presence_score'] = 0.5
                        self.tracker.pos = self.tracker.pos2         # replace search region into f_1
                        self.tracker.target_sz = self.tracker.target_sz2
                        self.tracker.target_scale = self.tracker.target_scale2
                        self.tracker.target_scales.pop(-1)
                        self.tracker.target_scales.append(self.tracker.target_scale2)
                        return out['target_bbox'], 2
                    elif out1['flag'] <= 1 and out2['flag'] <=1 and IOU2==0. and IOU1==0.:
                        self.failure_num += 1
                        self.tracker.pos = self.tracker.pre_pos      
                        self.tracker.target_sz = self.tracker.pre_target_sz
                        self.tracker.target_scale = self.tracker.pre_target_scale 
                        return None, 0

        if out['object_presence_score']<0.5 and out['object_presence_score2']<0.1 and out['flag']<=1:
            self.failure_num += 1
            self.tracker.pos = self.tracker.pre_pos      
            self.tracker.target_sz = self.tracker.pre_target_sz
            self.tracker.target_scale = self.tracker.pre_target_scale 
            return out['target_bbox'], 1
        else:
            self.failure_num  = 0
            self.tracker.pre_pos = self.tracker.pos         # self.pos2
            self.tracker.pre_target_sz = self.tracker.target_sz
            self.tracker.pre_target_scale = self.tracker.target_scale
            self.prev_output = OrderedDict(out)
            return out['target_bbox'], out['flag']


    def run_vots2023_15(self, image, info):
        if self.tracker.frame_num == 1:
            # print('self.tracker.frame_num: ', self.tracker.frame_num)
            self.prev_output['target_bbox'] = self.info['init_bbox']
            self.prev_output['object_presence_score'] = 1
            self.prev_output['flag'] = 3
            self.prev_img = image

        if self.failure_num >= 5:
            self.tracker.params.scale_factors = torch.ones(1) * 2
        else:
            self.tracker.params.scale_factors = torch.ones(1)
        self.info['previous_output'] = self.prev_output
        out = self.tracker.track(image, info)
        if out['object_presence_score'] < 0.25 and out['object_presence_score2'] < 0.1:
            self.failure_num += 1
            self.tracker.pos = self.tracker.pre_pos      
            self.tracker.target_sz = self.tracker.pre_target_sz
            self.tracker.target_scale = self.tracker.pre_target_scale 
            return None, out['flag']

        IOU, _ = box_iou(out['target_bbox'], self.prev_output['target_bbox'])
        if self.prev_output['object_presence_score']>0.5:
            if out['flag'] < 3 or out['object_presence_score']<0.5 or IOU==0.0: 
                out1 = self.tracker.track_gt(self.prev_img)
                IOU1, _ = box_iou(out1['target_bbox'], self.prev_output['target_bbox'])
                if IOU1 < 0.1:
                    out2 = self.tracker.track2(self.prev_img)
                    IOU2, _ = box_iou(out2['target_bbox'], self.prev_output['target_bbox'])
                    if IOU2>IOU1 and out2['flag']==3 and out1['flag'] != 3 and out1['object_presence_score'] < 0.5: 
                        out['target_bbox'] = out['output_state_2nd']
                        out['object_presence_score'] = 0.5
                        self.tracker.pos = self.tracker.pos2         # replace search region into f_1
                        self.tracker.target_sz = self.tracker.target_sz2
                        self.tracker.target_scale = self.tracker.target_scale2
                        self.tracker.target_scales.pop(-1)
                        self.tracker.target_scales.append(self.tracker.target_scale2)
                        return out['target_bbox'], 3
                    elif out1['flag'] <= 1 and out2['flag'] <=1 and IOU2==0. and IOU1==0.:
                        self.failure_num += 1
                        self.tracker.pos = self.tracker.pre_pos      
                        self.tracker.target_sz = self.tracker.pre_target_sz
                        self.tracker.target_scale = self.tracker.pre_target_scale 
                        return None, 0

        if out['object_presence_score']<0.5 and out['object_presence_score2']<0.1 and out['flag']<=1:
            self.failure_num += 1
            self.tracker.pos = self.tracker.pre_pos      
            self.tracker.target_sz = self.tracker.pre_target_sz
            self.tracker.target_scale = self.tracker.pre_target_scale 
            return None, 1
        else:
            self.failure_num  = 0
            self.tracker.pre_pos = self.tracker.pos         # self.pos2
            self.tracker.pre_target_sz = self.tracker.target_sz
            self.tracker.pre_target_scale = self.tracker.target_scale
            self.prev_output = OrderedDict(out)
            return out['target_bbox'], out['flag']


    def run_vot(self, debug=None, visdom_info=None):
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        if debug is None:
            visualization_ = getattr(params, 'visualization', False)
        else:
            visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)

        tracker = self.create_tracker(params)
        tracker.initialize_features()

        import pytracking.evaluation.vot as vot

        def _convert_anno_to_list(vot_anno):
            vot_anno = [vot_anno[0][0][0], vot_anno[0][0][1], vot_anno[0][1][0], vot_anno[0][1][1],
                        vot_anno[0][2][0], vot_anno[0][2][1], vot_anno[0][3][0], vot_anno[0][3][1]]
            return vot_anno

        def _convert_image_path(image_path):
            image_path_new = image_path[20:- 2]
            return "".join(image_path_new)

        """Run tracker on VOT."""

        handle = vot.VOT("polygon")

        vot_anno_polygon = handle.region()
        vot_anno_polygon = _convert_anno_to_list(vot_anno_polygon)

        init_state = convert_vot_anno_to_rect(vot_anno_polygon, tracker.params.vot_anno_conversion_type)

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)

        image = self._read_image(image_path)
        tracker.initialize(image, {'init_bbox': init_state})

        # Track
        while True:
            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = self._read_image(image_path)
            out = tracker.track(image)
            state = out['target_bbox']

            handle.report(vot.Rectangle(state[0], state[1], state[2], state[3]))

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif tracker.params.visualization:
                self.visualize(image, out['target_bbox'], segmentation)

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()
        return params


    def init_visualization(self):
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state, segmentation=None):
        self.ax.cla()
        self.ax.imshow(image)
        if segmentation is not None:
            self.ax.imshow(segmentation, alpha=0.5)

        if isinstance(state, (OrderedDict, dict)):
            boxes = [v for k, v in state.items()]
        elif isinstance(state, list):
            boxes = state
        else:
            boxes = (state,)

        for i, box in enumerate(boxes, start=1):
            col = _tracker_disp_colors[i]
            col = [float(c) / 255.0 for c in col]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=col, facecolor='none')
            self.ax.add_patch(rect)

        if getattr(self, 'gt_state', None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        draw_figure(self.fig)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def _read_image(self, image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)


