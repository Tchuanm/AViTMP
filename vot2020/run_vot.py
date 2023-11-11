import os
import sys
import argparse


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker

def run_vot2020(tracker_name, tracker_param, run_id=None, debug=0, visdom_info=None):
    # vot2020=True for segmentation mask generation. 
    tracker = Tracker(tracker_name, tracker_param, run_id, vot2020=False)       
    tracker.run_vot2020(debug, visdom_info)


def main():
    parser = argparse.ArgumentParser(description='Run VOT.')
    parser.add_argument('tracker_name', type=str)
    parser.add_argument('tracker_param', type=str)
    parser.add_argument('--run_id', type=int, default=None)

    args = parser.parse_args()

    run_vot2020(args.tracker_name, args.tracker_param, args.run_id)


if __name__ == '__main__':
    main()
