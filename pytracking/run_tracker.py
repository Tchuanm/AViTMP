import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker
import torch

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                visdom_info=None, num_gpus=4):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    dataset = get_dataset(dataset_name) 

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='avitmp', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='avitmp', help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='uav', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()
    # for reduce cpu cover.
    # cpu_num = 1  # Num of CPUs you want to use
    # os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, args.num_gpus)

    # val_auc 
    if args.dataset_name == 'got10k_test':
        os.system('python zip_got10k.py')
    else:
        os.system('python val_auc.py --tracker_param {} --dataset_name {}'.format(args.tracker_param, args.dataset_name))


if __name__ == '__main__':

    main()
