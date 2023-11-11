import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]
import argparse


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.analysis.plot_results import plot_results, print_results
from pytracking.evaluation import get_dataset, trackerlist



def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='avitmp', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='avitmp50', help='Name of parameter file.')
    parser.add_argument('--dataset_name', type=str, default='oxuva', help='Name of dataset file.')
    args = parser.parse_args()

    trackers = []
    trackers.extend(trackerlist('avitmp', args.tracker_param, None, args.tracker_param))
    if  args.dataset_name == "uav":
        dataset = get_dataset('uav')
        plot_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
    if  args.dataset_name == "nfs":
        dataset = get_dataset('nfs')
        plot_results(trackers, dataset, 'NFS', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                    skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
    if  args.dataset_name == "otb":
        dataset = get_dataset('otb')
        plot_results(trackers, dataset, 'OTB100', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                    skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
    if  args.dataset_name == "lasot":
        dataset = get_dataset('lasot')
        plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                    skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
    if  args.dataset_name == "oxuva":
        dataset = get_dataset('oxuva_test')
        plot_results(trackers, dataset, 'OxUvA', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                    skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
    if  args.dataset_name == "avist":
        dataset = get_dataset('avist')
        # plot_results(trackers, dataset, 'AVisT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
        #             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
        print_results(trackers, get_dataset('avist'), 'avist', merge_results=True, force_evaluation=True,
                    skip_missing_seq=False, exclude_invalid_frames=False, plot_types=('success'))
    if  args.dataset_name == "lasot_extension_subset":
        dataset = get_dataset('lasot_extension_subset')
        plot_results(trackers, dataset, 'LaSOT_Ext', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                    skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

if __name__ == '__main__':
    main()
