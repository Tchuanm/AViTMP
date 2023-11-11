import torch
import os
import sys
from pathlib import Path
import importlib
import inspect
import argparse

def torch_load_legacy(path):
    """Load network with legacy environment."""

    # Setup legacy env (for older networks)
    _setup_legacy_env()

    # Load network
    checkpoint_dict = torch.load(path, map_location='cpu')

    # Cleanup legacy
    _cleanup_legacy_env()

    return checkpoint_dict


def _setup_legacy_env():
    importlib.import_module('ltr')
    sys.modules['dlframework'] = sys.modules['ltr']
    sys.modules['dlframework.common'] = sys.modules['ltr']
    importlib.import_module('ltr.admin')
    sys.modules['dlframework.common.utils'] = sys.modules['ltr.admin']
    for m in ('model_constructor', 'stats', 'settings', 'local'):
        importlib.import_module('ltr.admin.' + m)
        sys.modules['dlframework.common.utils.' + m] = sys.modules['ltr.admin.' + m]


def _cleanup_legacy_env():
    del_modules = []
    for m in sys.modules.keys():
        if m.startswith('dlframework'):
            del_modules.append(m)
    for m in del_modules:
        del sys.modules[m]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--epoch', type=int, default=300, help='Name of tracking method.')
    parser.add_argument('--tracker', type=str, default='avitmp', help='Name of tracking method.')

    args = parser.parse_args()
    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(current_path)
    if not os.path.exists(current_path +"/pytracking/networks"):
        os.makedirs(current_path +"/pytracking/networks")
    if os.path.exists(current_path +"pytracking/networks/{}.pth.tar".format(args.tracker)) == False:
        os.system("cp {}/ltr/checkpoints/ltr/avitmp/{}/AViTMPnet_ep{:04d}.pth.tar   {}/pytracking/networks/{}.pth.tar".format( current_path, args.tracker, args.epoch,  current_path, args.tracker))

    pth_path= current_path + '/pytracking/networks/{}.pth.tar'.format(args.tracker)
    print('pth:', pth_path)
    if not os.path.exists(pth_path):
        pass

    pth_dict=torch_load_legacy(pth_path) ## load pth


    keys_old_pth=[] ##
    for k in pth_dict:
        print("before_cut:%s"%(str(k)))
        keys_old_pth.append(k)

    # keys_we_need = ['model'] #
    keys_we_need = ['net','constructor','net_type'] #
    for key in keys_old_pth:
        if key not in keys_we_need:
            del pth_dict[key] ##

    for k in pth_dict:
        print("after_cut:%s"%(str(k)))

    torch.save(pth_dict, pth_path)

# cut_before: 1.1G, cut_after: 544M
