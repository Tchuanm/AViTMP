from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.avist_path = '/home/ctang/Github_upload/AViTMP/data/avist'
    settings.davis_dir = ''
    settings.got10k_path = '/home/ctang/Github_upload/AViTMP/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/ctang/Github_upload/AViTMP/data/lasot_extension_subset'
    settings.lasot_path = '/home/ctang/Github_upload/AViTMP/data/lasot'
    settings.network_path = '/home/ctang/Github_upload/AViTMP/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = '/home/ctang/Github_upload/AViTMP/data/nfs'
    settings.otb_path = '/home/ctang/Github_upload/AViTMP/data/otb'
    settings.oxuva_path = '/home/ctang/Github_upload/AViTMP/data/oxuva'
    settings.result_plot_path = '/home/ctang/Github_upload/AViTMP/pytracking/result_plots/'
    settings.results_path = '/home/ctang/Github_upload/AViTMP/pytracking/tracking_results'    # Where to store tracking results
    settings.segmentation_path = '/home/ctang/Github_upload/AViTMP/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/ctang/Github_upload/AViTMP/data/trackingnet'
    settings.uav_path = '/home/ctang/Github_upload/AViTMP/data/uav'
    settings.vot_path = '/home/ctang/Github_upload/AViTMP/data/vot2020'
    settings.youtubevos_dir = ''

    return settings

