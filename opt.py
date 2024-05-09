import configargparse
from models.pointnerf import find_model_class_by_name

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP'])

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'own_data', 'scannet'])


    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.2,
                        help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    
    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    


    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')



    parser.add_argument('--N_voxel_init',  # deprecated
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final', # deprecated
                        type=int,
                        default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')

    # ================= new flag ============================
    parser.add_argument("--points_generation_mode", type=str, default="none", help="ways to generate points",
                        choices=["none", "mvsnet", "depth", "lidar"])

    # ================= scannet dataset flag ===================
    parser.add_argument("--scannet_margin", type=int, default=10,
                        help="crop how many pixels around the image from scannet dataset")

    # ========= setting about generating points by mvsnet =========
    # parser.add_argument('--load_points',
    #                     type=int,
    #                     default=0,
    #                     help='normalize the ray_dir to unit length or not, default not')
    # parser.add_argument(
    #     '--ranges',
    #     type=float,
    #     nargs='+',
    #     default=(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0),
    #     help='vscale is the block size that store several voxels'
    # )
    # parser.add_argument(
    #     '--weight_feat_dim',
    #     type=int,
    #     default=8,
    #     help='color channel num')
    # parser.add_argument(
    #     '--shading_feature_mlp_layer0',
    #     type=int,
    #     default=0,
    #     help='interp to agged features mlp num')
    parser.add_argument('--feat_dim', type=int, default=0)
    # parser.add_argument('--depth_grid', type=int, default=128)
    # parser.add_argument('--mvs_point_sampler', type=str, default="gau_single_sampler")
    # parser.add_argument('--vox_res', type=int, default=0, help='vox_resolution if > 0')
    # parser.add_argument('--manual_depth_view', type=int, default=0,
    #                     help="-1 for learning probability, 0 for gt, 1 for pretrained MVSNet")
    # parser.add_argument('--pre_d_est', type=str, default=None, help="loading pretrained depth estimator")
    parser.add_argument('--prob_freq',
                        type=int,
                        default=0,
                        help='saving frequency')
    parser.add_argument('--prob_num_step',
                        type=int,
                        default=100,
                        help='saving frequency')
    parser.add_argument('--checkpoints_dir',
                        type=str,
                        default='./checkpoints',
                        help='models are saved here')
    parser.add_argument('--resume_iter',
                        type=str,
                        default='latest',
                        help='which epoch to resume from')
    parser.add_argument('--resume_dir',
                        type=str,
                        default='',
                        help='dir of the previous checkpoint')
    # todo check below parameter value
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='if specified, print more debugging information')
    parser.add_argument(
        '--bg_filtering',
        type=int,
        default=0,
        help=
        '0 for alpha channel filtering, 1 for background color filtering'
    )
    parser.add_argument(
        '--full_comb',
        type=int,
        default=0,
        help=''
    )
    parser.add_argument('--init_view_num',
                        type=int,
                        default=3,
                        help='number of random samples')
    parser.add_argument('--show_tensorboard',
                        type=int,
                        default=0,
                        help='plot loss curves with tensorboard')
    parser.add_argument('--geo_cnsst_num',
                        default=2,
                        type=int,
                        help='# threads for loading data')
    parser.add_argument('--alpha_range',
                        type=int,
                        default=0,
                        help='saving frequency')
    parser.add_argument('--inall_img',
                        type=int,
                        default=1,
                        help='all points must in the sight of all camera pose')

    # ========= setting about kd tree ========= #
    parser.add_argument('--leaf_size',
                        type=int,
                        default=20, # todo: no need
                        help='max number of points in each leaf node')
    parser.add_argument('--voxel_num',
                        type=int,
                        nargs='+',
                        default=(128, 128, 128),
                        help='cache grid num')
    parser.add_argument('--voxel_num_final',
                        type=int,
                        nargs='+',
                        default=(128, 128, 128),
                        help='voxel num after final upsampling')
    parser.add_argument('--k_query',
                        type=int,
                        default=1, # todo 7, 27
                        help='k nearest neighbor for each cache voxel')
    parser.add_argument('--camera_group_num',
                        type=int,
                        default=10,  # todo
                        help='k nearest neighbor for each cache voxel')
    parser.add_argument('--n_block_per_axis',
                        type=int,
                        default=2,  # todo
                        help='k nearest neighbor for each cache voxel')
    parser.add_argument('--batch_voxel_res',
                        type=int,
                        default=8,  # todo
                        help='k nearest neighbor for each cache voxel')
    parser.add_argument(
        "--tree_code_mode",
        type=str,
        default="xyzxyz",
        help="ways to generate tree node index",
        choices=[
            "xxyyzz",
            "xyzxyz"]) # todo delete
    parser.add_argument('--partition_mode', type=str, default='kdtree',
                        choices=['kdtree', 'voxel'])
    parser.add_argument('--density_mode', type=str, default='abs',
                        choices=['abs', 'square', 'none'])
    parser.add_argument(
        '--padding_tensor_on',
        action='store_true',
        help='if specified, use padding tensor, else padding border')
    parser.add_argument(
        '--sparse_voxel_on',
        action='store_true',
        help='if specified, use sparse voxel else dense voxel')
    parser.add_argument(
        '--relight_flag',
        action='store_true',
        help='if specified, use brdf model in tensoir')
    parser.add_argument(
        '--build_kdtree_without_cache',
        action='store_true',
        help='if specified, build kdtree and kd-cache everytime')
    parser.add_argument(
        '--debug_load_n_dataset',
        type=int,
        default=-1)

    # ============= setting about SScene ==============
    parser.add_argument("--block_num_on_ray", type=int, default=20,
                        help="number of block on each ray")
    parser.add_argument("--use_comp_points", action="store_true", default=False)

    parser.add_argument("--debug_mode_smaller_dataset", action="store_true", default=False)


    # ============= setting about demo mode ==============
    parser.add_argument("--demo_mode", action="store_true", default=False)
    parser.add_argument("--demo_mode_render", action="store_true", default=False)
    parser.add_argument("--demo_scene_ckpts", nargs='+', default=[],
                        help='include itself')
    parser.add_argument("--demo_scene_datadir", nargs='+', default=[],
                        help="dataset dir of other scenes, do not include itself")
    parser.add_argument('--load_train_frames',
                        type=int,
                        nargs='+',
                        default=(-1, -1),
                        help='voxel num after final upsampling')

    model_name = 'mvs_points_volumetric'
    find_model_class_by_name(model_name).modify_commandline_options(parser, is_train=False)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
