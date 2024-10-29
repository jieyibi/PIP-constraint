import os
import time
import argparse
import torch
import pprint as pp

def get_options(args=None):
    parser = argparse.ArgumentParser(description='PIP framework on Attention model for TSPTW')

    # Data
    parser.add_argument('--problem', default='tsptw', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=50, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000, help='Number of instances used for reporting validation performance')
    parser.add_argument('--generate_PI_mask', action='store_true', help='whether to generate the PI masking')
    parser.add_argument('--hardness', type=str, default="hard", choices=["hard", "medium", "easy"], help="Different levels of constraint hardness")
    parser.add_argument('--val_dataset', type=str, default="../data/TSPTW/tsptw50_hard.pkl", help='Dataset file to use for validation')
    parser.add_argument('--val_solution_path', type=str, default='../data/TSPTW/lkh_tsptw50_hard.pkl"')

    # PIP
    parser.add_argument('--pip_decoder', action='store_true', help="activate PIP-D")
    parser.add_argument('--simulation_stop_epoch', type=int, default=10)
    parser.add_argument('--pip_update_interval', type=int, default=10)
    parser.add_argument('--pip_update_epoch', type=int, default=2)
    parser.add_argument('--pip_last_growup', type=int, default=5)
    parser.add_argument('--pip_save', type=str, default="epoch")
    parser.add_argument('--load_which_pip', type=str, default="train_fsb_bsf", choices=["last_epoch", "train_fsb_bsf", "train_infsb_bsf", "train_accuracy_bsf"])
    parser.add_argument('--pip_checkpoint', type=str, default=None)
    parser.add_argument('--sl_loss', type=str, default="BCEWithLogitsLoss", choices=["BCEWithLogitsLoss", "BCELoss", "FL", "CE"], help="FL: focal loss; CE: cross entropy loss")
    parser.add_argument('--label_balance_sampling', type=bool, default=True)
    parser.add_argument('--fast_label_balance', type=bool, default=True)
    parser.add_argument('--fast_weight', type=bool, default=True)
    parser.add_argument('--decision_boundary', type=float, default=0.5)
    parser.add_argument('--dislocation_start', type=int, default=-1, help="-1 means deactivating")

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10., help='Clip the parameters to within +- this value using tanh. Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8, help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default='rollout', help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05, help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=10000, help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true', help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished to save memory (default None means no shrinking)')
    # parser.add_argument('--data_distribution', type=str, default=None,
    #                     help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run_tsptw', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', default=False, help='Disable progress bar')
    parser.add_argument('--all_cuda_visible',action='store_true', help='Whether to use all available cuda')
    parser.add_argument('--CUDA_VISIBLE_ID', default="0",
                        help='Make specific id of cuda visible and use them instead of all available cuda')


    opts = parser.parse_args(args)

    opts.val_dataset = f"../data/TSPTW/tsptw{opts.graph_size}_{opts.hardness}.pkl"
    opts.val_solution_path = f"../data/TSPTW/lkh_tsptw{opts.graph_size}_{opts.hardness}.pkl"

    # Pretty print the run args
    pp.pprint(vars(opts))

    if not opts.all_cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_ID
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')

    opts.run_name = f"{opts.run_name}{opts.graph_size}_{opts.hardness}_{opts.normalization}Norm"
    if opts.generate_PI_mask:
        opts.run_name = opts.run_name + "_generatePIMask"
    if opts.pip_decoder:
        opts.run_name = opts.run_name + "_PIPDecoder"
        opts.train_sl_epoch_list = list(range(opts.simulation_stop_epoch))
        for start in range(opts.pip_update_interval, opts.n_epochs+1, opts.pip_update_interval):
            opts.train_sl_epoch_list.extend(range(start - opts.pip_update_epoch , start))

        if opts.pip_last_growup > opts.pip_update_epoch:
            opts.train_sl_epoch_list.extend(range(opts.n_epochs - opts.pip_last_growup, opts.n_epochs))

        opts.load_sl_epoch_list = [opts.simulation_stop_epoch] + list(range(0, opts.n_epochs - opts.pip_last_growup, opts.pip_update_interval))[1:]

        opts.accuracy_bsf, opts.infsb_accuracy_bsf, opts.fsb_accuracy_bsf = 0., 0., 0.
        opts.is_train_pip_decoder = True
    else:
        opts.is_train_pip_decoder = False

    if opts.pip_decoder and opts.label_balance_sampling:
        opts.run_name = opts.run_name + "_labelBalance"
    if opts.dislocation_start != -1:
        opts.run_name = opts.run_name + "_dislocation{}".format(opts.dislocation_start)
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"

    from utils import create_logger
    create_logger(filename="run_log", log_path=opts.save_dir)
    return opts
