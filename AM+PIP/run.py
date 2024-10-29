#!/usr/bin/env python3

import os
import json

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem, seed_everything, copy_all_src



def run(opts):

    # Set the random seed
    seed_everything(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        generate_PI_mask = opts.generate_PI_mask,
        pip_decoder = opts.pip_decoder,
        sigmoid = False if opts.sl_loss == "BCEWithLogitsLoss" else True,
        n_encode_layers=opts.n_encode_layers,
        decision_boundary=opts.decision_boundary,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, hardness=opts.hardness)

    # Initialize pip model
    if opts.pip_decoder:
        pip_model = model_class(
            opts.embedding_dim,
            opts.hidden_dim,
            problem,
            generate_PI_mask=False,
            pip_decoder=opts.pip_decoder,
            sigmoid=False if opts.sl_loss == "BCEWithLogitsLoss" else True,
            decision_boundary=opts.decision_boundary,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size
        ).to(opts.device)
        if opts.use_cuda and torch.cuda.device_count() > 1:
            pip_model = torch.nn.DataParallel(pip_model)
        if opts.pip_checkpoint is not None:
            pip_model_param = torch_load_cpu(opts.pip_checkpoint)
            pip_model_ = get_inner_model(pip_model)
            pip_model_.load_state_dict({**pip_model_.state_dict(), **pip_model_param.get('model', {})})
            print('  [*] Loading SL pip model from {} [Accuracy: {:.4f}%; Infeasible: {:.4f}%; Feasible: {:.4f}%]'.format(opts.pip_checkpoint, pip_model_param['accuracy'] * 100, pip_model_param['infsb_accuracy'] * 100,pip_model_param['fsb_accuracy'] * 100))
            if "fsb_accuracy_bsf.pt" in opts.pip_checkpoint:
                opts.fsb_accuracy_bsf = pip_model_param['fsb_accuracy']
            elif "infsb_accuracy_bsf.pt" in opts.pip_checkpoint:
                opts.infsb_accuracy_bsf = pip_model_param['infsb_accuracy']
            else:
                opts.accuracy_bsf = pip_model_param['accuracy']

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        if opts.pip_checkpoint is not None:
            model = [model, pip_model]
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    copy_all_src(opts.save_dir)

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            print("====================================================================================================================")
            print(">> Start train PIP: epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))

            if opts.pip_decoder:
                if isinstance(model, list):
                    model, pip_model = model
                if epoch in opts.train_sl_epoch_list:
                    model.generate_PI_mask = True
                    opts.is_train_pip_decoder = True
                    print('>> PIP Decoder is training and PI mask {} generated...'.format('is' if model.generate_PI_mask else 'is not'))
                else:
                    model.generate_PI_mask = False
                    opts.is_train_pip_decoder = False
                    print('>> PIP Decoder is not training and PI mask {} generated...'.format('is' if model.generate_PI_mask else 'is not'))
                    if epoch in opts.load_sl_epoch_list  and epoch != opts.epoch_start:
                        pip_checkpoint = {"last_epoch": "epoch-{}.pt".format(epoch - 1),
                                           "train_fsb_bsf": "fsb_accuracy_bsf.pt",
                                           "train_infsb_bsf": "infsb_accuracy_bsf.pt",
                                           "train_accuracy_bsf": "accuracy_bsf.pt"}

                        checkpoint_fullname = os.path.join(opts.save_dir, pip_checkpoint[opts.load_which_pip])
                        pip_model_param = torch_load_cpu(checkpoint_fullname)
                        pip_model.load_state_dict(pip_model_param['model_state_dict'], strict=True)

                        print('  [*] Loading PIP model from {} [Accuracy: {:.4f}%; Infeasible: {:.4f}%; Feasible: {:.4f}%]'.format(
                                checkpoint_fullname, pip_model_param['accuracy'], pip_model_param['infsb_accuracy'], pip_model_param['fsb_accuracy']))
                        # print("model: ", model.state_dict()["init_embed.weight"][0])
                        # print("sl: ", pip_model.state_dict()["init_embed.weight"][0])
                        if "fsb_accuracy_bsf.pt" in checkpoint_fullname:
                            opts.fsb_accuracy_bsf = pip_model_param['fsb_accuracy']
                        elif "infsb_accuracy_bsf.pt" in checkpoint_fullname:
                            opts.infsb_accuracy_bsf = pip_model_param['infsb_accuracy']
                        else:
                            opts.accuracy_bsf = pip_model_param['accuracy']
                    pip_model.eval()
                    model = [model, pip_model]

            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    run(get_options())
