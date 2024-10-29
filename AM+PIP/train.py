import os
import time
from tqdm import tqdm
import torch
import math
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to, AverageMeter


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def validate(model, dataset, opts, epoch=None, tb_logger=None):
    # Validate
    print("=================================================================================")
    print('>> Validating on {}...'.format(opts.val_dataset))
    cost = rollout(model, dataset, opts, epoch, tb_logger, return_penalty=True)
    if len(cost.shape) > 1:
        cost, timeout, timeout_nodes = cost[:, 0], cost[:, 1], cost[:, 2]
        if opts.val_solution_path:
            with open(opts.val_solution_path, 'rb') as f:
                opt_sol = pickle.load(f)
            grid_factor = 100. if opts.problem == "tsptw" else 1.
            opt_sol = torch.tensor([i[0] / grid_factor for i in opt_sol]).to(cost.device)
            gap = ((cost[timeout_nodes==0] - opt_sol[timeout_nodes==0]) / opt_sol[timeout_nodes==0] * 100).mean() if (timeout_nodes==0).any() else 1000
        else:
            gap = 1000
        cost = cost[timeout_nodes==0] if (timeout_nodes==0).any() else torch.zeros(1) # hardcoded
        infsb_rate = ((timeout_nodes!=0).sum() / len(timeout_nodes)).item()*100
        avg_cost = cost.mean()
        print("Feasible cost: {:.4f} +- {:.4f} [Gap: {:.4f}%], Infeasible rate: {:.4f}%, timeout: {:.4f}, timeout_nodes: {:.4f}!".format(avg_cost, torch.std(cost) / math.sqrt(len(cost)), gap, infsb_rate, timeout.mean(), timeout_nodes.mean()))
        avg_cost = [avg_cost, infsb_rate, timeout.mean(), timeout_nodes.mean()]
    else:
        avg_cost = cost.mean()
        print('Validation overall avg_cost: {:.4f} +- {:.4f}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    print("=================================================================================")

    return avg_cost

def rollout(model, dataset, opts, epoch=None, tb_logger=None, return_penalty = False):
    pip_model = None
    if isinstance(model, list):
        model, pip_model = model
    if opts.is_train_pip_decoder:
        print('Using the current model as sl model')
        pip_model = None
    if pip_model is not None:
        print('Using the pip model model as sl model')
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, prob = model(move_to(bat, opts.device), input_pip_model=pip_model)
            if opts.problem in ["tsptw", "tspdl"]:
                cost, timeout, timeout_nodes = cost
                # return cost.data.cpu()
                if isinstance(prob, list):
                    _, prob_sl, PI_mask, visited_mask = prob
                    if opts.sl_loss == "BCEWithLogitsLoss":
                        prob_sl = torch.sigmoid(prob_sl)
                    prob_sl = torch.where(prob_sl > opts.decision_boundary, 1., prob_sl)
                    prob_sl = torch.where(prob_sl <= (1 - opts.decision_boundary), 0., prob_sl)
                    label = PI_mask.int()
                    if opts.dislocation_start != -1:
                        visited_mask = visited_mask[:, opts.dislocation_start:, :]
                        label = label[:, opts.dislocation_start:, :]
                        prob_sl = prob_sl[:, : (-opts.dislocation_start), :]
                    prob_sl = prob_sl[~visited_mask]
                    label = label[~visited_mask]
                    tn, fp, fn, tp = confusion_matrix((label.cpu().numpy()).astype(np.int32), (prob_sl.cpu().numpy()).astype(np.int32)).ravel()
                    if return_penalty:
                        return [torch.cat([cost[:, None], timeout[:, None], timeout_nodes[:, None]], dim=-1), tp, tn, (fn + tp), (tn + fp)]

                if return_penalty:
                    # return cost, timeout, timeout_nodes
                    return torch.cat([cost[:, None], timeout[:, None], timeout_nodes[:, None]], dim=-1)
                cost = cost + timeout + timeout_nodes
        if isinstance(prob, list):
            return [cost.data.cpu(), tp, tn, (fn + tp), (tn + fp)]
        else:
            return cost.data.cpu()

    costs = torch.tensor([]).cuda()
    sl_flag = False
    tps, tns, infsb_nums, fsb_nums = 0, 0, 0, 0
    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
        cost = eval_model_bat(bat)
        if isinstance(cost, list):
            sl_flag = True
            cost, tp, tn, infsb_num, fsb_num = cost
            tps += tp
            tns += tn
            infsb_nums += infsb_num
            fsb_nums += fsb_num

        costs = torch.cat([costs, cost.cuda()], 0)

    if sl_flag:
        accuracy = (tps + tns) / (infsb_nums + fsb_nums) * 100
        infsb_accuracy = tps / infsb_nums * 100
        fsb_accuracy = tns / fsb_nums * 100
        print("Validation, Auc: {:.4f}%, Infeasible Auc: {:.4f}% ({}), Feasible Auc: {:.4f}% ({}".format(accuracy, infsb_accuracy, infsb_nums, fsb_accuracy, fsb_nums))
        if epoch is not None and not opts.no_tensorboard:
            tb_logger.log_value('val/accuracy', accuracy, epoch)
            tb_logger.log_value('val/infsb_accuracy', infsb_accuracy, epoch)
            tb_logger.log_value('val/fsb_accuracy', fsb_accuracy, epoch)
    return costs

def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    pip_model = None
    if isinstance(model, list):
        model, pip_model = model
    if opts.pip_decoder:
        opts.accuracy, opts.infsb_accuracy, opts.fsb_accuracy, opts.sl_loss_list = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, hardness=opts.hardness))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        if pip_model is not None and not opts.is_train_pip_decoder and not isinstance(model, list):
            model = [model, pip_model]
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print(">> Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if isinstance(model, list):
        model, pip_model = model
    if opts.pip_decoder and opts.is_train_pip_decoder:
        accuracy, infsb_accuracy, fsb_accuracy, sl_loss =opts.accuracy.avg, opts.infsb_accuracy.avg, opts.fsb_accuracy.avg, opts.sl_loss_list.avg
        print('>> Finished epoch: {} SL loss: {} Auc: {:.4f}, Infeasible Auc: {:.4f} ({}), Feasible Auc: {:.4f} ({})'.format(epoch, sl_loss, accuracy, infsb_accuracy, opts.infsb_accuracy.count, fsb_accuracy, opts.fsb_accuracy.count))
        if opts.no_tensorboard:
            tb_logger.log_value('sl_epoch/accuracy', accuracy, epoch)
            tb_logger.log_value('sl_epoch/infsb_accuracy', infsb_accuracy, epoch)
            tb_logger.log_value('sl_epoch/fsb_accuracy', fsb_accuracy, epoch)

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('>> Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

        if opts.pip_decoder and opts.is_train_pip_decoder and opts.pip_save == "epoch":

            opts.accuracy_isbsf = True if accuracy > opts.accuracy_bsf else False
            opts.fsb_accuracy_isbsf = True if fsb_accuracy > opts.fsb_accuracy_bsf else False
            opts.infsb_accuracy_isbsf = True if infsb_accuracy > opts.infsb_accuracy_bsf else False

            opts.accuracy_bsf = accuracy if accuracy > opts.accuracy_bsf else opts.accuracy_bsf
            opts.fsb_accuracy_bsf = fsb_accuracy if fsb_accuracy > opts.fsb_accuracy_bsf else opts.fsb_accuracy_bsf
            opts.infsb_accuracy_bsf = infsb_accuracy if infsb_accuracy > opts.infsb_accuracy_bsf else opts.infsb_accuracy_bsf

            if opts.accuracy_isbsf:
                if not os.path.exists('{}/accuracy_bsf.pt'.format(opts.save_dir)) or (infsb_accuracy > 75. and fsb_accuracy > 75.):
                    print("Saving BSF accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(opts.accuracy_bsf, accuracy, infsb_accuracy , fsb_accuracy))
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': get_inner_model(model).state_dict(),
                        'accuracy': accuracy,
                        'fsb_accuracy': fsb_accuracy,
                        'infsb_accuracy': infsb_accuracy,
                    }
                    torch.save(checkpoint_dict, '{}/accuracy_bsf.pt'.format(opts.save_dir))
            if opts.fsb_accuracy_isbsf:
                if not os.path.exists('{}/fsb_accuracy_bsf.pt'.format(opts.save_dir)) or (infsb_accuracy > 75.) or (infsb_accuracy > 60. and opts.problem == "TSPDL"):
                    print("Saving BSF Feasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format( opts.fsb_accuracy_bsf , accuracy, infsb_accuracy, fsb_accuracy))
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': get_inner_model(model).state_dict(),
                        'accuracy': accuracy,
                        'fsb_accuracy': fsb_accuracy,
                        'infsb_accuracy': infsb_accuracy,
                    }
                    torch.save(checkpoint_dict, '{}/fsb_accuracy_bsf.pt'.format(opts.save_dir))
            if opts.infsb_accuracy_isbsf:
                if not os.path.exists('{}/infsb_accuracy_bsf.pt'.format(opts.save_dir)) or (fsb_accuracy > 75.) or (fsb_accuracy > 60. and opts.problem == "TSPDL"):
                    print("Saving BSF Infeasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(opts.infsb_accuracy_bsf, accuracy, infsb_accuracy, fsb_accuracy))
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': get_inner_model(model).state_dict(),
                        'accuracy': accuracy,
                        'fsb_accuracy': fsb_accuracy,
                        'infsb_accuracy': infsb_accuracy,
                    }
                    torch.save(checkpoint_dict, '{}/infsb_accuracy_bsf.pt'.format(opts.save_dir))

    if opts.pip_decoder and not opts.is_train_pip_decoder:
        print("Validating with model with PI mask...")
        model.generate_PI_mask = True
        model = [model, pip_model]
    avg_reward = validate(model, val_dataset, opts, epoch, tb_logger)
    if isinstance(avg_reward, list):
        avg_reward, infsb_rate, avg_timeout, avg_timeout_nodes = avg_reward
        if not opts.no_tensorboard:
            tb_logger.log_value('validation/val_infsb_rate', infsb_rate, epoch)
            tb_logger.log_value('validation/val_avg_timeout', avg_timeout, epoch)
            tb_logger.log_value('validation/val_avg_timeout_nodes', avg_timeout_nodes, epoch)

    if not opts.no_tensorboard:
        tb_logger.log_value('validation/val_avg_reward', avg_reward, epoch)

    baseline.epoch_callback(model, epoch) # FIXME: NOT UPLOAD pip MODEL YET

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    if isinstance(model, list):
        model, pip_model = model
    else:
        pip_model = None
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x, input_pip_model=pip_model)
    if isinstance(cost, list): # got penalty
        cost, timeout, timeout_nodes = cost
        cost = cost + timeout + timeout_nodes
    if isinstance(log_likelihood, list):
        log_likelihood, prob_sl, PI_mask, visited_mask = log_likelihood
        label = PI_mask.int()
        if opts.dislocation_start != -1:
            visited_mask = visited_mask[:, opts.dislocation_start:, :]
            label = label[:, opts.dislocation_start:, :]
            prob_sl = prob_sl[:, : (-opts.dislocation_start), :]
        probs_sl = prob_sl[~visited_mask]
        label = label[~visited_mask]

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    sl_loss = None
    if opts.is_train_pip_decoder and opts.pip_decoder and label.sum() != 0 and label.sum() != label.size():
        if opts.label_balance_sampling:
            if opts.fast_label_balance:
                # new version: accelerate the calculation of smaple weights
                assert opts.sl_loss == "BCEWithLogitsLoss", "only BCEWithLogitsLoss (output with no sigmoid) is supported when label_balance_sampling==True with speedup!"
                infsb_sample_number = torch.nonzero(label != 0).size(0)  # positive
                fsb_sample_number = torch.nonzero(label == 0).size(0)  # negative
                pos_weight = fsb_sample_number / infsb_sample_number  # neg / pos
                pos_weight = torch.ones_like(label) * pos_weight
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                sl_loss = criterion(probs_sl, label.float())
                if opts.fast_weight:
                    sl_weight = (fsb_sample_number + infsb_sample_number) / (2 * fsb_sample_number)
                    # with this weight, fast method totally equals to the non-fast one
                    sl_loss = sl_loss * sl_weight
            else:
                # assert self.trainer_params["sl_loss"] == "BCELoss", "only BCELoss is supported when label_balance_sampling==True without speedup!"
                edge_labels = (label != 0).int().cpu().numpy().flatten()
                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
                if opts.sl_loss == "BCELoss":
                    probs_sl = torch.clamp(probs_sl, min=1e-7, max=1 - 1e-7)  # add a clamp to avoid numerical instability
                    sl_loss = - edge_cw[1] * (label * torch.log(probs_sl)) - edge_cw[0] * (((1 - label) * torch.log(1 - probs_sl)))
                elif opts.sl_loss == "BCEWithLogitsLoss":
                    sl_loss = - edge_cw[1] * (label * torch.log(F.sigmoid(probs_sl))) - edge_cw[0] * (((1 - label) * torch.log(1 - F.sigmoid(probs_sl))))
                sl_loss = sl_loss.mean()
        else:
            if opts.sl_loss == "BCEWithLogitsLoss":
                criterion = torch.nn.BCEWithLogitsLoss()
            elif opts.sl_loss == "BCELoss":
                criterion = torch.nn.BCELoss()
            sl_loss = criterion(probs_sl, label)

        loss = loss + sl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    if sl_loss is not None:
        if opts.sl_loss == "BCEWithLogitsLoss":
            probs_sl = torch.sigmoid(probs_sl.detach())
        probs_sl = torch.where(probs_sl > opts.decision_boundary, 1., probs_sl)
        probs_sl = torch.where(probs_sl <= (1 - opts.decision_boundary), 0., probs_sl)
        tn, fp, fn, tp = confusion_matrix((label.cpu().numpy()).astype(np.int32),
                                          (probs_sl.cpu().numpy()).astype(np.int32)).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        infsb_accuracy = tp / (fn + tp) * 100
        fsb_accuracy = tn / (tn + fp) * 100
        opts.accuracy.update(accuracy, (tp + tn + fp + fn))
        opts.infsb_accuracy.update(infsb_accuracy, (fn + tp))
        opts.fsb_accuracy.update(fsb_accuracy, (tn + fp))
        opts.sl_loss_list.update(sl_loss)
        sl_loss = [sl_loss, accuracy, infsb_accuracy, (fn + tp), fsb_accuracy, (tn + fp)]
        if opts.is_train_pip_decoder and opts.pip_save == "batch":
            opts.accuracy_isbsf = True if accuracy > opts.accuracy_bsf else False
            opts.fsb_accuracy_isbsf = True if fsb_accuracy > opts.fsb_accuracy_bsf else False
            opts.infsb_accuracy_isbsf = True if infsb_accuracy > opts.infsb_accuracy_bsf else False

            opts.accuracy_bsf = accuracy if accuracy > opts.accuracy_bsf else opts.accuracy_bsf
            opts.fsb_accuracy_bsf = fsb_accuracy if fsb_accuracy > opts.fsb_accuracy_bsf else opts.fsb_accuracy_bsf
            opts.infsb_accuracy_bsf = infsb_accuracy if infsb_accuracy > opts.infsb_accuracy_bsf else opts.infsb_accuracy_bsf

            if opts.accuracy_isbsf:
                if not os.path.exists('{}/accuracy_bsf.pt'.format(opts.save_dir)) or (infsb_accuracy > 75. and fsb_accuracy > 75.):
                    print("Saving BSF accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(opts.accuracy_bsf, accuracy, infsb_accuracy , fsb_accuracy))
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': get_inner_model(model).state_dict(),
                        'accuracy': accuracy,
                        'fsb_accuracy': fsb_accuracy,
                        'infsb_accuracy': infsb_accuracy,
                    }
                    torch.save(checkpoint_dict, '{}/accuracy_bsf.pt'.format(opts.save_dir))
            if opts.fsb_accuracy_isbsf:
                if not os.path.exists('{}/fsb_accuracy_bsf.pt'.format(opts.save_dir)) or (infsb_accuracy > 75.) or (infsb_accuracy > 60. and opts.problem == "TSPDL"):
                    print("Saving BSF Feasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format( opts.fsb_accuracy_bsf , accuracy, infsb_accuracy, fsb_accuracy))
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': get_inner_model(model).state_dict(),
                        'accuracy': accuracy,
                        'fsb_accuracy': fsb_accuracy,
                        'infsb_accuracy': infsb_accuracy,
                    }
                    torch.save(checkpoint_dict, '{}/fsb_accuracy_bsf.pt'.format(opts.save_dir))
            if opts.infsb_accuracy_isbsf:
                if not os.path.exists('{}/infsb_accuracy_bsf.pt'.format(opts.save_dir)) or (fsb_accuracy > 75.) or (fsb_accuracy > 60. and opts.problem == "TSPDL"):
                    print("Saving BSF Infeasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(opts.infsb_accuracy_bsf, accuracy, infsb_accuracy, fsb_accuracy))
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': get_inner_model(model).state_dict(),
                        'accuracy': accuracy,
                        'fsb_accuracy': fsb_accuracy,
                        'infsb_accuracy': infsb_accuracy,
                    }
                    torch.save(checkpoint_dict, '{}/infsb_accuracy_bsf.pt'.format(opts.save_dir))

    # Logging
    if step % int(opts.log_step) == 0:
        if opts.problem in ["tsptw", "tspdl"]:
            cost = [cost, timeout, timeout_nodes]
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, sl_loss, tb_logger, opts)






