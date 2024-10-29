def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, sl_loss, tb_logger, opts):
    penalty = False
    if isinstance(cost, list):
        penalty = True
        cost, timeout, timeout_nodes = cost
        fsb_cost = cost[timeout==0].mean().item() if (timeout==0).any() else 0.
        avg_timeout = timeout.mean().item()
        avg_timeout_nodes = timeout_nodes.float().mean().item()
        ins_infeasible_rate = (timeout != 0).sum() / len(timeout) *100

    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    if penalty:
        print("")
        print('epoch: {}, train_batch_id: {}, avg_cost: {:.4f} [Fsb_cost: {:.4f}, Infsb_rate: {:.4f}%], avg_timeout: {:.4f}, avg_timeout_nodes: {:.4f}'.format(epoch, batch_id, avg_cost, fsb_cost, ins_infeasible_rate, avg_timeout, avg_timeout_nodes))
    else:
        print('epoch: {}, train_batch_id: {}, avg_cost: {:.4f}'.format(epoch, batch_id, avg_cost))

    if isinstance(sl_loss, list):
        sl_loss, accuracy, infsb_accuracy, infsb_num, fsb_accuracy, fsb_num = sl_loss
        print('epoch: {} SL loss: {} Auc: {:.4f}, Infeasible Auc: {:.4f} ({}), Feasible Auc: {:.4f} ({})'.format(epoch, sl_loss, accuracy, infsb_accuracy, infsb_num, fsb_accuracy, fsb_num))
        tb_logger.log_value('sl_batch/sl_loss', sl_loss, step)
        tb_logger.log_value('sl_batch/accuracy', accuracy, step)
        tb_logger.log_value('sl_batch/infsb_accuracy', infsb_accuracy, step)
        tb_logger.log_value('sl_batch/infsb_num', infsb_num, step)
        tb_logger.log_value('sl_batch/fsb_accuracy', fsb_accuracy, step)
        tb_logger.log_value('sl_batch/fsb_num', fsb_num, step)

    print('grad_norm: {:.4f}, clipped: {:.4f}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        if penalty:
            tb_logger.log_value('avg_fsb_cost', fsb_cost, step)
            tb_logger.log_value('avg_timeout', avg_timeout, step)
            tb_logger.log_value('avg_timeout_nodes', avg_timeout_nodes, step)
            tb_logger.log_value('ins_infeasible_rate', ins_infeasible_rate, step)
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
