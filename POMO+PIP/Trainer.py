from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from tensorboard_logger import Logger as TbLogger
from utils import *
from models.SINGLEModel import SINGLEModel
from sklearn.utils.class_weight import compute_class_weight
import os, wandb
from sklearn.metrics import confusion_matrix

class Trainer:
    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.problem = self.args.problem
        self.penalty_factor = args.penalty_factor

        self.device = args.device
        self.log_path = args.log_path
        self.result_log = {"val_score": [], "val_gap": [], "val_infsb_rate": []}
        if args.tb_logger:
            self.tb_logger = TbLogger(self.log_path)
        else:
            self.tb_logger = None
        self.wandb_logger = args.wandb_logger

        # Main Components
        self.envs = get_env(self.args.problem)  # a list of envs classes (different problems), remember to initialize it!
        self.model = SINGLEModel(**self.model_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        num_param(self.model)


        if self.model_params["pip_decoder"]:
            self.is_train_pip_decoder = True if self.trainer_params["simulation_stop_epoch"] > 0 else False
            self.accuracy_bsf, self.fsb_accuracy_bsf, self.infsb_accuracy_bsf =  0., 0., 0.
            self.accuracy_isbsf, self.fsb_accuracy_isbsf, self.infsb_accuracy_isbsf = False, False, False

            self.train_sl_epoch_list = list(range(1, self.trainer_params["simulation_stop_epoch"] + 1))

            for start in range(self.trainer_params["pip_update_interval"], self.trainer_params["epochs"] + 1, self.trainer_params["pip_update_interval"]):
                self.train_sl_epoch_list.extend(range(start - self.trainer_params["pip_update_epoch"] + 1, start + 1))

            if self.trainer_params["pip_last_growup"] > self.trainer_params["pip_update_epoch"]:
                self.train_sl_epoch_list.extend(range(self.trainer_params["epochs"] - self.trainer_params["pip_last_growup"] + 1, self.trainer_params["epochs"]+1))

            self.load_sl_epoch_list = [self.trainer_params["simulation_stop_epoch"] + 1] + list(range(1, self.trainer_params["epochs"] - self.trainer_params["pip_last_growup"] + 1, self.trainer_params["pip_update_interval"]))[1:]

            # print(self.train_sl_epoch_list)
            # print(self.load_sl_epoch_list)

            # PIP decoder does not update frequently,
            # Hence we record the latest updated one and use it to predict PI mask until the next update
            if self.trainer_params["lazy_pip_model"]:
                self.lazy_model = SINGLEModel(**self.model_params)
            else:
                self.lazy_model = None

            if args.pip_checkpoint:
                checkpoint_fullname = args.pip_checkpoint
                checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
                try:
                    self.lazy_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                except:
                    self.lazy_model.load_state_dict(checkpoint, strict=True)
                try:
                    print(
                        ">> Load lazy PIP-D model from {} [Accuracy: {:.4f}%; Infeasible: {:.4f}%; Feasible: {:.4f}%]".format(
                            checkpoint_fullname, checkpoint['accuracy'] * 100, checkpoint['infsb_accuracy'] * 100,
                            checkpoint['fsb_accuracy'] * 100))
                    if "fsb_accuracy_bsf.pt" in checkpoint_fullname:
                        self.fsb_accuracy_bsf = checkpoint['fsb_accuracy']
                    elif "infsb_accuracy_bsf.pt" in checkpoint_fullname:
                        self.infsb_accuracy_bsf = checkpoint['infsb_accuracy']
                    else:
                        self.accuracy_bsf = checkpoint['accuracy']
                except:
                    print(">> Load lazy PIP-D model from {}".format(checkpoint_fullname))

        # Restore
        self.start_epoch = 1
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except:
                self.model.load_state_dict(checkpoint, strict=True)

            self.start_epoch = 1 + checkpoint['epoch']
            self.scheduler.last_epoch = checkpoint['epoch'] - 1
            if self.trainer_params["load_optimizer"]:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(">> Optimizer (Epoch: {}) Loaded (lr = {})!".format(checkpoint['epoch'], self.optimizer.param_groups[0]['lr']))
            print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))
            print(">> Load from {}".format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            print('================================================================================')

            # Load the latest updated PIP model for PI masking prediction
            if self.model_params["pip_decoder"]:
                if epoch not in self.train_sl_epoch_list:
                    print('>> PIP decoder is not training...')
                    self.is_train_pip_decoder = False
                    self.model_params["generate_PI_mask"] = False
                    if self.trainer_params["lazy_pip_model"] and (epoch in self.load_sl_epoch_list) and epoch != self.start_epoch: # if epoch == start_epoch, ckpt already loaded or it is training (no need to load)
                        pip_checkpoint = {"last_epoch": "epoch-{}.pt".format(epoch - 1),
                                           "train_fsb_bsf": "fsb_accuracy_bsf.pt",
                                           "train_infsb_bsf": "infsb_accuracy_bsf.pt",
                                           "train_accuracy_bsf": "accuracy_bsf.pt"}
                        checkpoint_fullname = os.path.join(self.log_path, pip_checkpoint[self.trainer_params["load_which_pip"]])
                        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
                        try:
                            self.lazy_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                        except:
                            self.lazy_model.load_state_dict(checkpoint, strict=True)
                        try:
                            print(">> Load lazy PIP-D model from {} [Accuracy: {:.4f}%; Infeasible: {:.4f}%; Feasible: {:.4f}%]".format(checkpoint_fullname, checkpoint['accuracy']*100, checkpoint['infsb_accuracy']*100, checkpoint['fsb_accuracy']*100))
                        except:
                            print(">> Load lazy PIP-D model from {}".format(checkpoint_fullname))
                else:
                    print('>> PIP decoder is training...')
                    self.is_train_pip_decoder = True
                    self.model_params["generate_PI_mask"] = True

            # Update penalty factor if you want to train it in a curriculum learning way
            if self.trainer_params["penalty_increase"]:
                self.penalty_factor = 0.5 + epoch / self.trainer_params["epochs"] * 1.5

            # Train
            train_score, train_loss, infeasible = self._train_one_epoch(epoch)

            # Step
            self.scheduler.step()

            # Log
            if isinstance(train_score, list):
                dist_reward, total_timeout_reward, timeout_nodes_reward = train_score
                train_score = dist_reward
            if self.trainer_params["fsb_dist_only"]:
                try:
                    sol_infeasible_rate, ins_infeasible_rate, feasible_dist_mean, feasible_dist_max_pomo_mean = infeasible
                except:
                    pass
            else:
                sol_infeasible_rate, ins_infeasible_rate = infeasible
            if self.tb_logger:
                self.tb_logger.log_value('train/train_score', train_score, epoch)
                self.tb_logger.log_value('train/train_loss', train_loss, epoch)
                try:
                    self.tb_logger.log_value('feasibility/solution_infeasible_rate', sol_infeasible_rate, epoch)
                    self.tb_logger.log_value('feasibility/instance_infeasible_rate', ins_infeasible_rate, epoch)
                except:
                    pass
                if self.trainer_params["timeout_reward"]:
                    self.tb_logger.log_value("feasibility/total_timeout", total_timeout_reward, epoch)
                    self.tb_logger.log_value("feasibility/timeout_nodes", timeout_nodes_reward, epoch)
                if self.trainer_params["fsb_dist_only"]:
                    self.tb_logger.log_value("feasibility/feasible_dist_mean", feasible_dist_mean, epoch)
                    self.tb_logger.log_value("feasibility/feasible_dist_max_pomo_mean", feasible_dist_max_pomo_mean, epoch)
            if self.wandb_logger:
                wandb.log({'train/train_score': train_score})
                wandb.log({'train/train_loss': train_loss})
                try:
                    wandb.log({'feasibility/solution_infeasible_rate': sol_infeasible_rate})
                    wandb.log({'feasibility/instance_infeasible_rate': ins_infeasible_rate})
                except:
                    pass
                if self.trainer_params["timeout_reward"]:
                    wandb.log({"feasibility/total_timeout": total_timeout_reward})
                    wandb.log({"feasibility/timeout_nodes": timeout_nodes_reward})
                if self.trainer_params["fsb_dist_only"]:
                    wandb.log({"feasibility/feasible_dist_mean": feasible_dist_mean})
                    wandb.log({"feasibility/feasible_dist_max_pomo_mean": feasible_dist_max_pomo_mean})

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['model_save_interval']
            validation_interval = self.trainer_params['validation_interval']

            # Validation & save latest images
            try:
                if train_score < best_score:
                    best_score = train_score
                    torch.save(self.model.state_dict(), os.path.join(self.log_path, "trained_model_best.pt"))
                    print(">> Best model saved!")
            except:
                best_score = train_score

            # Save model
            if all_done or (epoch % model_save_interval == 0):
                print("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'problem': self.args.problem,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log,
                }
                torch.save(checkpoint_dict, '{}/epoch-{}.pt'.format(self.log_path, epoch))


            # validation
            if epoch == 1 or (epoch % validation_interval == 0):
                val_problems = [self.args.problem]
                val_episodes, problem_size = self.env_params['val_episodes'], self.env_params['problem_size']
                if self.env_params['val_dataset'] is not None:
                    paths = self.env_params['val_dataset']
                    dir = ["../data/{}/".format(self.args.problem)] * len(paths)
                    val_envs = [get_env(prob)[0] for prob in val_problems] * len(paths)
                else:
                    dir = [os.path.join("../data", prob) for prob in val_problems]
                    paths = ["{}{}_uniform.pkl".format(prob.lower(), problem_size) for prob in val_problems]
                    val_envs = [get_env(prob)[0] for prob in val_problems]
                for i, path in enumerate(paths):
                    # if no optimal solution provided, set compute_gap to False
                    if not self.env_params["pomo_start"]:
                        # sampling pomo_size routes is useless due to the argmax operator when selecting next node based on probability
                        init_pomo_size = self.env_params["pomo_size"]
                        self.env_params["pomo_size"] = 1
                    score, gap, infsb_rate = self._val_and_stat(dir[i], path, val_envs[i](**self.env_params), batch_size=self.trainer_params["validation_batch_size"], val_episodes=val_episodes, epoch = epoch)
                    if not self.env_params["pomo_start"]:
                        self.env_params["pomo_size"] = init_pomo_size
                    self.result_log["val_score"].append(score)
                    self.result_log["val_gap"].append(gap)
                    if infsb_rate is not None:
                        self.result_log["val_infsb_rate"].append(infsb_rate)
                    if self.tb_logger:
                        self.tb_logger.log_value('val_score/{}'.format(path.split(".")[0]), score, epoch)
                        self.tb_logger.log_value('val_gap/{}'.format(path.split(".")[0]), gap, epoch)
                        try:
                            self.tb_logger.log_value('val_sol_infsb_rate/{}'.format(path.split(".")[0]), infsb_rate[0], epoch)
                            self.tb_logger.log_value('val_ins_infsb_rate/{}'.format(path.split(".")[0]), infsb_rate[1], epoch)
                        except:
                            pass
                    if self.wandb_logger:
                        wandb.log({'val_score/{}'.format(path.split(".")[0]): score})
                        wandb.log({'val_gap/{}'.format(path.split(".")[0]): gap})
                        try:
                            wandb.log({'val_sol_infsb_rate/{}'.format(path.split(".")[0]): infsb_rate[0]})
                            wandb.log({'val_ins_infsb_rate/{}'.format(path.split(".")[0]): infsb_rate[1]})
                        except:
                            pass

                    try:
                        if score < best_val_score:
                            best_val_score = score
                            torch.save(self.model.state_dict(), os.path.join(self.log_path, "trained_model_val_best.pt"))
                            print(">> Best model on validation dataset saved!")
                    except:
                        best_val_score = score

    def _train_one_epoch(self, epoch):
        episode = 0
        score_AM, loss_AM, sol_infeasible_rate_AM, ins_infeasible_rate_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        if self.trainer_params["fsb_dist_only"]:
            feasible_dist_mean_AM, feasible_dist_max_pomo_mean_AM = AverageMeter(), AverageMeter()
        if self.trainer_params["timeout_reward"]:
            timeout_AM, timeout_nodes_AM = AverageMeter(), AverageMeter()
        if self.model_params["pip_decoder"] and self.is_train_pip_decoder:
            sl_loss_AM, accuracy_AM, infsb_accuracy_AM, fsb_accuracy_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        total_step = math.floor(train_num_episode /self.trainer_params['train_batch_size'])
        batch_id = 0
        while episode < train_num_episode:
            for accumulation_step in range(self.trainer_params['accumulation_steps']):
                remaining = train_num_episode - episode
                batch_size = min(self.trainer_params['train_batch_size'], remaining)

                env = random.sample(self.envs, 1)[0](**self.env_params)
                data = env.get_random_problems(batch_size, self.env_params["problem_size"])

                avg_score, avg_loss, infeasible, sl_output = self._train_one_batch(data, env, accumulation_step=accumulation_step)

                if sl_output is not None and self.model_params["pip_decoder"]:
                    sl_loss, accuracy, infsb_accuracy, infsb_samples, fsb_accuracy, fsb_samples = sl_output

                    sl_loss_AM.update(sl_loss, infsb_samples+fsb_samples)
                    accuracy_AM.update(accuracy, infsb_samples+fsb_samples)
                    infsb_accuracy_AM.update(infsb_accuracy, infsb_samples)
                    fsb_accuracy_AM.update(fsb_accuracy, fsb_samples)

                    if self.tb_logger:
                        self.tb_logger.log_value('sl_batch/sl_loss', sl_loss, (epoch-1) * total_step + batch_id)
                        self.tb_logger.log_value('sl_batch/accuracy', accuracy, (epoch-1) * total_step + batch_id)
                        self.tb_logger.log_value('sl_batch/infsb_accuracy', infsb_accuracy, (epoch-1) * total_step + batch_id)
                        self.tb_logger.log_value('sl_batch/infsb_samples_number', infsb_samples, (epoch-1) * total_step + batch_id)
                        self.tb_logger.log_value('sl_batch/fsb_accuracy', fsb_accuracy, (epoch-1) * total_step + batch_id)
                        self.tb_logger.log_value('sl_batch/fsb_samples_number', fsb_samples, (epoch-1) * total_step + batch_id)
                    if self.wandb_logger:
                        wandb.log({'sl_batch/sl_loss': sl_loss})
                        wandb.log({'sl_batch/accuracy': accuracy})
                        wandb.log({'sl_batch/infsb_accuracy': infsb_accuracy})
                        wandb.log({'sl_batch/infsb_samples_number': infsb_samples})
                        wandb.log({'sl_batch/fsb_accuracy': fsb_accuracy})
                        wandb.log({'sl_batch/fsb_samples_number': fsb_samples})

                    if self.trainer_params["pip_save"] == "batch":
                        self.accuracy_isbsf = True if accuracy > self.accuracy_bsf else False
                        self.fsb_accuracy_isbsf = True if fsb_accuracy > self.fsb_accuracy_bsf else False
                        self.infsb_accuracy_isbsf = True if infsb_accuracy > self.infsb_accuracy_bsf else False

                        self.accuracy_bsf = accuracy if accuracy > self.accuracy_bsf else self.accuracy_bsf
                        self.fsb_accuracy_bsf = fsb_accuracy if fsb_accuracy > self.fsb_accuracy_bsf else self.fsb_accuracy_bsf
                        self.infsb_accuracy_bsf = infsb_accuracy if infsb_accuracy > self.infsb_accuracy_bsf else self.infsb_accuracy_bsf

                        if self.accuracy_isbsf:
                            if not os.path.exists('{}/accuracy_bsf.pt'.format(self.log_path)) or (infsb_accuracy > 0.75 and fsb_accuracy > 0.75):
                                print("Saving BSF accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format( self.accuracy_bsf * 100, accuracy* 100, infsb_accuracy* 100, fsb_accuracy* 100))
                                checkpoint_dict = {
                                    'epoch': epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'accuracy': accuracy,
                                    'fsb_accuracy': fsb_accuracy,
                                    'infsb_accuracy': infsb_accuracy,
                                }
                                torch.save(checkpoint_dict, '{}/accuracy_bsf.pt'.format(self.log_path))
                        if self.fsb_accuracy_isbsf:
                            if not os.path.exists('{}/fsb_accuracy_bsf.pt'.format(self.log_path)) or (infsb_accuracy > 0.75) or (infsb_accuracy > 0.6 and self.problem=="TSPDL"):
                                print("Saving BSF Feasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(
                                        self.fsb_accuracy_bsf * 100, accuracy * 100, infsb_accuracy * 100, fsb_accuracy * 100))
                                checkpoint_dict = {
                                    'epoch': epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'accuracy': accuracy,
                                    'fsb_accuracy': fsb_accuracy,
                                    'infsb_accuracy': infsb_accuracy,
                                }
                                torch.save(checkpoint_dict, '{}/fsb_accuracy_bsf.pt'.format(self.log_path))
                        if self.infsb_accuracy_isbsf:
                            if not os.path.exists('{}/infsb_accuracy_bsf.pt'.format(self.log_path)) or (fsb_accuracy > 0.75)or (fsb_accuracy > 0.6 and self.problem=="TSPDL"):
                                print("Saving BSF Infeasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(
                                        self.infsb_accuracy_bsf * 100, accuracy * 100, infsb_accuracy * 100,
                                        fsb_accuracy * 100))
                                checkpoint_dict = {
                                    'epoch': epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'accuracy': accuracy,
                                    'fsb_accuracy': fsb_accuracy,
                                    'infsb_accuracy': infsb_accuracy,
                                }
                                torch.save(checkpoint_dict, '{}/infsb_accuracy_bsf.pt'.format(self.log_path))

                if isinstance(infeasible, dict):
                    sol_infeasible_rate = infeasible["sol_infeasible_rate"]
                    ins_infeasible_rate = infeasible["ins_infeasible_rate"]
                    try:
                        feasible_dist_mean, feasible_dist_mean_num = infeasible["feasible_dist_mean"]
                        feasible_dist_max_pomo_mean, feasible_dist_max_pomo_mean_num = infeasible["feasible_dist_max_pomo_mean"]
                        feasible_dist_mean_AM.update(feasible_dist_mean, feasible_dist_mean_num)
                        feasible_dist_max_pomo_mean_AM.update(feasible_dist_max_pomo_mean, feasible_dist_max_pomo_mean_num)
                    except:
                        pass
                else:
                    infeasible_rate = infeasible

                if isinstance(avg_score, list):
                    dist_reward, total_timeout_reward, timeout_nodes_reward = avg_score
                    avg_score = dist_reward
                    timeout_AM.update(total_timeout_reward, batch_size)
                    timeout_nodes_AM.update(timeout_nodes_reward, batch_size)
                score_AM.update(avg_score, batch_size)
                loss_AM.update(avg_loss, batch_size)
                try:
                    sol_infeasible_rate_AM.update(sol_infeasible_rate, batch_size)
                    ins_infeasible_rate_AM.update(ins_infeasible_rate, batch_size)
                except:
                    pass

                episode += batch_size
                batch_id += 1
                if episode >= train_num_episode:
                    break

        # Log Once, for each epoch

        if self.model_params["pip_decoder"] and self.is_train_pip_decoder:
            if self.trainer_params["timeout_reward"]:
                print(
                    'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Timeout: {:.4f}, Timeout_nodes: {:.0f}, Feasible_dist: {:.4f}'.format(
                        epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg,
                               sol_infeasible_rate_AM.avg * 100, ins_infeasible_rate_AM.avg * 100, timeout_AM.avg,
                        timeout_nodes_AM.avg, feasible_dist_max_pomo_mean_AM.avg))
            else:
                print(
                    'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Feasible_dist: {:.4f}'.format(
                        epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg,
                        sol_infeasible_rate_AM.avg * 100, ins_infeasible_rate_AM.avg * 100,
                        feasible_dist_max_pomo_mean_AM.avg))
            print('Epoch {:3d}: PIP-D Loss: {:.4f},  Accuracy: {:.4f}% (BSF: {:.4f}%) [Infeasible: {:.4f}% (BSF: {:.4f}%), Feasible: {:.4f}% (BSF: {:.4f}%)]'.format(epoch, sl_loss_AM.avg, accuracy_AM.avg*100, self.accuracy_bsf*100, infsb_accuracy_AM.avg*100, self.infsb_accuracy_bsf*100, fsb_accuracy_AM.avg*100, self.fsb_accuracy_bsf*100))

            if self.tb_logger:
                self.tb_logger.log_value('sl_epoch/sl_loss', sl_loss_AM.avg, epoch)
                self.tb_logger.log_value('sl_epoch/accuracy', accuracy_AM.avg, epoch)
                self.tb_logger.log_value('sl_epoch/infsb_accuracy', infsb_accuracy_AM.avg, epoch)
                self.tb_logger.log_value('sl_epoch/infsb_samples_number', infsb_accuracy_AM.count, epoch)
                self.tb_logger.log_value('sl_epoch/fsb_accuracy', fsb_accuracy_AM.avg, epoch)
                self.tb_logger.log_value('sl_epoch/fsb_samples_number', fsb_accuracy_AM.count, epoch)
            if self.wandb_logger:
                wandb.log({'sl_epoch/sl_loss': sl_loss_AM.avg})
                wandb.log({'sl_epoch/accuracy': accuracy_AM.avg})
                wandb.log({'sl_epoch/infsb_accuracy': infsb_accuracy_AM.avg})
                wandb.log({'sl_epoch/infsb_samples_number': infsb_accuracy_AM.count})
                wandb.log({'sl_epoch/fsb_accuracy': fsb_accuracy_AM.avg})
                wandb.log({'sl_epoch/fsb_samples_number': fsb_accuracy_AM.count})

            # save lazy model every epoch
            if self.trainer_params["pip_save"] == "epoch":
                self.accuracy_isbsf = True if accuracy_AM.avg > self.accuracy_bsf else False
                self.fsb_accuracy_isbsf = True if fsb_accuracy_AM.avg > self.fsb_accuracy_bsf else False
                self.infsb_accuracy_isbsf = True if infsb_accuracy_AM.avg > self.infsb_accuracy_bsf else False

                self.accuracy_bsf = accuracy_AM.avg if accuracy_AM.avg > self.accuracy_bsf else self.accuracy_bsf
                self.fsb_accuracy_bsf = fsb_accuracy_AM.avg if fsb_accuracy_AM.avg > self.fsb_accuracy_bsf else self.fsb_accuracy_bsf
                self.infsb_accuracy_bsf = infsb_accuracy_AM.avg if infsb_accuracy_AM.avg > self.infsb_accuracy_bsf else self.infsb_accuracy_bsf

                if self.accuracy_isbsf:
                    if not os.path.exists('{}/accuracy_bsf.pt'.format(self.log_path)) or (infsb_accuracy > 0.75 and fsb_accuracy > 0.75):
                        # if not exist, save
                        # then check whether the current batch is bad, if no then save
                        print("Saving BSF accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(self.accuracy_bsf * 100, accuracy* 100, infsb_accuracy* 100, fsb_accuracy* 100))
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'accuracy': accuracy_AM.avg,
                            'fsb_accuracy': fsb_accuracy_AM.avg,
                            'infsb_accuracy': infsb_accuracy_AM.avg,
                        }
                        torch.save(checkpoint_dict, '{}/accuracy_bsf.pt'.format(self.log_path))
                if self.fsb_accuracy_isbsf:
                    if not os.path.exists('{}/fsb_accuracy_bsf.pt'.format(self.log_path)) or infsb_accuracy > 0.75 or  (infsb_accuracy > 0.6 and self.problem=="TSPDL"):
                        # if not exist, save
                        # then check whether the current batch is bad, if yes then don't save
                        print("Saving BSF Feasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format( self.fsb_accuracy_bsf * 100, accuracy* 100, infsb_accuracy* 100, fsb_accuracy* 100))
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'accuracy': accuracy_AM.avg,
                            'fsb_accuracy': fsb_accuracy_AM.avg,
                            'infsb_accuracy': infsb_accuracy_AM.avg,
                        }
                        torch.save(checkpoint_dict, '{}/fsb_accuracy_bsf.pt'.format(self.log_path))
                if self.infsb_accuracy_isbsf:
                    if not os.path.exists('{}/infsb_accuracy_bsf.pt'.format(self.log_path)) or fsb_accuracy > 0.75 or (fsb_accuracy > 0.6 and self.problem=="TSPDL"):
                        # if not exist, save
                        # then check whether the current batch is bad, if yes then don't save
                        print("Saving BSF Infeasible accuracy ({:.4f}%) trained_model [Accuracy: {:.4f}%, Infeasible: {:.4f}%, Feasible: {:.4f}%]".format(self.infsb_accuracy_bsf * 100, accuracy* 100, infsb_accuracy* 100, fsb_accuracy* 100))
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'accuracy': accuracy_AM.avg,
                            'fsb_accuracy': fsb_accuracy_AM.avg,
                            'infsb_accuracy': infsb_accuracy_AM.avg,
                        }
                        torch.save(checkpoint_dict, '{}/infsb_accuracy_bsf.pt'.format(self.log_path))
        else:
            if self.trainer_params["timeout_reward"]:
                print(
                    'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Timeout: {:.4f}, Timeout_nodes: {:.0f}, Feasible_dist: {:.4f}'.format(
                        epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg,
                        sol_infeasible_rate_AM.avg * 100, ins_infeasible_rate_AM.avg * 100, timeout_AM.avg,
                        timeout_nodes_AM.avg, feasible_dist_max_pomo_mean_AM.avg))
            else:
                try:
                    print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Feasible_dist: {:.4f}'.format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg, sol_infeasible_rate_AM.avg*100, ins_infeasible_rate_AM.avg*100, feasible_dist_max_pomo_mean_AM.avg))
                except:
                    print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'.format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))

        if self.trainer_params["fsb_dist_only"]:
            try:
                infeasible_output = [sol_infeasible_rate_AM.avg, ins_infeasible_rate_AM.avg, feasible_dist_mean_AM.avg, feasible_dist_max_pomo_mean_AM.avg]
            except:
                infeasible_output = None
        else:
            infeasible_output = [sol_infeasible_rate_AM.avg, ins_infeasible_rate_AM.avg]

        if self.trainer_params["timeout_reward"]:
            score_output = [score_AM.avg, timeout_AM.avg, timeout_nodes_AM.avg]
        else:
            score_output = score_AM.avg

        return score_output, loss_AM.avg, infeasible_output

    def _train_one_batch(self, data, env, accumulation_step):

        self.model.train()
        self.model.set_eval_type(self.model_params["eval_type"])

        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)

        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        if self.model_params["pip_decoder"] and self.lazy_model is not None and (not self.is_train_pip_decoder):
            self.lazy_model.eval()
            self.lazy_model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        if self.model_params["pip_decoder"] and self.is_train_pip_decoder:
            sl_loss_list = torch.zeros(size=(0,))
            pred_LIST, label_LIST = np.array([]), np.array([])

        ########################################
        ############ POMO Rollout ##############
        ########################################
        state, reward, done = env.pre_step()
        while not done:
            # Use PIP decoder to predict PI masking when the decoder is not trained.
            use_predicted_PI_mask = True if (self.model_params['pip_decoder'] and not self.is_train_pip_decoder) else False
            if self.model_params["pip_decoder"] and self.lazy_model is not None and env.selected_count >= 1 and (not self.is_train_pip_decoder):
                with torch.no_grad():
                    use_predicted_PI_mask = self.lazy_model(state, pomo=self.env_params["pomo_start"],
                                                            use_predicted_PI_mask = False, no_select_prob= True,
                                                            tw_end = env.node_tw_end if self.problem == "TSPTW" else None,
                                                            no_sigmoid = (self.trainer_params["sl_loss"] == "BCEWithLogitsLoss"))
            # Forward
            selected, prob = self.model(state, pomo=self.env_params["pomo_start"],
                                            use_predicted_PI_mask=use_predicted_PI_mask,
                                            tw_end = env.node_tw_end if self.problem == "TSPTW" else None,
                                            no_sigmoid = (self.trainer_params["sl_loss"] == "BCEWithLogitsLoss"))
            # Calculate the loss for the PIP decoder
            if self.model_params['pip_decoder']:
                prob, probs_sl = prob
                if self.model_params['pip_decoder'] and env.selected_count >= 1 and (env.selected_count < env.problem_size - 1) and self.is_train_pip_decoder:
                    # FIXME: now still calculate the loss when left 2 nodes unvisited (not necessary?)
                    visited_mask = env.visited_ninf_flag == float('-inf')
                    if env.is_sparse: visited_mask = ~env.visited_and_notneigh_ninf_flag
                    sl_losses = torch.tensor(0.)
                    label = torch.where(env.simulated_ninf_flag == float('-inf'), 1., env.simulated_ninf_flag)
                    label = label[~visited_mask]
                    if label.sum() != 0 and label.sum() != label.reshape(-1).size(-1):  # not all fsb or all infsb
                        probs_sl = probs_sl[~visited_mask]
                        if self.trainer_params["label_balance_sampling"]:
                            if self.trainer_params["fast_label_balance"]:
                                # new version: accelerate the calculation of smaple weights
                                assert self.trainer_params["sl_loss"] == "BCEWithLogitsLoss", "only BCEWithLogitsLoss (output with no sigmoid) is supported when label_balance_sampling==True with speedup!"
                                infsb_sample_number = torch.nonzero(label != 0).size(0)  # positive
                                fsb_sample_number = torch.nonzero(label == 0).size(0)  # negative
                                pos_weight = fsb_sample_number / infsb_sample_number  # neg / pos
                                pos_weight = torch.ones_like(label) * pos_weight
                                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                                sl_loss = criterion(probs_sl, label)
                                if self.trainer_params["fast_weight"]:
                                    sl_weight = (fsb_sample_number + infsb_sample_number) / (2 * fsb_sample_number)
                                    # with this weight, fast method totally equals to the non-fast one
                                    sl_loss = sl_loss * sl_weight
                            else:
                                # assert self.trainer_params["sl_loss"] == "BCELoss", "only BCELoss is supported when label_balance_sampling==True without speedup!"
                                edge_labels = (label != 0).int().cpu().numpy().flatten()
                                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
                                if self.trainer_params["sl_loss"] == "BCELoss":
                                    probs_sl = torch.clamp(probs_sl, min=1e-7, max=1 - 1e-7)  # add a clamp to avoid numerical instability
                                    sl_loss = - edge_cw[1] * (label * torch.log(probs_sl)) - edge_cw[0] * (((1 - label) * torch.log(1 - probs_sl)))
                                elif self.trainer_params["sl_loss"] == "BCEWithLogitsLoss":
                                    sl_loss = - edge_cw[1] * (label * torch.log(F.sigmoid(probs_sl))) - edge_cw[0] * (((1 - label) * torch.log(1 - F.sigmoid(probs_sl))))
                                else:
                                    raise NotImplementedError(f"Unsupported sl_loss: {self.trainer_params['sl_loss']}")
                                sl_loss = sl_loss.mean()
                        else:
                            if self.trainer_params["sl_loss"] == "BCEWithLogitsLoss":
                                criterion = nn.BCEWithLogitsLoss()
                            elif self.trainer_params["sl_loss"] == "BCELoss":
                                criterion = nn.BCELoss()
                            else:
                                raise NotImplementedError(f"Unsupported sl_loss: {self.trainer_params['sl_loss']}")
                            sl_loss = criterion(probs_sl, label)
                        # sl_loss shape: (batch, pomo)
                        label = label.reshape(-1)
                        probs_sl = probs_sl.reshape(-1) if self.trainer_params["sl_loss"] != "BCEWithLogitsLoss" else F.sigmoid(probs_sl).reshape(-1)
                        pred_LIST = np.append(pred_LIST, probs_sl.detach().cpu().numpy())
                        label_LIST = np.append(label_LIST, label.detach().cpu().numpy())
                        sl_losses += sl_loss
                    sl_loss_list = torch.cat([sl_loss_list, sl_losses.unsqueeze(0)], dim=0)

            # if True, then don't use predicted PI mask
            use_predicted_PI_mask = ((not isinstance(use_predicted_PI_mask, bool)  # if True, PI mask is predicted from the PIP decoder
                                      or use_predicted_PI_mask) # PIP decoder isn't training
                                     or not self.trainer_params["use_real_PI_mask"])  # don't use real PI mask

            # Step
            state, reward, done, infeasible = env.step(selected,
                                                       out_reward = self.trainer_params["timeout_reward"],
                                                       generate_PI_mask = self.model_params["generate_PI_mask"],
                                                       use_predicted_PI_mask = use_predicted_PI_mask,
                                                       pip_step = self.trainer_params["pip_step"])
            # print(">> Cause Infeasibility: Inlegal rate: {}".format(infeasible_rate))

            # Handle outputs
            if isinstance(infeasible, list):
                infeasible, infsb_level_value = infeasible
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)




        ########################################
        ############ Calculate Loss ############
        ########################################
        # Rewards calculation
        infeasible_output = infeasible
        if isinstance(reward, list):
            dist_reward, total_timeout_reward, timeout_nodes_reward = reward
            dist = dist_reward.clone()
        else:
            dist_reward = reward
            dist = reward
        if self.trainer_params["fsb_dist_only"]:
            problem_size, pomo_size = self.env_params["problem_size"], env.pomo_size
            feasible_number = (batch_size*pomo_size) - infeasible.sum()
            feasible_dist_mean, feasible_dist_max_pomo_mean = 0., 0.
            batch_feasible = torch.tensor([0.])
            if feasible_number:
                feasible_dist = torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward) # feasible dist left only
                feasible_dist_mean = -feasible_dist.sum() / feasible_number # negative sign to make positive value, and calculate mean
                feasible_dist_mean = (feasible_dist_mean, feasible_number)
                reward_masked = dist.masked_fill(infeasible, -1e10)  # get feasible results from pomo
                feasible_max_pomo_dist = reward_masked.max(dim=1)[0]# get best results from pomo, shape: (batch)
                # feasible_max_pomo_dist = dist.max(dim=1)[0] # get best results from pomo, shape: (batch)
                batch_feasible = (infeasible==False).any(dim=-1) # shape: (batch)
                feasible_max_pomo_dist = torch.where(batch_feasible==False, torch.zeros_like(feasible_max_pomo_dist), feasible_max_pomo_dist) # feasible dist left only
                feasible_dist_max_pomo_mean = -feasible_max_pomo_dist.sum() / batch_feasible.sum() # negative sign to make positive value, and calculate mean
                feasible_dist_max_pomo_mean = (feasible_dist_max_pomo_mean, batch_feasible.sum())

            infeasible_output = {
                "sol_infeasible_rate": infeasible.sum() / (batch_size*pomo_size),
                "ins_infeasible_rate": 1. - batch_feasible.sum() / batch_size,
                "feasible_dist_mean": feasible_dist_mean,
                "feasible_dist_max_pomo_mean": feasible_dist_max_pomo_mean
            }
        if isinstance(reward, list):
            reward = dist +  self.penalty_factor * (total_timeout_reward +  timeout_nodes_reward)  # (batch, pomo)
        if not self.trainer_params["timeout_reward"] and self.trainer_params["fsb_reward_only"]: # activate when not using LM
            feasible_reward_number = (infeasible==False).sum(-1)
            feasible_reward_mean = (torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward).sum(-1) / feasible_reward_number)[:,None]
            feasible_advantage = dist_reward - feasible_reward_mean
            feasible_advantage = torch.masked_select(feasible_advantage, infeasible==False)
            log_prob = torch.masked_select(prob_list.log().sum(dim=2), infeasible==False)
            advantage = feasible_advantage
        else:
            advantage = reward - reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)
            log_prob = prob_list.log().sum(dim=2)
        loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
        loss_mean = loss.mean()
        # add SL loss
        if self.model_params['pip_decoder'] and self.is_train_pip_decoder:
            sl_loss_mean = sl_loss_list.mean()
            loss_mean = sl_loss_mean if loss_mean.isnan() else loss_mean + sl_loss_mean
        # Calculate the prediction accuracy
        if self.model_params['pip_decoder'] and self.is_train_pip_decoder:
            try:
                tn, fp, fn, tp = confusion_matrix((label_LIST > self.trainer_params["decision_boundary"]).astype(np.int32),
                                                  (pred_LIST > self.trainer_params["decision_boundary"]).astype(np.int32)).ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                infsb_accuracy = tp / (fn + tp)
                fsb_accuracy = tn / (tn + fp)
            except:
                accuracy = 0.
                infsb_accuracy = 0.
                fsb_accuracy = 0.
                tn, fp, fn, tp = 0, 0, 0, 0

            # tn, fp, fn, tp = confusion_matrix((labels > 0.5).int().cpu(), (F.sigmoid(predict_out) > 0.5).int().cpu()).ravel()
            # print('Accuracy: {:.4f}% [Infeasible: {:.4f}% ({}), Feasible: {:.4f}% ({})]'.format(accuracy, infsb_accuracy, (fn + tp), fsb_accuracy, (tn + fp)))

        ########################################
        ############# Step & Return ############
        ########################################
        if accumulation_step == 0:
            self.model.zero_grad()
        loss_mean = loss_mean/self.trainer_params["accumulation_steps"]
        loss_mean.backward()
        if accumulation_step == self.trainer_params["accumulation_steps"] - 1:
            # update the parameters until accumulating enough accumulation_steps
            self.optimizer.step()

        if not self.trainer_params["timeout_reward"]:
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
            score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
            score_mean = score_mean.item()
        else:
            max_dist_reward = dist_reward.max(dim=1)[0]  # get best results from pomo
            dist_mean = -max_dist_reward.float().mean()  # negative sign to make positive value
            max_timeout_reward = total_timeout_reward.max(dim=1)[0]  # get best results from pomo
            timeout_mean = -max_timeout_reward.float().mean()  # negative sign to make positive value
            max_timeout_nodes_reward = timeout_nodes_reward.max(dim=1)[0]  # get best results from pomo
            timeout_nodes_mean = -max_timeout_nodes_reward.float().mean()  # negative sign to make positive value
            score_mean = [dist_mean, timeout_mean, timeout_nodes_mean]

        loss_out = loss_mean.item()
        if self.model_params['pip_decoder'] and self.is_train_pip_decoder:
            sl_loss_out = sl_loss_list.mean().item()
            return score_mean, loss_out, infeasible_output, [sl_loss_out, accuracy, infsb_accuracy, (fn + tp), fsb_accuracy, (tn + fp)]
        else:
            return score_mean, loss_out, infeasible_output, None

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
        self.model.eval()
        self.model.set_eval_type(eval_type)

        if self.model_params["pip_decoder"]:
            pred_LIST, label_LIST = np.array([]), np.array([])
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor, normalize=True)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            if self.model_params["pip_decoder"] and (self.lazy_model is not None) and (not self.is_train_pip_decoder):
                # ATTENTION: only use the predicted mask for validation when not training?
                self.lazy_model.eval()
                self.lazy_model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:

                use_predicted_PI_mask = True if (self.model_params['pip_decoder'] and not self.is_train_pip_decoder) else False
                # print(use_predicted_PI_mask)
                if self.model_params["pip_decoder"] and self.lazy_model is not None and not (self.is_train_pip_decoder) and env.selected_count >= 1:
                    use_predicted_PI_mask = self.lazy_model(state, pomo=self.env_params["pomo_start"],
                                                            tw_end = env.node_tw_end if self.problem == "TSPTW" else None,
                                                            use_predicted_PI_mask=False, no_select_prob=True)
                selected, prob = self.model(state, pomo=self.env_params["pomo_start"],
                                               tw_end = env.node_tw_end if self.problem == "TSPTW" else None,
                                               use_predicted_PI_mask=use_predicted_PI_mask)
                # shape: (batch, pomo)
                # state, reward, done, infeasible = env.step(selected,timeout_reward=self.trainer_params["timeout_reward"])
                if self.model_params['pip_decoder']:
                    _, probs_sl = prob
                    if self.model_params['pip_decoder'] and (env.selected_count >= 1) and (env.selected_count < env.problem_size - 1):
                        label = torch.where(env.simulated_ninf_flag == float('-inf'), 1., env.simulated_ninf_flag)
                        visited_mask = (env.visited_ninf_flag == float('-inf'))
                        if env.is_sparse: visited_mask = ~env.visited_and_notneigh_ninf_flag
                        label = label[~visited_mask]
                        probs_sl = probs_sl[~visited_mask]
                        pred_LIST = np.append(pred_LIST, probs_sl.detach().cpu().numpy())
                        label_LIST = np.append(label_LIST, label.detach().cpu().numpy())

                # ATTENTION: always generate PI mask during validation
                generate_PI_mask = True if self.model_params['pip_decoder'] else False
                # print(generate_PI_mask)
                use_predicted_PI_mask = ((not isinstance(use_predicted_PI_mask, bool) or use_predicted_PI_mask==True) or not self.trainer_params["use_real_PI_mask"])
                state, reward, done, infeasible = env.step(selected,
                                                           generate_PI_mask=generate_PI_mask,
                                                           use_predicted_PI_mask = use_predicted_PI_mask,
                                                           pip_step = self.trainer_params["pip_step"])

        # Return
        if isinstance(reward, list):
            dist_reward, total_timeout_reward, timeout_nodes_reward = reward
            dist = dist_reward.clone()

            aug_total_timeout_reward = total_timeout_reward.reshape(aug_factor, batch_size, env.pomo_size)
            # shape: (augmentation, batch, pomo)
            max_pomo_total_timeout_reward, _ = aug_total_timeout_reward.max(dim=2)  # get best results from pomo
            no_aug_total_timeout_score = -max_pomo_total_timeout_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_total_timeout_reward, _ = max_pomo_total_timeout_reward.max(dim=0)  # get best results from augmentation
            aug_total_timeout_score = -max_aug_pomo_total_timeout_reward.float()  # negative sign to make positive value

            aug_timeout_nodes_reward = timeout_nodes_reward.reshape(aug_factor, batch_size, env.pomo_size)
            # shape: (augmentation, batch, pomo)
            max_pomo_timeout_nodes_reward, _ = aug_timeout_nodes_reward.max(dim=2)  # get best results from pomo
            no_aug_timeout_nodes_score = -max_pomo_timeout_nodes_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_timeout_nodes_reward, _ = max_pomo_timeout_nodes_reward.max(dim=0)  # get best results from augmentation
            aug_timeout_nodes_score = -max_aug_pomo_timeout_nodes_reward.float()  # negative sign to make positive value
        else:
            dist = reward

        aug_reward = dist.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)

        if self.trainer_params["fsb_dist_only"]:
            # shape: (augmentation, batch, pomo)
            infeasible = infeasible.reshape(aug_factor, batch_size, env.pomo_size)  # shape: (augmentation, batch, pomo)
            no_aug_feasible = (infeasible[0, :, :] == False).any(dim=-1)  # shape: (batch)
            aug_feasible = (infeasible == False).any(dim=0).any(dim=-1)  # shape: (batch)

            reward_masked = aug_reward.masked_fill(infeasible, -1e10) # get feasible results from pomo
            fsb_no_aug = reward_masked[0,:,:].max(dim=1, keepdim=True).values # shape: (augmentation, batch)
            fsb_aug = reward_masked.max(dim=0).values.max(dim=-1).values
            no_aug_score, aug_score = -fsb_no_aug, -fsb_aug

            infeasible_output = {
                "sol_infeasible_rate": infeasible.sum() / (batch_size * env.pomo_size * aug_factor),
                "ins_infeasible_rate": 1. - aug_feasible.sum() / batch_size,
                "no_aug_feasible": no_aug_feasible,
                "aug_feasible": aug_feasible
            }
        else:
            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
            infeasible_output = infeasible

        if self.model_params["pip_decoder"]:
            return no_aug_score, aug_score, infeasible_output, pred_LIST, label_LIST
        return no_aug_score, aug_score, infeasible_output, None, None

    def _val_and_stat(self, dir, val_path, env, batch_size=500, val_episodes=1000, compute_gap=False, epoch=1):
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list, sol_infeasible_rate_list, ins_infeasible_rate_list = [], [], [], [], [], []
        episode, no_aug_score, aug_score, sol_infeasible_rate, ins_infeasible_rate = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device), torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        # if self.trainer_params["timeout_reward"]:
        #     no_aug_total_timeout_score, no_aug_timeout_nodes_score = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        #     aug_total_timeout_score, aug_timeout_nodes_score = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        if self.trainer_params["fsb_dist_only"]:
            no_aug_feasible, aug_feasible = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        if self.model_params["pip_decoder"]:
            pred_LIST, label_LIST = np.array([]), np.array([])
        if self.model_params["pip_decoder"] and (self.lazy_model is not None) and (not self.is_train_pip_decoder):
            print(">> Use PIP-D predicted mask for validation...")
        elif self.trainer_params["use_real_PI_mask"] and self.model_params["generate_PI_mask"]:
            print(">> Use PI masking for validation...")

        while episode < val_episodes:
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            data = env.load_dataset(os.path.join(dir, val_path), offset=episode, num_samples=bs)
            no_aug, aug, infsb_rate, pred_list, label_list  = self._val_one_batch(data, env, aug_factor=8, eval_type="argmax")
            if isinstance(aug, list):
                no_aug, no_aug_total_timeout, no_aug_timeout_nodes = no_aug
                aug, aug_total_timeout, aug_timeout_nodes = aug
            no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
            aug_score = torch.cat((aug_score, aug), dim=0)
            if isinstance(infsb_rate, dict):
                no_aug_fsb = infsb_rate["no_aug_feasible"]
                aug_fsb = infsb_rate["aug_feasible"]
                sol_infsb_rate = infsb_rate["sol_infeasible_rate"]
                ins_infsb_rate = infsb_rate["ins_infeasible_rate"]
                no_aug_feasible = torch.cat((no_aug_feasible, no_aug_fsb), dim=0)
                aug_feasible = torch.cat((aug_feasible, aug_fsb), dim=0)
            try:
                sol_infeasible_rate = torch.cat((sol_infeasible_rate, torch.tensor([sol_infsb_rate])), dim=0)
                ins_infeasible_rate = torch.cat((ins_infeasible_rate, torch.tensor([ins_infsb_rate])), dim=0)
            except:
                pass
            episode += bs
            if self.model_params["pip_decoder"]:
                pred_LIST = np.append(pred_LIST, pred_list)
                label_LIST = np.append(label_LIST, label_list)

        if self.model_params["pip_decoder"]:
            tn, fp, fn, tp = confusion_matrix((label_LIST > 0.5).astype(np.int32),(pred_LIST > 0.5).astype(np.int32)).ravel()
            # tn, fp, fn, tp = confusion_matrix((labels > 0.5).int().cpu(), (F.sigmoid(predict_out) > 0.5).int().cpu()).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            infsb_accuracy = tp / (fn + tp)
            fsb_accuracy = tn / (tn + fp)
            if self.tb_logger:
                self.tb_logger.log_value('val_sl/accuracy', accuracy, epoch)
                self.tb_logger.log_value('val_sl/infsb_accuracy', infsb_accuracy, epoch)
                self.tb_logger.log_value('val_sl/fsb_accuracy', fsb_accuracy, epoch)
                self.tb_logger.log_value('val_sl/infsb_sample_nums', (label_LIST > 0.5).astype(np.int32).sum(), epoch)
                self.tb_logger.log_value('val_sl/fsb_sample_nums', (label_LIST < 0.5).astype(np.int32).sum(), epoch)
            if self.wandb_logger:
                wandb.log({'val_sl/accuracy': accuracy})
                wandb.log({'val_sl/infsb_accuracy': infsb_accuracy})
                wandb.log({'val_sl/fsb_accuracy': fsb_accuracy})
                wandb.log({'val_sl/infsb_sample_nums': (label_LIST > 0.5).astype(np.int32).sum()})
                wandb.log({'val_sl/fsb_sample_nums': (label_LIST < 0.5).astype(np.int32).sum()})
            print("PIP-D Validation, Auc: {:.4f}, Infeasible Auc: {:.4f} ({}), Feasible Auc: {:.4f} ({})".format(accuracy, infsb_accuracy,(fn + tp), fsb_accuracy,(tn + fp)))
        if self.trainer_params["fsb_dist_only"]:
            print(">> Only feasible solutions are under consideration!")
            no_aug_score_list.append(round(no_aug_score[no_aug_feasible.bool()].mean().item(), 4))
            aug_score_list.append(round(aug_score[aug_feasible.bool()].mean().item(), 4))
        else:
            no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
            aug_score_list.append(round(aug_score.mean().item(), 4))
        if sol_infeasible_rate.size(0) > 0:
            sol_infeasible_rate_list.append(round(sol_infeasible_rate.mean().item()*100, 3))
            ins_infeasible_rate_list.append(round(ins_infeasible_rate.mean().item() * 100, 3))

        try:
            sol_path = get_opt_sol_path(dir, env.problem, data[1].size(1))
        except:
            sol_path = os.path.join(dir, "lkh_" + val_path)

        compute_gap = os.path.exists(sol_path)

        if compute_gap:
            opt_sol = load_dataset(sol_path, disable_print=True)[: val_episodes]
            # grid_factor = 1.
            grid_factor = 100. if self.args.problem == "TSPTW" else 1.
            opt_sol = torch.tensor([i[0]/grid_factor for i in opt_sol])
            if self.trainer_params["fsb_dist_only"]:
                gap = (no_aug_score[no_aug_feasible.bool()] - opt_sol[no_aug_feasible.bool()]) / opt_sol[no_aug_feasible.bool()] * 100
                aug_gap = (aug_score[aug_feasible.bool()] - opt_sol[aug_feasible.bool()]) / opt_sol[aug_feasible.bool()] * 100
            else:
                gap = (no_aug_score - opt_sol) / opt_sol * 100
                aug_gap = (aug_score - opt_sol) / opt_sol * 100
            no_aug_gap_list.append(round(gap.mean().item(), 4))
            aug_gap_list.append(round(aug_gap.mean().item(), 4))
            try:
                print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}%; Infeasible rate: {}% (solution-level), {}% (instance-level)".format(val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list, sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]))
                return aug_score_list[0], aug_gap_list[0], [sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]]
            except:
                print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}%".format(val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list))
                return aug_score_list[0], aug_gap_list[0], None

        else:
            print(">> Val Score on {}: NO_AUG_Score: {}, --> AUG_Score: {}; Infeasible rate: {}% (solution-level), {}% (instance-level)".format(val_path, no_aug_score_list, aug_score_list, sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]))
            return aug_score_list[0], 0, [sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]]
