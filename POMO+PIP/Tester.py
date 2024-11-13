import pandas as pd
from utils import *
from models.SINGLEModel import SINGLEModel


class Tester:
    def __init__(self, args, env_params, model_params, tester_params):

        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # ENV, MODEL, & Load checkpoint
        self.envs = get_env(self.args.problem)  # Env Class
        self.device = args.device
        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        self.model = SINGLEModel(**self.model_params)
        num_param(self.model)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            # self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except:
            self.model.load_state_dict(checkpoint, strict=False)
            # self.model.load_state_dict(checkpoint, strict=True)
        print(">> Checkpoint ({}) Loaded!".format(args.checkpoint))

        if self.tester_params["lazy_pip_model"] and self.tester_params["use_predicted_PI_mask"]:
            self.lazy_model = SINGLEModel(**self.model_params)
        else:
            self.lazy_model = None

        if args.pip_checkpoint and self.tester_params["use_predicted_PI_mask"]:
            checkpoint_fullname = args.pip_checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            try:
                self.lazy_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except:
                self.lazy_model.load_state_dict(checkpoint, strict=True)
            try:
                print(">> Load lazy PIP-D model from {} [Accuracy: {:.4f}%; Infeasible: {:.4f}%; Feasible: {:.4f}%]".format(
                        checkpoint_fullname, checkpoint['accuracy'] * 100, checkpoint['infsb_accuracy'] * 100,
                                             checkpoint['fsb_accuracy'] * 100))
            except:
                print(">> Load lazy PIP-D model from {}".format(checkpoint_fullname))

        # load dataset
        if tester_params['test_set_path'] is None or tester_params['test_set_path'].endswith(".pkl"):
            self.data_dir = "./data"
        else:
            # for solving instances with CVRPLIB format
            self.path_list = [os.path.join(tester_params['test_set_path'], f) for f in sorted(os.listdir(tester_params['test_set_path']))] \
                if os.path.isdir(tester_params['test_set_path']) else [tester_params['test_set_path']]
            # assert self.path_list[-1].endswith(".vrp"), "Unsupported file types."

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        for env_class in self.envs:
            start_time = time.time()
            if self.tester_params['test_set_path'] is None or self.tester_params['test_set_path'].endswith(".pkl"):
                compute_gap = not (self.tester_params['test_set_path'] is not None and self.tester_params['test_set_opt_sol_path'] is None)
                scores, aug_scores, sol_infeasible_rate, ins_infeasible_rate = self._test(env_class, compute_gap=compute_gap)
            else:
                results = []
                for i, path in enumerate(self.path_list):
                    if self.args.problem == "TSPDL":
                        no_aug_score, aug_score, ins_infeasible_rate, sol_infeasible_rate = self._solve_tspdllib(path, env_class)
                        name = path.split('/')[-1].split('.pkl')[0]
                        results.append([name, no_aug_score, aug_score, ins_infeasible_rate, sol_infeasible_rate])
                    elif self.args.problem == "TSPTW":
                        no_aug_score, aug_score, ins_infeasible_rate, sol_infeasible_rate = self._solve_tsptwlib(path, env_class)
                        name = path.split('/')[-1].split('.pkl')[0]
                        results.append([name, no_aug_score, aug_score, ins_infeasible_rate, sol_infeasible_rate])
                    else:
                        raise NotImplementedError
                df = pd.DataFrame(np.array(results))
                excel_file = f"{self.args.problem}_lib.xlsx"
                df.to_excel(excel_file, index=False, header=False)
            print(">> Evaluation finished within {:.2f}s\n".format(time.time() - start_time))
        return scores, aug_scores, sol_infeasible_rate, ins_infeasible_rate

    def _test(self, env_class, compute_gap=False):
        self.time_estimator.reset()
        env = env_class(**self.env_params)
        score_AM, gap_AM, sol_infeasible_rate_AM, ins_infeasible_rate_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        aug_score_AM, aug_gap_AM = AverageMeter(), AverageMeter()
        scores, aug_scores = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        no_aug_feasibles, aug_feasibles = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        opt_sols = torch.zeros(0).to(self.device)
        episode, test_num_episode = 0, self.tester_params['test_episodes']

        data_path = self.tester_params['test_set_path'] if self.tester_params['test_set_path'] \
            else os.path.join(self.data_dir, env.problem, "{}{}_uniform.pkl".format(env.problem.lower(), env.problem_size))

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            data = env.load_dataset(data_path, offset=episode, num_samples=batch_size)
            score, aug_score, all_score, all_aug_score, sol_infeasible_rate, ins_infeasible_rate, no_aug_feasible, aug_feasible = self._test_one_batch(data, env)
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            sol_infeasible_rate_AM.update(sol_infeasible_rate*100, batch_size)
            ins_infeasible_rate_AM.update(ins_infeasible_rate * 100, batch_size)
            scores = torch.cat((scores, all_score), dim=0)
            aug_scores = torch.cat((aug_scores, all_aug_score), dim=0)
            no_aug_feasibles = torch.cat((no_aug_feasibles, no_aug_feasible), dim=0)
            aug_feasibles = torch.cat((aug_feasibles, aug_feasible), dim=0)

            # compute_gap = False
            if compute_gap:
                opt_sol_path = self.tester_params['test_set_opt_sol_path'] if self.tester_params['test_set_opt_sol_path'] \
                    else get_opt_sol_path(os.path.join(self.data_dir, env.problem), env.problem, env.problem_size)
                print(">> Load optimal solution path: {}".format(opt_sol_path))
                opt_sol = load_dataset(opt_sol_path, disable_print=True)[episode: episode + batch_size]  # [(obj, route), ...]
                opt_sol = [i[0] /100 for i in opt_sol] if self.args.problem=="TSPTW" else [i[0]  for i in opt_sol]
                opt_sols = torch.cat((opt_sols, torch.tensor(opt_sol).float()), dim=0)
                if self.tester_params['fsb_dist_only']:
                    gap, aug_gap = [], []
                    for i in range(batch_size):
                        if no_aug_feasible[i]:
                            gap.append((all_score[i].item() - opt_sol[i]) / opt_sol[i] * 100)
                        if aug_feasible[i]:
                            aug_gap.append((all_aug_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 )
                    if len(gap):
                        gap_AM.update(sum(gap) / len(gap), len(gap))
                    if len(aug_gap):
                        aug_gap_AM.update(sum(aug_gap) / len(aug_gap), len(aug_gap))
                else:
                    gap = [(all_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
                    aug_gap = [(all_aug_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
                    gap_AM.update(sum(gap)/batch_size, batch_size)
                    aug_gap_AM.update(sum(aug_gap)/batch_size, batch_size)

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            print("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}, Sol-Infeasible_rate: {:.3f}, Ins-Infeasible_rate: {:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score, sol_infeasible_rate, ins_infeasible_rate))

            all_done = (episode == test_num_episode)

            if all_done:
                print(" \n*** Test Done on {} *** ".format(env.problem))
                print(" NO-AUG SCORE: {}, Gap: {:.4f} ".format(score_AM.avg, gap_AM.avg))
                print(" AUGMENTATION SCORE: {}, Gap: {:.4f} ".format(aug_score_AM.avg, aug_gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(score_AM.avg, gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(aug_score_AM.avg, aug_gap_AM.avg))
                print("Solution level Infeasible rate: {:.3f}%".format(sol_infeasible_rate_AM.avg))
                print("Instance level Infeasible rate: {:.3f}%".format(ins_infeasible_rate_AM.avg))

        return scores, aug_scores, sol_infeasible_rate_AM.avg, ins_infeasible_rate_AM.avg

    def _test_one_batch(self, test_data, env):
        aug_factor = self.tester_params['aug_factor']
        batch_size = test_data.size(0) if isinstance(test_data, torch.Tensor) else test_data[-1].size(0)
        sample_size = self.tester_params['sample_size'] if self.model_params['eval_type'] == "softmax" else 1

        # Sampling: augment data based on sample_size: [batch_size, ...] -> [batch_size x sample_size, ...]
        if self.model_params['eval_type'] == "softmax":
            test_data = list(test_data)
            for i, data in enumerate(test_data):
                if data.dim() == 1:
                    test_data[i] = data.repeat(sample_size)
                elif data.dim() == 2:
                    test_data[i] = data.repeat(sample_size, 1)
                elif data.dim() == 3:
                    test_data[i] = data.repeat(sample_size, 1, 1)

        # Ready
        self.model.eval()
        with torch.no_grad():
            env.load_problems(batch_size, problems=test_data, aug_factor=aug_factor, normalize=True)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            if self.model_params["pip_decoder"] and self.lazy_model is not None:
                self.lazy_model.eval()
                self.lazy_model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:

            use_predicted_PI_mask = True if (self.model_params['pip_decoder'] and self.tester_params['use_predicted_PI_mask']) else False
            # print(use_predicted_PI_mask)
            # if self.model_params["pip_decoder"] and self.lazy_model is not None and env.selected_count >= 1:
            #     # use when not training the lazy PIP-D model
            #     with torch.no_grad():
            #         use_predicted_PI_mask = self.lazy_model(state, pomo=self.env_params["pomo_start"],
            #                                       tw_end=env.node_tw_end if self.args.problem == "TSPTW" else None,
            #                                       use_predicted_PI_mask=False, no_select_prob=True,
            #                                       no_sigmoid=True)
            selected, prob = self.model(state, pomo=self.env_params["pomo_start"],
                                                      tw_end=env.node_tw_end if self.args.problem == "TSPTW" else None,
                                                      use_predicted_PI_mask=use_predicted_PI_mask, no_sigmoid=True)
            # shape: (batch, pomo)
            state, reward, done, infeasible = env.step(selected,
                                                       generate_PI_mask=self.model_params["generate_PI_mask"],
                                                       pip_step=self.tester_params["pip_step"])

        # Return
        aug_reward = reward.reshape(aug_factor * sample_size, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)

        infeasible = infeasible.reshape(aug_factor * sample_size, batch_size, env.pomo_size)  # shape: (augmentation, batch, pomo)
        no_aug_feasible = (infeasible[0, :, :] == False).any(dim=-1)  # shape: (batch)
        aug_feasible = (infeasible == False).any(dim=0).any(dim=-1)  # shape: (batch)
        sol_infeasible_rate = infeasible.sum() / (batch_size * env.pomo_size * aug_factor * sample_size)
        ins_infeasible_rate = 1. - aug_feasible.sum() / batch_size
        if self.tester_params["fsb_dist_only"]:
            # get feasible results from pomo
            reward_masked = aug_reward.masked_fill(infeasible, -1e10)
            fsb_no_aug = reward_masked[0,:,:].max(dim=1, keepdim=True).values
            no_aug_score_mean = -fsb_no_aug[no_aug_feasible.bool()].mean()
            # shape: (augmentation, batch)
            fsb_aug = reward_masked.max(dim=0).values.max(dim=-1).values
            tour_path = self.tester_params["output_best_tour_path"]
            if tour_path is not None:
                assert env.pomo_size == 1, NotImplementedError("Only support env.pomo_size == 1!")
                best_idx = reward_masked.max(dim=0)[1].squeeze(-1)
                route_list = env.selected_node_list.reshape(aug_factor * sample_size, batch_size, -1)
                route_list = route_list[best_idx, torch.arange(batch_size)]
                test_data = test_data + tuple(route_list.unsqueeze(0))
                if not os.path.exists(tour_path): # first
                    write_pkl_file(tour_path, test_data)
                else:
                    add_data_to_pkl(tour_path, test_data, env.problem_size)
                    updated_data = read_pkl_file(tour_path, env.problem_size)
                    print("data size: ", updated_data[0].size())

            aug_score_mean = -fsb_aug[aug_feasible.bool()].mean()
            no_aug_score, aug_score = -fsb_no_aug, -fsb_aug
        else:
            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
            no_aug_score_mean = no_aug_score.mean()

            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
            aug_score_mean = aug_score.mean()

        return no_aug_score_mean.item(), aug_score_mean.item(), no_aug_score, aug_score, sol_infeasible_rate, ins_infeasible_rate, no_aug_feasible, aug_feasible

    def _solve_tsptwlib(self, path, env_class):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # print(">> Load data from {}".format(path))
        node_xy, service_time, tw_start, tw_end = torch.Tensor(data[0]).unsqueeze(0), torch.Tensor(data[1]).unsqueeze(0), torch.Tensor(data[2]).unsqueeze(0),  torch.Tensor(data[3]).unsqueeze(0)
        # loc_scaler = node_xy.max().item()
        # print(node_xy.max().item() , loc_scaler)

        loc_scaler = node_xy.max()
        node_xy = node_xy /loc_scaler
        env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': loc_scaler,
                      'device': self.device, "tw_duration":None, "reverse":None, "tw_type": None, "random_delta_t": 0, }
        env = env_class(**env_params)
        # print(node_xy.size())
        tw_end = tw_end/loc_scaler
        tw_start = tw_start/loc_scaler

        data = (node_xy, service_time, tw_start, tw_end)
        no_aug_score, aug_score, _, _, sol_infeasible_rate, ins_infeasible_rate, _, _ = self._test_one_batch(data, env)
        # no_aug_score = torch.Tensor([no_aug_score])
        # aug_score = torch.Tensor([aug_score])
        # no_aug_score = torch.round(torch.Tensor([no_aug_score]) ).long()
        # aug_score = torch.round(torch.Tensor([aug_score]) ).long()
        no_aug_score = torch.round(torch.Tensor([no_aug_score]) * loc_scaler).long()
        aug_score = torch.round(torch.Tensor([aug_score]) * loc_scaler).long()

        print(">> Finish solving {} -> no_aug: {} aug: {} ins_infsb {} sol_infsb {}".format(path, no_aug_score.item(), aug_score.item(),ins_infeasible_rate.item(), sol_infeasible_rate.item()))

        return no_aug_score.item(), aug_score.item(),ins_infeasible_rate.item(), sol_infeasible_rate.item()

    def _solve_tspdllib(self, path, env_class):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # print(">> Load data from {}".format(path))
        node_xy, node_demand, node_draft_limit = torch.Tensor(data[0]).unsqueeze(0), torch.Tensor(data[1]).unsqueeze(0), torch.Tensor(data[2]).unsqueeze(0)
        # loc_scaler = node_xy.max().item()
        # print(node_xy.max().item() , loc_scaler)
        # node_xy = node_xy /loc_scaler
        loc_scaler = 1

        env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': loc_scaler,
                      'device': self.device, "dl_percent":None, "reverse":None, "original_lib_xy": node_xy}
        env = env_class(**env_params)
        # print(node_xy.size())

        min_xy = node_xy.min(dim=1)[0].unsqueeze(1)
        max_xy = node_xy.max(dim=1)[0].unsqueeze(1)
        node_xy = (node_xy - min_xy) / (max_xy - min_xy)

        # node_xy = (node_xy - node_xy.min()) / (node_xy.max() - node_xy.min())
        # print(node_xy.max())

        data = (node_xy, node_demand, node_draft_limit)
        no_aug_score, aug_score, _, _, sol_infeasible_rate, ins_infeasible_rate, _, _ = self._test_one_batch(data, env)
        # no_aug_score = torch.Tensor([no_aug_score])
        # aug_score = torch.Tensor([aug_score])
        # no_aug_score = torch.round(torch.Tensor([no_aug_score]) ).long()
        # aug_score = torch.round(torch.Tensor([aug_score]) ).long()
        no_aug_score = torch.round(torch.Tensor([no_aug_score]) * loc_scaler).long()
        aug_score = torch.round(torch.Tensor([aug_score]) * loc_scaler).long()

        print(">> Finish solving {} -> no_aug: {} aug: {} ins_infsb {} sol_infsb {}".format(path, no_aug_score.item(), aug_score.item(),ins_infeasible_rate.item(), sol_infeasible_rate.item()))

        return no_aug_score.item(), aug_score.item(),ins_infeasible_rate.item(), sol_infeasible_rate.item()
