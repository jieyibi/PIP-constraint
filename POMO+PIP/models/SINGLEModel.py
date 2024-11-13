import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['SINGLEModel']


class SINGLEModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']

        self.encoder = SINGLE_Encoder(**model_params)
        self.decoder = SINGLE_Decoder(**model_params)

        self.encoded_nodes = None

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        if not self.problem.startswith('TSP'):
            depot_xy = reset_state.depot_xy
            # shape: (batch, 1, 2)
            node_demand = reset_state.node_demand
        else:
            depot_xy = None

        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)

        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            feature = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
            # shape: (batch, problem, 3)
        elif self.problem in ["TSPTW"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            tw_start = node_tw_start[:, :, None]
            tw_end = node_tw_end[:, :, None]
            # _, problem_size = node_tw_end.size()
            if self.model_params["tw_normalize"]:
                tw_end_max = node_tw_end[:, :1, None]
                tw_start = tw_start / tw_end_max
                tw_end = tw_end / tw_end_max
            feature =  torch.cat((node_xy, tw_start, tw_end), dim=2)
            # shape: (batch, problem, 4)
        elif self.problem in ['TSPDL']:
            node_demand = reset_state.node_demand
            node_draft_limit = reset_state.node_draft_limit
            feature = torch.cat((node_xy, node_demand[:, :, None], node_draft_limit[:, :, None]), dim=2)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            feature = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
            # shape: (batch, problem, 5)
        else:
            raise NotImplementedError

        self.encoded_nodes = self.encoder(depot_xy, feature)
        # shape: (batch, problem(+1), embedding)

        self.decoder.set_kv(self.encoded_nodes)
        if self.model_params["pip_decoder"] and self.model_params["W_kv_sl"]:
            self.decoder.set_kv_sl(self.encoded_nodes)

        return self.encoded_nodes, feature

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None, pomo = False, use_predicted_PI_mask=False, no_select_prob=False, no_sigmoid=False, tw_end=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            prob = torch.ones(size=(batch_size, pomo_size))
            if self.model_params["pip_decoder"]:
                probs_sl = torch.ones(size=(batch_size, pomo_size))
            # shape: (batch, pomo, problem_size+1)
        elif pomo and state.selected_count == 1 and pomo_size > 1:  # Second Move, POMO
            selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size))
            if self.model_params["pip_decoder"]:
                probs_sl = torch.ones(size=(batch_size, pomo_size))
        else: # Sample from the action distribution
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            attr = self.get_context(state, tw_end)
            ninf_mask = state.ninf_mask
            if self.model_params["pip_decoder"]: # auxiliary decoder to predict whether nodes are feasible
                if not no_select_prob:
                    probs, probs_sl = self.decoder(encoded_last_node, attr, ninf_mask=ninf_mask, use_predicted_PI_mask = use_predicted_PI_mask, no_sigmoid=no_sigmoid)
                else:
                    probs_sl = self.decoder(encoded_last_node, attr, ninf_mask=ninf_mask, use_predicted_PI_mask=use_predicted_PI_mask, no_select_prob=no_select_prob, no_sigmoid=no_sigmoid)
                    return probs_sl
            else:
                probs = self.decoder(encoded_last_node, attr, ninf_mask=ninf_mask)

            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            # torch.save(probs,"prob.pt")
                            print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = selected
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        if self.model_params['pip_decoder']:
            return selected, [prob, probs_sl]
        return selected, prob

    def get_context(self, state, tw_end):
        if self.problem in ["CVRP"]:
            attr = state.load[:, :, None]
        elif self.problem in ["VRPB", 'TSPDL']:
            attr = state.load[:, :, None]  # shape: (batch, pomo, 1)
        elif self.problem in ["TSPTW"]:
            attr = state.current_time[:, :, None]  # shape: (batch, pomo, 1)
            if self.model_params["tw_normalize"]:
                attr = attr / tw_end[:, 0][:, None, None]
        elif self.problem in ["OVRP", "OVRPB"]:
            attr = torch.cat((state.load[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPTW", "VRPBTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPL", "VRPBL"]:
            attr = torch.cat((state.load[:, :, None], state.length[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPLTW", "VRPBLTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None]), dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPL", "OVRPBL"]:
            attr = torch.cat((state.load[:, :, None], state.length[:, :, None], state.open[:, :, None]),  dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPTW", "OVRPBTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None],
                              state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 4)
        else:
            raise NotImplementedError

        return attr

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class SINGLE_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        if not self.problem.startswith("TSP"):
            self.embedding_depot = nn.Linear(2, embedding_dim)
        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            self.embedding_node = nn.Linear(3, embedding_dim)
        elif self.problem in ["TSPTW", "TSPDL"]:
            self.embedding_node = nn.Linear(4, embedding_dim)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            self.embedding_node = nn.Linear(5, embedding_dim)
        else:
            raise NotImplementedError
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2) if self.problem is CVRP variants
        # node_xy_demand_tw.shape: (batch, problem, 3/4/5) - based on self.problem
        if depot_xy is not None:
            embedded_depot = self.embedding_depot(depot_xy)
            # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand_tw)
        # shape: (batch, problem, embedding)

        if depot_xy is not None:
            out = torch.cat((embedded_depot, embedded_node), dim=1)
            # shape: (batch, problem+1, embedding)
        else:
            out = embedded_node
            # shape: (batch, problem, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        """
        Two implementations:
            norm_last: the original implementation of AM/POMO: MHA -> Add & Norm -> FFN/MOE -> Add & Norm
            norm_first: the convention in NLP: Norm -> MHA -> Add -> Norm -> FFN/MOE -> Add
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)  # (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)  # (batch, problem, EMBEDDING_DIM)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3

########################################
# DECODER
########################################

class SINGLE_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        if self.problem == "CVRP":
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPB", "TSPTW", "TSPDL"]:
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRP", "OVRPB", "VRPTW", "VRPBTW", "VRPL", "VRPBL"]:
            attr_num = 3 if self.model_params["extra_feature"] else 2
            self.Wq_last = nn.Linear(embedding_dim + attr_num, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + attr_num, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPLTW", "VRPBLTW", "OVRPL", "OVRPBL", "OVRPTW", "OVRPBTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
            if self.model_params["pip_decoder"]:
                self.Wq_last_sl = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        else:
            raise NotImplementedError

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        if self.model_params["pip_decoder"] and self.model_params['W_kv_sl']:
            self.Wk_sl = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wv_sl = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.k_sl = None  # saved key, for multi-head_attention
            self.v_sl = None  # saved value, for multi-head_attention
            self.single_head_key_sl = None

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        if self.model_params["pip_decoder"] and self.model_params['W_out_sl']:
            self.multi_head_combine_sl = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head_attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head_attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_kv_sl(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        if self.model_params['detach_from_encoder']:
            self.k_sl = reshape_by_heads(self.Wk_sl(encoded_nodes.detach()), head_num=head_num)
            self.v_sl = reshape_by_heads(self.Wv_sl(encoded_nodes.detach()), head_num=head_num)
            # shape: (batch, head_num, problem+1, qkv_dim)
            self.single_head_key_sl = encoded_nodes.transpose(1, 2).detach()
            # shape: (batch, embedding, problem+1)
        else:
            self.k_sl = reshape_by_heads(self.Wk_sl(encoded_nodes), head_num=head_num)
            self.v_sl = reshape_by_heads(self.Wv_sl(encoded_nodes), head_num=head_num)
            self.single_head_key_sl = encoded_nodes.transpose(1, 2)


    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, attr, ninf_mask, use_predicted_PI_mask=False, no_select_prob = False, no_sigmoid=False):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, 1~4)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        if self.model_params['pip_decoder']:
            if isinstance(use_predicted_PI_mask, bool):
                if self.model_params['detach_from_encoder']:
                    q_last_sl = reshape_by_heads(self.Wq_last_sl(input_cat.detach()), head_num=head_num)
                else:
                    q_last_sl = reshape_by_heads(self.Wq_last_sl(input_cat), head_num=head_num)

                ninf_mask_sl = ninf_mask if self.model_params['use_ninf_mask_in_sl_MHA'] else None
                if self.model_params['W_kv_sl']:
                    out_concat_sl = multi_head_attention(q_last_sl, self.k_sl, self.v_sl, rank3_ninf_mask=ninf_mask_sl)
                else:
                    out_concat_sl = multi_head_attention(q_last_sl, self.k, self.v, rank3_ninf_mask=ninf_mask_sl)
                if self.model_params['W_out_sl']:
                    mh_atten_out_sl = self.multi_head_combine_sl(out_concat_sl)
                else:
                    mh_atten_out_sl = self.multi_head_combine(out_concat_sl)

                if self.model_params['W_kv_sl']:
                    score_sl = torch.matmul(mh_atten_out_sl, self.single_head_key_sl)
                else:
                    score_sl = torch.matmul(mh_atten_out_sl, self.single_head_key)

                probs_sl = score_sl if no_sigmoid else torch.sigmoid(score_sl)
                if no_select_prob:
                    return probs_sl
            else:
                probs_sl = use_predicted_PI_mask

            if not isinstance(use_predicted_PI_mask, bool) or use_predicted_PI_mask:
                ninf_mask0 =  ninf_mask.clone()
                if isinstance(probs_sl, list):
                    for i in range(len(probs_sl)):
                        ninf_mask = ninf_mask +torch.where(probs_sl[i] > self.model_params["decision_boundary"], float('-inf'),ninf_mask) if not no_sigmoid \
                            else torch.where(torch.sigmoid(probs_sl[i]) > self.model_params["decision_boundary"], float('-inf'), ninf_mask)
                else:
                    ninf_mask = torch.where(probs_sl>self.model_params["decision_boundary"], float('-inf'), ninf_mask) if not no_sigmoid \
                        else torch.where(torch.sigmoid(probs_sl)>self.model_params["decision_boundary"], float('-inf'), ninf_mask)
                all_infsb = ((ninf_mask == float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1, -1, self.single_head_key.size(-1))
                ninf_mask = torch.where(all_infsb, ninf_mask0, ninf_mask)
        # shape: (batch, head_num, pomo, qkv_dim)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)
        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        if self.model_params['pip_decoder']:
            return probs, probs_sl

        return probs

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

class FC(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.W1 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input):
        # input.shape: (batch, problem, embedding)
        return F.relu(self.W1(input))