# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, init_out, rescale=True, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)
        stitched_init = torch.cat(init_out, 0).detach()
        stitched_init = torch.cat(init_out, 0).detach()
        #print(stitched.device)
        #print(stitched_init.device)
        if rescale:
            expert_norm = stitched.norm(p=2, dim=1, keepdim=True)  # [8, 1]
            init_norm = stitched_init.norm(p=2, dim=1, keepdim=True)  # [8, 1]
            #print(init_norm.device)
            stitched = stitched / (expert_norm + 1e-8)
            stitched = stitched * init_norm
            stitched_hard = stitched.clone()
            # print(stitched.shape)
        # softmax값 곱해주는 곳
        # breakpoint()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())

        # breakpoint()
        # batch 개수 (예: 4)
        num_batches = self._batch_index.max().item() + 1  # 4

        # 각 배치별로 모아서 리스트에 저장
        
        grouped = [stitched[self._batch_index == i] for i in range(num_batches)]
        expert_index_clone = [self._expert_index[self._batch_index == i] for i in range(num_batches)]
        expert_prob_clone = [self._nonzero_gates[self._batch_index == i] for i in range(num_batches)]
        expert_index_hard_clone = [stitched_hard[self._batch_index == i] for i in range(num_batches)]
        
        # 모두 같은 길이(여기서는 2)라고 가정 → stack 가능
        try:
            token_wise = torch.stack(grouped, dim=0)  # shape: [4, 2, 768]
        except RuntimeError as e:
            print(f"batch_index: {self._batch_index}") 
            print(f"grouped: {grouped.shape}")
        token_wise_index = torch.stack(expert_index_clone, dim=0)  # shape: [4, 2, 1]
        token_wise_prob = torch.stack(expert_prob_clone, dim=0)  # shape: [4, 2, 1]
        
        token_wise_hard = torch.stack(expert_index_hard_clone, dim=0)  # shape: [4, 2, 768]
        return combined, token_wise, token_wise_index, token_wise_prob, token_wise_hard

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        self.gate_temp = nn.Parameter(torch.tensor(1.0))

        # instantiate experts
        # self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        nn.init.normal_(self.w_gate, mean=0.0, std=1.0) # 9-1 수정 ################### 평균 0, 표준편차 1.0인 가우시안 분포에서 샘플한 값들로 초기화화

        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-5):
        clean_logits = x @ self.w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-5)  # normalization
        # 값이 0이 되어서 선택 안되는 현상 방지
        top_k_gates = top_k_gates.clamp(min=1e-5)
        
        zeros = torch.zeros_like(logits, requires_grad=True)
        # gates = zeros.scatter(1, top_k_indices, top_k_gates)
        gates_discrete = zeros.scatter(1, top_k_indices, top_k_gates)
        gates = logits + (gates_discrete - logits).detach()

        

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, clean_logits

    def forward(self, x, expert_embeddings, init_embeddings, importance_loss_weight=0.0, load_loss_weight=0.0, rescale=False,
            # lambda 인자들 (기본값 0으로 변경)
            lambda_cos_sim=0.0,
            lambda_variance=0.0,
            lambda_orthogonality=0.0,
            lambda_inv_sq_dist=0.0,
            lambda_svd=0.0,
            lambda_rbf=0.0,
            lambda_exp_dist=0.0,
            # Specialization loss lambda 
            lambda_specialization=0.0,
            timesteps=None,             # specialization loss 용
            num_train_timesteps=1000,   # specialization loss 용
            specialization_sigma=1.0,   # specialization loss 용
            # 새로운 손실 함수 파라미터
            rbf_gamma=1.0,
            exp_dist_alpha=1.0,
            svd_epsilon=1e-8,
            inv_sq_dist_epsilon=1e-8
           ):
        # self.training은 nn.Module의 속성이라 따로 선언 안해줘도됨
        # self.training은 nn.Module의 속성
        self.gates, load, clean_logits = self.noisy_top_k_gating(x, self.training) # clean_logits 받기 추가

        # 1. Load balancing loss 계산
        importance = self.gates.sum(0)
        load_balancing_loss = self.cv_squared(importance) * importance_loss_weight + self.cv_squared(load) * load_loss_weight

        # 2. 전문가 임베딩 정규화 손실 계산
        total_reg_loss = torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

        # 각 lambda 값에 따라 해당 정규화 손실 계산 및 추가
        if lambda_cos_sim > 0:
            cosine_reg_loss = calculate_cosine_similarity_loss(expert_embeddings)
            total_reg_loss = total_reg_loss + lambda_cos_sim * cosine_reg_loss

        if lambda_variance > 0:
            variance_loss = calculate_variance_maximization_loss(expert_embeddings)
            total_reg_loss = total_reg_loss + lambda_variance * variance_loss

        if lambda_orthogonality > 0:
            ortho_loss = calculate_orthogonality_loss(expert_embeddings)
            total_reg_loss = total_reg_loss + lambda_orthogonality * ortho_loss

        if lambda_inv_sq_dist > 0:
            inv_sq_dist_loss = calculate_inverse_sq_distance_loss(expert_embeddings, epsilon=inv_sq_dist_epsilon)
            total_reg_loss = total_reg_loss + lambda_inv_sq_dist * inv_sq_dist_loss

        if lambda_svd > 0:
            svd_loss = calculate_svd_log_sum_loss(expert_embeddings, epsilon=svd_epsilon)
            total_reg_loss = total_reg_loss + lambda_svd * svd_loss

        if lambda_rbf > 0:
            rbf_loss = calculate_rbf_kernel_loss(expert_embeddings, gamma=rbf_gamma)
            total_reg_loss = total_reg_loss + lambda_rbf * rbf_loss

        if lambda_exp_dist > 0:
            exp_dist_loss = calculate_exp_min_distance_loss(expert_embeddings, alpha=exp_dist_alpha)
            total_reg_loss = total_reg_loss + lambda_exp_dist * exp_dist_loss

        # 3. Specialization loss 계산
        if lambda_specialization > 0 and timesteps is not None:
            spec_loss = specialization_loss_gaussian(
                clean_logits,               # Softmax 적용 전 로짓 사용
                timesteps,                  # 전달받은 원본 타임스텝
                num_train_timesteps,        # 전달받은 총 타임스텝 수
                sigma=specialization_sigma  # 시그마 값
            )
            total_reg_loss = total_reg_loss + lambda_specialization * spec_loss # 정규화 손실에 추가


        # 전체 손실 = 로드 밸런싱 손실 + 모든 정규화 손실 합계
        total_combined_loss = load_balancing_loss + total_reg_loss

        # Sparse Dispatcher 준비 ... (이하 로직은 기존과 유사하게 유지)
        dispatcher = SparseDispatcher(self.num_experts, self.gates) # 'gates' 변수명 확인 필요 (기존 코드에서 self.gates 사용했다면 변경)
        # --- dispatcher에 k값 전달 (필요시 추가) ---
        # dispatcher.k = self.k

        # expert_inputs 계산 등 dispatcher 사용 부분
        expert_inputs = dispatcher.dispatch(x) # dispatcher 입력이 x가 맞는지 확인 필요
        expert_sizes = [tensor.shape[0] for tensor in expert_inputs if tensor.numel() > 0] # 빈 텐서 제외

        expert_outputs = []
        init_outputs = []
        for i in range(self.num_experts):
            # expert_sizes[i]가 0보다 클 때만 리스트에 추가
            if i < len(expert_sizes) and expert_sizes[i] > 0:
                expert_outputs.extend([expert_embeddings[i].unsqueeze(0)] * expert_sizes[i])
                init_outputs.extend([init_embeddings[i].unsqueeze(0)] * expert_sizes[i])
            elif i >= len(expert_sizes): # expert_sizes 리스트 길이보다 i가 크거나 같으면 빈 리스트 처리 가정 (오류 방지)
                pass # 또는 로깅/에러 처리

        # expert_outputs나 init_outputs가 비어있는 경우 combine 호출 전 처리
        if not expert_outputs: # 하나라도 비면 둘 다 빈 것으로 간주하거나 오류 처리
            # 예시: 빈 결과 반환 또는 기본값 처리
            y = torch.zeros(x.shape[0], self.output_size, device=x.device, dtype=x.dtype) # 크기 주의
            token_wise = torch.zeros(x.shape[0], self.k, self.output_size, device=x.device, dtype=x.dtype) # self.k 필요
            token_wise_index = torch.zeros(x.shape[0], self.k, 1, device=x.device, dtype=torch.long)
            token_wise_prob = torch.zeros(x.shape[0], self.k, 1, device=x.device, dtype=x.dtype)
        else:
            # combine 메소드 시그니처 확인 (rescale 인자 등)
            y, token_wise, token_wise_index, token_wise_prob, token_wise_hard = dispatcher.combine(expert_out=expert_outputs, init_out=init_outputs, rescale=rescale) # rescale 필요 여부 확인

        return y, total_combined_loss, token_wise, token_wise_index, token_wise_prob, token_wise_hard # total_loss -> total_combined_loss
    


################################################################
# Regularization Term
################################################################
import itertools
import torch.nn.functional as F

def calculate_cosine_similarity_loss(expert_embeddings: torch.Tensor) -> torch.Tensor:
    num_experts, embedding_dim = expert_embeddings.shape
    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

    total_cosine_similarity = 0.0
    num_pairs = 0
    for i, j in itertools.combinations(range(num_experts), 2):
        emb1 = expert_embeddings[i]
        emb2 = expert_embeddings[j]

        cosine_sim = torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2) + 1e-8)
        total_cosine_similarity = total_cosine_similarity + cosine_sim
        num_pairs += 1
    return total_cosine_similarity

def specialization_loss_gaussian(logits, timesteps, num_train_timesteps, sigma=1.0):
    if logits is None or timesteps is None:
        return torch.tensor(0.0, device=logits.device if logits is not None else 'cpu')
    
    K = logits.size(1)                                      # num_experts
    if K <= 1:                                              # embedding 한개면 의미 없음
        return torch.tensor(0.0, device=logits.device)

    bin_size = num_train_timesteps / K


    centers = (timesteps / bin_size).floor().clamp(max=K-1) # .floor() 추가, clamp max 수정
    j_idx = torch.arange(K, device=logits.device).float()   # 모든 전문가 인덱스 (0 ~ K-1)
    distances = j_idx[None, :] - centers.float()[:, None]   # shape: [batch, K]

    # 각 sample마다 자신의 center를 기준으로 gaussian distribution 생성
    unnorm = torch.exp(-0.5 * (distances / sigma)**2)       # shape: [batch, K]
    p_target = unnorm / unnorm.sum(dim=-1, keepdim=True)    # shape: [batch, K]
    log_p = F.log_softmax(logits, dim=-1)                   # shape: [batch, K]
    kl_loss = F.kl_div(log_p, p_target, reduction="batchmean", log_target=False) # log_target=False 중요

    return kl_loss

def calculate_variance_maximization_loss(expert_embeddings: torch.Tensor) -> torch.Tensor:
    num_experts, embedding_dim = expert_embeddings.shape
    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

    mu_E = torch.mean(expert_embeddings, dim=0)
    variance = torch.mean(torch.sum((expert_embeddings - mu_E)**2, dim=1))
    return -variance # 분산 최대화 = 음수 분산 최소화

def calculate_orthogonality_loss(expert_embeddings: torch.Tensor) -> torch.Tensor:
    num_experts, embedding_dim = expert_embeddings.shape
    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

    total_dot_product_sq = 0.0
    num_pairs = 0
    # 정규화된 임베딩 사용 권장 (옵션)
    # expert_embeddings = F.normalize(expert_embeddings, p=2, dim=1)
    for i, j in itertools.combinations(range(num_experts), 2):
        dot_product = torch.dot(expert_embeddings[i], expert_embeddings[j])
        total_dot_product_sq = total_dot_product_sq + dot_product**2
        num_pairs += 1

    return total_dot_product_sq / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

def calculate_inverse_sq_distance_loss(expert_embeddings: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    num_experts, embedding_dim = expert_embeddings.shape
    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

    total_inv_sq_dist = 0.0
    num_pairs = 0
    for i, j in itertools.combinations(range(num_experts), 2):
        dist_sq = torch.sum((expert_embeddings[i] - expert_embeddings[j])**2)
        total_inv_sq_dist = total_inv_sq_dist + 1.0 / (dist_sq + epsilon)
        num_pairs += 1

    return total_inv_sq_dist / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

def calculate_svd_log_sum_loss(expert_embeddings: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    num_experts, embedding_dim = expert_embeddings.shape
    if num_experts <= 1 or embedding_dim < num_experts: # SVD를 위해 num_experts <= embedding_dim 필요
        return torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

    mu_E = torch.mean(expert_embeddings, dim=0, keepdim=True)
    centered_embeddings = expert_embeddings - mu_E

    try:
        # _, sigma, _ = torch.linalg.svd(centered_embeddings, full_matrices=False)
        sigma = torch.linalg.svdvals(centered_embeddings) # 특이값만 계산하는 것이 더 효율적
    except torch.linalg.LinAlgError:
         print("Warning: SVD computation failed. Returning 0 for SVD loss.")
         return torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)


    log_sigma_sum = torch.sum(torch.log(sigma + epsilon))
    return -log_sigma_sum

def calculate_rbf_kernel_loss(expert_embeddings: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    num_experts, embedding_dim = expert_embeddings.shape
    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

    total_rbf_val = 0.0
    num_pairs = 0
    for i, j in itertools.combinations(range(num_experts), 2):
        dist_sq = torch.sum((expert_embeddings[i] - expert_embeddings[j])**2)
        rbf_val = torch.exp(-gamma * dist_sq)
        total_rbf_val = total_rbf_val + rbf_val
        num_pairs += 1

    return total_rbf_val / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

def calculate_exp_min_distance_loss(expert_embeddings: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    num_experts, embedding_dim = expert_embeddings.shape
    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)

    total_exp_dist_penalty = 0.0
    num_pairs = 0
    for i, j in itertools.combinations(range(num_experts), 2):
        dist = torch.norm(expert_embeddings[i] - expert_embeddings[j], p=2)
        exp_penalty = torch.exp(-alpha * dist)
        total_exp_dist_penalty = total_exp_dist_penalty + exp_penalty
        num_pairs += 1

    return total_exp_dist_penalty / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=expert_embeddings.device, dtype=expert_embeddings.dtype)