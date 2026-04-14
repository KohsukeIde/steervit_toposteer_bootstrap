import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    from xformers.ops import cross_entropy

    def lossfunc(t, s, temp):
        s = s.float()
        t = t.float()
        if s.ndim == 2:
            return -cross_entropy(s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True).squeeze(0)
        elif s.ndim == 3:
            return -cross_entropy(s, t, temp, bw_inplace=True)

except ImportError:

    def lossfunc(t, s, temp):
        return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


def compute_entropy(matrix):
    # Compute the entropy of the matrix
    matrix = matrix + 1e-12  # Add a small value to avoid log(0)
    entropy = -torch.sum(matrix * torch.log(matrix))
    return entropy


class MRLiBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None
        self.n_iter = 10

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        if isinstance(teacher_patch_tokens, tuple):
            return tuple(F.softmax((t - self.center) / teacher_temp, dim=-1) for t in teacher_patch_tokens)
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teach_out, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        result = []
        for teacher_output in teach_out:
            teacher_output = teacher_output.float()
            Q = torch.exp(teacher_output / teacher_temp).t()
            B = n_masked_patches_tensor
            dist.all_reduce(B)
            K = Q.shape[0]

            # make the matrix sums to 1
            sum_Q = torch.sum(Q)
            if dist.is_initialized():
                dist.all_reduce(sum_Q)
            Q /= sum_Q

            for it in range(n_iterations):
                # normalize each row: total weight per prototype must be 1/K
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                if dist.is_initialized():
                    dist.all_reduce(sum_of_rows)
                Q /= sum_of_rows
                Q /= K

                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B

            Q *= B  # the columns must sum to 1 so that Q is an assignment
            result.append(Q.t())

        return tuple(result)

    def forward(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        if isinstance(teacher_patch_tokens_masked, tuple):
            total_loss = 0
            for s, t in zip(student_patch_tokens_masked, teacher_patch_tokens_masked):
                loss = lossfunc(t, s, self.student_temp)
                if masks_weight is None:
                    masks_weight = (
                        (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                        .unsqueeze(-1)
                        .expand_as(student_masks_flat)[student_masks_flat]
                    )
                if n_masked_patches is not None:
                    loss = loss[:n_masked_patches]
                loss = loss * masks_weight
                total_loss -= loss.sum() / student_masks_flat.shape[0]
            return total_loss
        else:
            t = teacher_patch_tokens_masked
            s = student_patch_tokens_masked
            loss = lossfunc(t, s, self.student_temp)
            if masks_weight is None:
                masks_weight = (
                    (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                    .unsqueeze(-1)
                    .expand_as(student_masks_flat)[student_masks_flat]
                )
            if n_masked_patches is not None:
                loss = loss[:n_masked_patches]
            loss = loss * masks_weight
            return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        if isinstance(teacher_patch_tokens, tuple):
            self.reduce_center_update(teacher_patch_tokens[0])
        else:
            self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True
