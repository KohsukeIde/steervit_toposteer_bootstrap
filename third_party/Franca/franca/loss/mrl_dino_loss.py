# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class MRLDINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
        relative_importance: Optional[List[float]] = None,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        self.n_iter = 10
        self.relative_importance = relative_importance

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        if isinstance(teacher_output, tuple):
            return tuple(F.softmax((t - self.center) / teacher_temp, dim=-1) for t in teacher_output)
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_global_crops_teacher, n_iterations=3):
        results = []
        for t_out in teacher_output:
            t_out = t_out.float()
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            Q = torch.exp(t_out / teacher_temp).t()
            B = Q.shape[1] * world_size  # number of samples to assign
            K = Q.shape[0]  # how many prototypes

            for it in range(n_iterations):
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                if dist.is_initialized():
                    dist.all_reduce(sum_of_rows)
                Q /= sum_of_rows
                Q /= K
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B

            Q *= B
            results.append(Q.t())

        return tuple(results)

    def forward(
        self,
        student_output_list,
        teacher_out_softmaxed_centered_list,
        n_crops,
        teacher_global=False,
    ):
        if not teacher_global:
            total_loss = 0
            # Iterate over each granularity level
            for student_outputs, teacher_outputs in zip(student_output_list, teacher_out_softmaxed_centered_list):
                student_feat = student_outputs.chunk(n_crops[0])
                teacher_feat = teacher_outputs.view(n_crops[1], -1, teacher_outputs.shape[-1])
                for s in student_feat:
                    lsm = F.log_softmax(s / self.student_temp, dim=-1)
                    for t in teacher_feat:
                        # print(t.shape, lsm.shape)
                        loss = torch.sum(t * lsm, dim=-1)
                        total_loss -= loss.mean()

            return total_loss

        elif teacher_global:
            total_loss = 0
            # Iterate over each granularity level
            for student_outputs, teacher_outputs in zip(student_output_list, teacher_out_softmaxed_centered_list):

                teacher_outputs = teacher_outputs.view(n_crops, -1, teacher_outputs.shape[-1])
                teacher_feats = [teacher_outputs.flatten(0, 1)]
                student_feats = [student_outputs]
                for s in student_feats:
                    lsm = F.log_softmax(s / self.student_temp, dim=-1)
                    for t in teacher_feats:
                        loss = torch.sum(t * lsm, dim=-1)
                        total_loss -= loss.mean()

            return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        if isinstance(teacher_output, tuple):
            # If teacher_output is a tuple, update center using the first element
            self.reduce_center_update(teacher_output[0])
        else:
            self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True
