import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class DistillationLoss(nn.Module):
    """
    Cosine-similarity knowledge distillation loss against precomputed teacher features.
    """

    def __init__(
        self,
        student_feature_dim: int,
        teacher_feature_dim: int,
        student_frame_rate: float,
        teacher_frame_rate: float,
    ):
        """
        Initialise the distillation loss.

        :param student_feature_dim: Dimensionality of the student features passed to `forward`.
        :type student_feature_dim: int
        :param teacher_feature_dim: Dimensionality of the teacher's features.
        :type teacher_feature_dim: int
        :param student_frame_rate: Frame rate of the student features.
        :type student_frame_rate: float
        :param teacher_frame_rate: Frame rate of the teacher's features.
        :type teacher_frame_rate: float
        """

        super().__init__()

        self.student_frame_rate = student_frame_rate
        self.teacher_frame_rate = teacher_frame_rate

        self.frame_rate_ratio = max(1, math.ceil(teacher_frame_rate / student_frame_rate))
        self.projector = nn.Linear(student_feature_dim, self.frame_rate_ratio * teacher_feature_dim)

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the distillation loss between the student's features and the teacher's.

        :param student_feature: Student features with shape [B, T, student_feature_dim].
        :type student_feature: torch.Tensor
        :param teacher_feature: Precomputed teacher features with shape [B, S, teacher_feature_dim].
        :type teacher_feature: torch.Tensor

        :returns: Scalar loss in [0, 2]; minimising it aligns each projected student slot's
            direction with the corresponding teacher feature.
        :rtype: torch.Tensor
        """

        teacher_feature = teacher_feature.to(student_feature.dtype)

        num_student_frames = student_feature.shape[1]
        num_teacher_frames = teacher_feature.shape[1]
        num_aligned_slots = min(
            num_student_frames * self.frame_rate_ratio,
            int(num_teacher_frames * self.frame_rate_ratio * self.student_frame_rate / self.teacher_frame_rate),
        )

        projected_student_feature = self.projector(student_feature).unflatten(-1, (self.frame_rate_ratio, -1))
        projected_student_feature = projected_student_feature.flatten(1, 2)[:, :num_aligned_slots]
        teacher_feature = F.interpolate(teacher_feature.swapaxes(1, 2), size=num_aligned_slots, mode="linear").swapaxes(
            1, 2
        )

        return (1 - F.cosine_similarity(projected_student_feature, teacher_feature, dim=-1)).mean()
