# SPDX-License-Identifier: Apache-2.0
"""Causal DMD (Self-Forcing) pipeline with a Fun-Control style control video.

The transformer is a Self-Forcing causal Wan 1.3B whose patch embedding was
widened to 48 input channels (noisy 16 | control 16 | zero pad 16) by merging
the Wan2.1-Fun-1.3B-Control delta. The control video is VAE-encoded once and
the matching latent frames are concatenated to every causal block the DiT
sees (denoising calls and the clean-context KV refresh alike) via a forward
wrapper installed before the denoising stage.
"""
import functools
from dataclasses import dataclass

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.wan.pipeline_config import SelfForcingWanT2V480PConfig
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.basic.wan.stages.causal_denoising import CausalDMDDenosingStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage, InputValidationStage,
                                        LatentPreparationStage, TextEncodingStage, VideoVAEEncodingStage)
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


@dataclass
class SelfForcingControlConfig(SelfForcingWanT2V480PConfig):
    """Self-Forcing config that also loads the VAE encoder for the control video."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


class ControlLatentHookStage(PipelineStage):
    """Installs a transformer.forward wrapper that concatenates the control latents."""

    def __init__(self, transformer) -> None:
        self.transformer = transformer

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult()

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult()

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        tr = self.transformer
        control = batch.video_latent  # [B, 16, T_lat, h, w]
        assert control is not None and control.ndim == 5, "control latents missing"
        tr._control_latent = control
        if getattr(tr, "_control_hooked", False):
            return batch
        orig_forward = tr.forward

        @functools.wraps(orig_forward)
        def forward(hidden_states, *args, start_frame: int = 0, **kwargs):
            if hidden_states.shape[1] == tr.in_channels:
                return orig_forward(hidden_states, *args, start_frame=start_frame, **kwargs)
            n = hidden_states.shape[2]
            ctrl = tr._control_latent[:, :, start_frame:start_frame + n].to(hidden_states.device, hidden_states.dtype)
            assert ctrl.shape[2] == n, (ctrl.shape, n, start_frame)
            x = torch.cat([hidden_states, ctrl, torch.zeros_like(ctrl)], dim=1)
            return orig_forward(x, *args, start_frame=start_frame, **kwargs)

        tr.forward = forward
        tr._control_hooked = True
        return batch


class WanCausalControlDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer", None)))
        self.add_stage(stage_name="video_latent_preparation_stage",
                       stage=VideoVAEEncodingStage(vae=self.get_module("vae")))
        self.add_stage(stage_name="control_hook_stage",
                       stage=ControlLatentHookStage(transformer=self.get_module("transformer")))
        self.add_stage(stage_name="denoising_stage",
                       stage=CausalDMDDenosingStage(transformer=self.get_module("transformer"),
                                                    transformer_2=self.get_module("transformer_2", None),
                                                    scheduler=self.get_module("scheduler")))
        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanCausalControlDMDPipeline
