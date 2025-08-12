"""
class AttentionControl and class AttentionStore are modified from
https://github.com/google/prompt-to-prompt/blob/main/prompt-to-prompt_stable.ipynb
https://github.com/google/prompt-to-prompt/blob/main/ptp_utils.py
"""

import abc
from typing import Literal, final, override

import positional_encodings.torch_encodings as pos_enc
import torch
from diffusers import UNet2DConditionModel
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

LOW_RESOURCE = False


PeType = Literal["absolute", "rotary"]


@final
class PositionalEncoding1D(nn.Module):
    """
    This class fixes CUDA/CPU device inconsistency from the library code.
    """

    def __init__(self, channels: int, device: torch.device = torch.device("cuda")):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = ((channels + 1) // 2) * 2
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    @override
    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = pos_enc.get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class ExECProcessor:
    def __init__(self, text_embeddings_dim: int = 768) -> None:
        self.pe = PositionalEncoding1D(text_embeddings_dim)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            pe = torch.zeros_like(encoder_hidden_states)
            # print("self")
        else:
            # <<< add positional encoding >>>
            pe = self.pe(encoder_hidden_states)
            # print("cross")
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

        key = attn.to_k(encoder_hidden_states + pe)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        # attn: 8 * res * res

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn.clone())

        return attn

    def between_steps(self):
        assert len(self.attention_store) == 0

        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_pe_processor(
    unet_or_ddp: UNet2DConditionModel | DistributedDataParallel,
    pe_type: PeType,
) -> tuple[UNet2DConditionModel, dict]:
    """
    Register ExECProcessor into give `unet`.
    """

    if isinstance(unet_or_ddp, DistributedDataParallel):
        unet = unet_or_ddp.module
    else:
        unet = unet_or_ddp

    if pe_type == "absolute":
        Processor = ExECProcessor
    else:
        raise NotImplementedError(f"Unrecognized positional encoding type '{pe_type}'")

    processors = {}
    for name in unet.attn_processors.keys():
        processors[name] = Processor(unet.config.cross_attention_dim)

    original_processors = unet.attn_processors.copy()
    unet.set_attn_processor(processors)

    return unet, original_processors


def register_attention_control(
    unet_model_or_ddp, controller, pe_type: PeType | None = None
):
    if isinstance(unet_model_or_ddp, DistributedDataParallel):
        unet_model = unet_model_or_ddp.module
    else:
        unet_model = unet_model_or_ddp
    if pe_type is not None:
        text_embeddings_dim = unet_model.config.cross_attention_dim
        print(f"Using text embeddings dim = {text_embeddings_dim}")
        abspe = PositionalEncoding1D(text_embeddings_dim)

    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
        ):
            is_cross = encoder_hidden_states is not None

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            if not is_cross:
                key = self.to_k(encoder_hidden_states)
            elif pe_type == "absolute":
                key = self.to_k(encoder_hidden_states + abspe(encoder_hidden_states))
            elif pe_type is None:
                key = self.to_k(encoder_hidden_states)
            else:
                raise NotImplementedError(
                    f"God unrecognized positional encoding type '{pe_type}'"
                )
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)
            # all drop out in diffusers are 0.0
            # so we here ignore dropout

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    assert controller is not None, "controller must be specified"

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    down_count = 0
    up_count = 0
    mid_count = 0

    cross_att_count = 0
    sub_nets = unet_model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            down_temp = register_recr(net[1], 0, "down")
            cross_att_count += down_temp
            down_count += down_temp
        elif "up" in net[0]:
            up_temp = register_recr(net[1], 0, "up")
            cross_att_count += up_temp
            up_count += up_temp
        elif "mid" in net[0]:
            mid_temp = register_recr(net[1], 0, "mid")
            cross_att_count += mid_temp
            mid_count += mid_temp

    controller.num_att_layers = cross_att_count


def get_cross_attn_map_from_unet(
    attention_store: AttentionStore, training_resolution: int
):
    # reses=[64, 32, 16, 8], poses=["down", "mid", "up"]):
    reses = [64, 32, 16, 8]
    poses = ["down", "mid", "up"]
    attention_maps = attention_store.get_average_attention()

    attn_dict = {}

    reses = [int(training_resolution * item // 512) for item in reses]

    for pos in poses:
        for res in reses:
            temp_list = []
            for item in attention_maps[f"{pos}_cross"]:
                if item.shape[1] == res**2:
                    cross_maps = item.reshape(-1, res, res, item.shape[-1])
                    temp_list.append(cross_maps)
            # if such resolution exists
            if len(temp_list) > 0:
                attn_dict[f"{pos}_{res}"] = temp_list
    return attn_dict
