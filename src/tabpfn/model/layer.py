#  Copyright (c) Prior Labs GmbH 2025.

# TODO: Seems like there's a lot in this file that is over-parametrized for regular
# usage. Could likely just remove it.
from __future__ import annotations

import functools
from functools import partial
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.modules.transformer import Module, Tensor

from tabpfn.model.memory import support_save_peak_mem_factor
from tabpfn.model.mlp import MLP
from tabpfn.model.multi_head_attention import MultiHeadAttention

HIDDEN_SIZE_LIMIT = 512
MLP_SAVE_PEAK_MEM_FACTOR = 32


class LayerNorm(torch.nn.LayerNorm):
    """Custom LayerNorm module that supports saving peak memory factor.

    This module extends the PyTorch LayerNorm implementation to handle FP16 inputs
    efficiently and support saving peak memory factor.

    Args:
        *args: Positional arguments passed to the base LayerNorm class.
        **kwargs: Keyword arguments passed to the base LayerNorm class.
    """

    # TODO: Not sure why this needs to be wrapped with functools.wraps
    @functools.wraps(torch.nn.LayerNorm.__init__)  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any):  # type: ignore
        super().__init__(*args, **kwargs)

    @support_save_peak_mem_factor  # type: ignore
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the layer normalization.

        If the input is FP16 and the normalized shape is less than 512, the computation
        is forced to be in FP16 to improve performance.

        Args:
            x: The input tensor.

        Returns:
            The layer normalized tensor.
        """
        # this if statement has the following function:
        # torch.amp.autocast wants to run layer_norm in fp32
        # but that has very bad effects for our performance (up to 2x slower)
        # thus we force fp16, if the input to the module is fp16, which is the case
        # if autocast is used.
        # WARNING: this could lead to instabilities for higher hidden sizes (> 512),
        # thus we only do this for smaller hidden sizes

        if x.dtype == torch.float16 and sum(self.normalized_shape) < HIDDEN_SIZE_LIMIT:
            with torch.amp.autocast("cuda" if x.is_cuda else "cpu", enabled=False):
                return super().forward(x)

        return super().forward(x)

    def forward(
        self,
        input: torch.Tensor,
        *,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
    ) -> torch.Tensor:
        """Perform layer normalization on the input tensor.

        Args:
            input: The input tensor.
            allow_inplace: Whether to allow in-place operations. Default is False.
            save_peak_mem_factor: The factor to save peak memory. Default is None.

        Returns:
            The layer normalized tensor.
        """
        x = input
        input_shape = x.shape

        x = x.reshape(-1, *self.normalized_shape)
        x = self._compute(
            x,
            allow_inplace=allow_inplace,
            save_peak_mem_factor=save_peak_mem_factor,
        )
        return x.reshape(input_shape)


class PerFeatureEncoderLayer(Module):
    """Transformer encoder layer that processes each feature block separately.

    This layer consists of multi-head attention between features, multi-head
    attention between items, and feedforward neural networks (MLPs).

    It supports various configurations and optimization options.

    Args:
        d_model: The dimensionality of the input and output embeddings.
        nhead: The number of attention heads.
        dim_feedforward:
            The dimensionality of the feedforward network.
            Default is None (2 * d_model).
        activation: The activation function to use in the MLPs.
        layer_norm_eps: The epsilon value for layer normalization.
        pre_norm:
            Whether to apply layer normalization before or after the attention
            and MLPs.
        device: The device to use for the layer parameters.
        dtype: The data type to use for the layer parameters.
        recompute_attn: Whether to recompute attention during backpropagation.
        second_mlp: Whether to include a second MLP in the layer.
        layer_norm_with_elementwise_affine:
            Whether to use elementwise affine parameters in layer normalization.
        zero_init: Whether to initialize the output of the MLPs to zero.
        save_peak_mem_factor:
            The factor to save peak memory, only effective with post-norm.
        attention_between_features: Whether to apply attention between feature blocks.
        multiquery_item_attention: Whether to use multiquery attention for items.
        multiquery_item_attention_for_test_set:
            Whether to use multiquery attention for the test set.
        attention_init_gain: The gain value for initializing attention parameters.
        d_k:
            The dimensionality of the query and key vectors.
            Default is (d_model // nhead).
        d_v:
            The dimensionality of the value vectors. Default is (d_model // nhead).
        precomputed_kv: Precomputed key-value pairs for attention.
    """

    __constants__: ClassVar = ["batch_first"]

    def __init__(  # noqa: PLR0913
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        recompute_attn: bool = False,
        second_mlp: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        zero_init: bool = False,
        save_peak_mem_factor: int | None = None,
        attention_between_features: bool = True,
        multiquery_item_attention: bool = False,
        multiquery_item_attention_for_test_set: bool = False,
        two_sets_of_queries: bool = False,
        attention_init_gain: float = 1.0,
        d_k: int | None = None,
        d_v: int | None = None,
        precomputed_kv: None | torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert d_model % nhead == 0 or (d_k is not None and d_v is not None)
        if multiquery_item_attention_for_test_set and multiquery_item_attention:
            raise ValueError(
                "Cannot use both multiquery_item_attention_for_test_set"
                "and multiquery_item_attention",
            )
        if two_sets_of_queries and not multiquery_item_attention_for_test_set:
            raise ValueError(
                "two_sets_of_queries requires multiquery_item_attention_for_test_set",
            )

        if d_k is None:
            d_k = d_model // nhead

        if d_v is None:
            d_v = d_model // nhead

        self.self_attn_between_features: MultiHeadAttention | None = None
        if attention_between_features:
            self.self_attn_between_features = MultiHeadAttention(
                input_size=d_model,
                output_size=d_model,
                d_k=d_k,
                d_v=d_v,
                nhead=nhead,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
                init_gain=attention_init_gain,
            )

        if isinstance(precomputed_kv, tuple):
            precomputed_k, precomputed_v = precomputed_kv
            precomputed_kv = None
        else:
            precomputed_k = precomputed_v = None

        self.self_attn_between_items = MultiHeadAttention(
            input_size=d_model,
            output_size=d_model,
            d_k=d_k,
            d_v=d_v,
            nhead=nhead,
            device=device,
            dtype=dtype,
            share_kv_across_n_heads=nhead if multiquery_item_attention else 1,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
            precomputed_k=precomputed_k,
            precomputed_v=precomputed_v,
            precomputed_kv=precomputed_kv,
            init_gain=attention_init_gain,
            two_sets_of_queries=(
                multiquery_item_attention_for_test_set and two_sets_of_queries
            ),
        )

        if dim_feedforward is None:
            dim_feedforward = 2 * d_model

        self.mlp = MLP(
            size=d_model,
            hidden_size=dim_feedforward,
            activation=activation,
            device=device,
            dtype=dtype,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
        )

        self.layer_norms = nn.ModuleList(
            [
                LayerNorm(
                    d_model,  # type: ignore
                    layer_norm_eps,
                    elementwise_affine=layer_norm_with_elementwise_affine,
                    **factory_kwargs,
                )
                for _ in range(4 if second_mlp else 3)
            ],
        )

        self.second_mlp: MLP | None = None
        if second_mlp:
            self.second_mlp = MLP(
                size=d_model,
                hidden_size=dim_feedforward,
                activation=activation,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
            )

        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.save_peak_mem_factor = save_peak_mem_factor
        self.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )
        self.two_sets_of_queries = two_sets_of_queries

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("save_peak_mem_factor", None)
        super().__setstate__(state)

    def forward(  # noqa: C901
        self,
        state: Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,
        att_src: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Pass the input through the encoder layer.
        
        Args:
            state:
                The transformer state passed as input to the layer of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos:
                The position from which on everything is treated as test
                set.
            cache_trainset_representation:
                Whether to cache the trainset representation.
                If single_eval_pos is set (> 0 and not None), create a cache of the
                trainset KV. This may require a lot of memory. Otherwise, use
                cached KV representations for inference.
            att_src:
                The tensor to attend to from the final layer of the encoder.
                It has a shape of
                (batch_size, num_train_items, num_feature_blocks, d_model).
                This does not work with multiquery_item_attention_for_test_set and
                cache_trainset_representation at this point.

        Returns:
            The transformer state passed through the encoder layer.
        """

        assert (
            len(state.shape) == 4
        ), "src must be of shape (batch_size, num_items, num feature blocks, d_model)"
        if single_eval_pos is None:
            single_eval_pos = 0

        save_peak_mem_factor = self.save_peak_mem_factor
        if cache_trainset_representation and not single_eval_pos:
            assert self.self_attn_between_items.has_cached_kv
            save_peak_mem_factor = None

        if att_src is not None:
            assert (
                not self.multiquery_item_attention_for_test_set
            ), "Not implemented yet."
            assert not cache_trainset_representation, "Not implemented yet."
            assert not single_eval_pos, (
                "single_eval_pos should not be set, as the train representation"
                " is in att_src"
            )

        ps_final: Tensor | None = None

        if self.self_attn_between_features is None:
            assert not cache_trainset_representation, "Not implemented yet."
            assert state.shape[2] == 1, (
                f"One group architecture expects one feature group, "
                f"but got {state.shape[2]} feature groups."
            )

        def attn_between_features(x: torch.Tensor) -> tuple[torch.Tensor, Tensor | None]:
            assert self.self_attn_between_features is not None
            # Assuming self.self_attn_between_features now returns (output, ps)
            # The subtask description implies this is the case for "self.self_attn" calls
            if return_attention:
                output, ps = self.self_attn_between_features(
                    x,
                    save_peak_mem_factor=save_peak_mem_factor,
                    add_input=True,
                    allow_inplace=True,
                    return_attention=return_attention,
                )
                return output, ps
            else:
                output = self.self_attn_between_features(
                    x,
                    save_peak_mem_factor=save_peak_mem_factor,
                    add_input=True,
                    allow_inplace=True,
                )
                return output, None

        def attn_between_items(x: torch.Tensor) -> tuple[torch.Tensor, Tensor | None]:
            # we need to transpose as self attention always treats
            # dim -2 as the sequence dimension
            ps_item_specific = None
            if self.multiquery_item_attention_for_test_set:
                new_x_test_res, new_x_train_res = None, None
                if single_eval_pos < x.shape[1]:
                    if return_attention:
                        new_x_test_output, ps_test = self.self_attn_between_items(
                            x[:, single_eval_pos:].transpose(1, 2),
                            x[:, :single_eval_pos].transpose(1, 2)
                            if single_eval_pos
                            else None,
                            save_peak_mem_factor=save_peak_mem_factor,
                            cache_kv=False,
                            add_input=True,
                            allow_inplace=True,
                            use_cached_kv=not single_eval_pos,
                            reuse_first_head_kv=True,
                            use_second_set_of_queries=self.two_sets_of_queries,
                            return_attention=return_attention,
                        )
                    else:
                        new_x_test_output = self.self_attn_between_items(
                            x[:, single_eval_pos:].transpose(1, 2),
                            x[:, :single_eval_pos].transpose(1, 2)
                            if single_eval_pos
                            else None,
                            save_peak_mem_factor=save_peak_mem_factor,
                            cache_kv=False,
                            add_input=True,
                            allow_inplace=True,
                            use_cached_kv=not single_eval_pos,
                            reuse_first_head_kv=True,
                            use_second_set_of_queries=self.two_sets_of_queries,
                        )
                        ps_test = None
                    new_x_test_res = new_x_test_output.transpose(1, 2)
                    ps_item_specific = ps_test # Prioritize test ps
                else:
                    new_x_test_res = None

                if single_eval_pos:
                    if return_attention:
                        new_x_train_output, ps_train = self.self_attn_between_items(
                            x[:, :single_eval_pos].transpose(1, 2),
                            x[:, :single_eval_pos].transpose(1, 2),
                            save_peak_mem_factor=save_peak_mem_factor,
                            cache_kv=cache_trainset_representation,
                            only_cache_first_head_kv=True,
                            add_input=True,
                            allow_inplace=True,
                            use_cached_kv=False,
                            return_attention=return_attention,
                        )
                    else:
                        new_x_train_output = self.self_attn_between_items(
                            x[:, :single_eval_pos].transpose(1, 2),
                            x[:, :single_eval_pos].transpose(1, 2),
                            save_peak_mem_factor=save_peak_mem_factor,
                            cache_kv=cache_trainset_representation,
                            only_cache_first_head_kv=True,
                            add_input=True,
                            allow_inplace=True,
                            use_cached_kv=False,
                        )
                        ps_train = None
                    new_x_train_res = new_x_train_output.transpose(1, 2)
                    if ps_item_specific is None: # If no test ps, use train ps
                        ps_item_specific = ps_train
                else:
                    new_x_train_res = None
                
                output_combined = torch.cat(
                    [x_ for x_ in [new_x_train_res, new_x_test_res] if x_ is not None],
                    dim=1,
                )
                return output_combined, ps_item_specific

            attention_src_x = None
            if att_src is not None:
                attention_src_x = att_src.transpose(1, 2)
            elif single_eval_pos:
                attention_src_x = x[:, :single_eval_pos].transpose(1, 2)

            # Assuming self.self_attn_between_items now returns (output, ps)
            if return_attention:
                output_transposed, ps_item_specific = self.self_attn_between_items(
                    x.transpose(1, 2),
                    attention_src_x,
                    save_peak_mem_factor=save_peak_mem_factor,
                    cache_kv=cache_trainset_representation and single_eval_pos,
                    add_input=True,
                    allow_inplace=True,
                    use_cached_kv=cache_trainset_representation and not single_eval_pos,
                    return_attention=return_attention,
                )
            else:
                output_transposed = self.self_attn_between_items(
                    x.transpose(1, 2),
                    attention_src_x,
                    save_peak_mem_factor=save_peak_mem_factor,
                    cache_kv=cache_trainset_representation and single_eval_pos,
                    add_input=True,
                    allow_inplace=True,
                    use_cached_kv=cache_trainset_representation and not single_eval_pos,
                )
                ps_item_specific = None
            return output_transposed.transpose(1, 2), ps_item_specific

        mlp_save_peak_mem_factor = (
            save_peak_mem_factor * 8 if save_peak_mem_factor is not None else None
        )

        # Storing functions and a flag indicating if they return ps
        sublayer_info = []
        if self.self_attn_between_features is not None:
            sublayer_info.append({'func': attn_between_features, 'returns_ps': True, 'name': 'attn_features'})
        else:
            assert state.shape[2] == 1, (
                "If there is no attention between features, the number of feature"
                " blocks must be 1."
            )

        mlp_call = partial(
            self.mlp.__call__,
            save_peak_mem_factor=mlp_save_peak_mem_factor
            if (
                mlp_save_peak_mem_factor is not None
                and state.numel() // state.shape[-1] // mlp_save_peak_mem_factor
            )
            > 32
            else None,
            add_input=True,
            allow_inplace=True,
        )

        if self.second_mlp is not None:
            second_mlp_call = partial(
                self.second_mlp.__call__,
                save_peak_mem_factor=mlp_save_peak_mem_factor,
                add_input=True,
                allow_inplace=True,
            )
            # Order: attn_features, second_mlp, attn_items, mlp
            # If attn_features is None, then: second_mlp, attn_items, mlp (but attn_features check is outside)
            # So, if second_mlp is present, it's always after attn_features (if any) and before attn_items
            # This means we might need to adjust insertion point if attn_features is not present.
            # However, the original code appends attn_items and mlp later.
            # Let's reconstruct the sequence carefully.
            # Original: [attn_features (optional)], attn_items, mlp. If second_mlp: inserts at index 1.
            # This means:
            #   - [attn_features, second_mlp, attn_items, mlp]
            #   - [second_mlp, attn_items, mlp] (if no attn_features, this is wrong, second_mlp is not at index 1)
            # The original code implies:
            #   1. Start with [attn_features] or []
            #   2. If second_mlp, insert it at index 1 (if list is shorter, effectively appends or becomes index 0 or 1).
            #      A more robust way is to define the sequence.
            # Let's define the sequence explicitly based on original logic:
            # sublayers = [attn_features (opt), attn_items, mlp] then second_mlp is inserted.

            # Tentative explicit order:
            # 1. attn_between_features (if present)
            # 2. second_mlp (if present)
            # 3. attn_between_items
            # 4. mlp

            # If second_mlp is present, it's inserted at index 1 of the 'sublayers' list.
            # If attn_features is present, sublayers = [attn_features, attn_items, mlp], then second_mlp makes it [attn_features, second_mlp, attn_items, mlp]
            # If attn_features is NOT present, sublayers = [attn_items, mlp], then second_mlp makes it [attn_items, second_mlp, mlp]
            # This seems to be the logic.

            idx_for_second_mlp = 0
            if self.self_attn_between_features is not None:
                idx_for_second_mlp = 1
            
            sublayer_info.insert(idx_for_second_mlp, {'func': second_mlp_call, 'returns_ps': False, 'name': 'second_mlp'})
            sublayer_info.insert(idx_for_second_mlp +1, {'func': attn_between_items, 'returns_ps': True, 'name': 'attn_items'})
            sublayer_info.append({'func': mlp_call, 'returns_ps': False, 'name': 'mlp'})

        else: # No second_mlp
            sublayer_info.append({'func': attn_between_items, 'returns_ps': True, 'name': 'attn_items'})
            sublayer_info.append({'func': mlp_call, 'returns_ps': False, 'name': 'mlp'})


        for i, layer_info in enumerate(sublayer_info):
            sublayer_func = layer_info['func']
            returns_ps = layer_info['returns_ps']
            layer_name = layer_info['name'] # For debugging

            layer_norm = self.layer_norms[i] # Assuming layer_norms corresponds to this new explicit order

            if self.pre_norm:
                raise AssertionError(
                    "Pre-norm implementation is wrong, as the residual should never"
                    " be layer normed here.",
                )
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=save_peak_mem_factor,
                )
            
            if returns_ps:
                current_ps: Tensor | None
                state, current_ps = sublayer_func(state)
                if current_ps is not None: # Update ps_final if current sublayer provides ps
                    ps_final = current_ps
            else:
                state = sublayer_func(state)

            if not self.pre_norm:
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=save_peak_mem_factor,
                )

        return state, ps_final

    def empty_trainset_representation_cache(self) -> None:
        """Empty the trainset representation cache."""
        self.self_attn_between_items.empty_kv_cache()

        # TODO: This could be None but this just ignored that fact here.
        assert self.self_attn_between_features is not None
        self.self_attn_between_features.empty_kv_cache()  # not necessary, just in case
