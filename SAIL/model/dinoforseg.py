from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder
from typing import Optional, Union
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn
import math
import torch

class Dinov2EncoderForSegmentation(Dinov2Encoder):
    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
        ) -> Union[tuple, BaseModelOutput]:
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None

            for i, layer_module in enumerate(self.layer[:-1]):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        layer_head_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # Customized last layer for segmentation operation
            for layer_module in self.layer[-1:]:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i+1] if head_mask is not None else None
                layer_outputs = self.custom_layer(layer_module, hidden_states, layer_head_mask, output_attentions)
                hidden_states = layer_outputs[0]
                # TODO: support gradient_checkpointing


            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
    
    def custom_layer(
            self,
            layer_module: nn.Module,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor],
            output_attentions: bool
        ):
        """
        Customized layer for segmentation operation
        Whether skip the residual connection in this layer operation
        """
        # print(f"custom layer without residual and with qq weight")
        self_attention_outputs = self.custom_attn(
            layer_module.attention,
            layer_module.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = layer_module.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        # suit for ealier and latter transformer versions
        if hasattr(layer_module, 'drop_path1'):
            hidden_states = layer_module.drop_path1(attention_output) + hidden_states
        else:
            hidden_states = layer_module.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention

        layer_output = layer_module.norm2(hidden_states)

        if not self.ignore_residual:
            # skip the residual connection
            layer_output = layer_module.mlp(layer_output)
            layer_output = layer_module.layer_scale2(layer_output)

            # second residual connection
            if hasattr(layer_module, 'drop_path2'):
                layer_output = layer_module.drop_path2(layer_output) + hidden_states
            else:
                layer_output = layer_module.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs
    
    def custom_attn(
            self,
            attn: nn.Module,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor],
            output_attentions: bool
        ):
        self_outputs = self.custom_selfattn(attn.attention, hidden_states, head_mask, output_attentions)

        attention_output = attn.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    
    def custom_selfattn(
            self,
            attn: nn.Module,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor],
            output_attentions: bool
        ):
        mixed_query_layer = attn.query(hidden_states)

        key_layer = attn.transpose_for_scores(attn.key(hidden_states))
        value_layer = attn.transpose_for_scores(attn.value(hidden_states))
        query_layer = attn.transpose_for_scores(mixed_query_layer)


        if self.attetion_type == 'qk':
            # qk
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        elif self.attetion_type == 'qq':
            # qq
            attention_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))
        
        else:
            raise ValueError(f"Unsupported attention type: {self.attetion_type}")

        attention_scores = attention_scores / math.sqrt(attn.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = attn.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

if __name__ == "__main__":
    from PIL import Image
    import requests
    import torch
    from transformers import AutoImageProcessor, AutoModel

    def modify_vit(type_: str) -> None:
        from transformers.models.dinov2 import modeling_dinov2
        if type_ == 'seg':
            modeling_dinov2.Dinov2Encoder = Dinov2EncoderForSegmentation
        else:
            pass
    modify_vit('seg')
    
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to('cuda')

    inputs = processor(images=image, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs[0]
