import logging
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.configuration_bert import BertConfig

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from geoopt.manifolds import PoincareBallExact
from typing import Union, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)

Manifold = PoincareBallExact()

class MLPLayer(nn.Module):
    """
    Head for getting sentence/word embedding based on `pooler_type`.
    'post': hyperbolic MLP.
    'pre': euclidean MLP.
    """
    def __init__(self, euclidean_size: int, hyperbolic_size: int):
        super().__init__()
        self.linear = nn.Linear(euclidean_size, hyperbolic_size)
        self.activation = nn.Tanh()

    def forward(self, features:torch.Tensor, **kwargs) -> torch.Tensor:
        euclidean_features = self.linear(features)
        euclidean_features = self.activation(euclidean_features)
        hyperbolic_features = Manifold.expmap0(euclidean_features)
        hyperbolic_features = euclidean_features

        return hyperbolic_features

class Similarity(nn.Module):
    """
    Hyperbolic Similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp

    def forward(self, x, y):
        return -Manifold.dist(x, y) / self.temp

class BertHyperConfig(BertConfig):
    def __init__(self,
        pooler_type: str = 'cls',
        hyperbolic_size: int = 768,
        temp: float = 0.05, 
        num_layers: int = 12,
        **kwargs
    ): 
        super().__init__(**kwargs)
        self.pooler_type = pooler_type
        self.hyperbolic_size = hyperbolic_size
        self.temp = temp
        self.num_layers = num_layers

class RobertaHyperConfig(RobertaConfig):
    def __init__(self,
        pooler_type: str = 'cls',
        hyperbolic_size: int = 768,
        temp: float = 0.05, 
        num_layers: int = 12,
        **kwargs
    ): 
        super().__init__(**kwargs)
        self.pooler_type = pooler_type
        self.hyperbolic_size = hyperbolic_size
        self.temp = temp
        self.num_layers = num_layers

class InitAndForward:

    def _hyper_init(self, config: Union[BertHyperConfig, RobertaHyperConfig]):
        self.pooler_type = config.pooler_type
        self.mlp = MLPLayer(config.hidden_size, config.hyperbolic_size)
        self.similarity = Similarity(temp=config.temp)

        logger.info(f'{config.num_layers} layers of Bert/Roberta are used for trainning.')
        unfreeze_layers = ['pooler'] + [f'layer.{11 - i}' for i in range(config.num_layers)]
        encoder = self.bert if hasattr(self, 'bert') else self.roberta
        for name, param in encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        self.init_weights() # linear embedding layernorm
        nn.init.uniform_(self.mlp.linear.weight, 0, 0.001) # mlp for hyperbolic
        nn.init.uniform_(self.mlp.linear.bias, 0, 0.001)
    
    def _avg_embedding(self, attention_mask: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        last_hidden: torch.Tensor = outputs.last_hidden_state # (bs, seq_len, hidden_len)
        hidden_states: torch.Tensor = outputs.hidden_states  # Tuple of (bs, seq_len, hidden_len)

        with_cls = True if 'with_special_tokens' in self.pooler_type else 'with_cls' in self.pooler_type
        with_sep = True if 'with_special_tokens' in self.pooler_type else 'with_sep' in self.pooler_type

        last_hidden = last_hidden * attention_mask.unsqueeze(-1)    # (bs, seq_len, hidden_len)
        word_hyperbolic_embeddings = self.mlp(last_hidden)  # (bs, seq_len, hidden_len)
        masked_seq_len = attention_mask.sum(dim=-1).tolist() # (bs)
        
        # 先考虑使用与 CLS 的余弦相似度作为权重
        cls_similarity = F.cosine_similarity(last_hidden[:, 0: 1], last_hidden, dim=-1) # (bs, seq_len)
        cls_rank = torch.argsort(cls_similarity, dim=-1, descending=True).tolist() # (bs, seq_len)

        sent_hyperbolic_embeddings = []
        for batch_idx, batch_seq_len in enumerate(masked_seq_len):
            sent_hyperbolic_embedding = None
            for word_idx in cls_rank[batch_idx]:
                if word_idx == 0 and not with_cls or \
                    word_idx == batch_seq_len - 1 and not with_sep or \
                    word_idx >= batch_seq_len:
                    continue
                if sent_hyperbolic_embedding != None:
                    sent_hyperbolic_embedding = Manifold.mobius_add(
                        sent_hyperbolic_embedding,
                        word_hyperbolic_embeddings[batch_idx, word_idx]
                    )
                else:
                    sent_hyperbolic_embedding = word_hyperbolic_embeddings[batch_idx, word_idx]
            sent_hyperbolic_embeddings.append(sent_hyperbolic_embedding)
        sent_hyperbolic_embeddings = torch.stack(sent_hyperbolic_embeddings, dim=0)
        return sent_hyperbolic_embeddings # (bs, hidden_len)

    def _avg_embedding_deprecated(self, attention_mask: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        last_hidden: torch.Tensor = outputs.last_hidden_state # (bs, seq_len, hidden_len)
        hidden_states: torch.Tensor = outputs.hidden_states  # Tuple of (bs, seq_len, hidden_len)

        with_cls = True if 'with_special_tokens' in self.pooler_type else 'with_cls' in self.pooler_type
        with_sep = True if 'with_special_tokens' in self.pooler_type else 'with_sep' in self.pooler_type

        masked_seq_len = attention_mask.sum(dim=-1).tolist() # (bs)
        # 先考虑使用与 CLS 的余弦相似度作为权重
        sent_hyperbolic_embeddings = []
        for batch_idx, batch_seq_len in enumerate(masked_seq_len):
            cls_embedding = last_hidden[batch_idx, 0]
            # (batch_seq_len, hidden_len)
            batch_hidden_states = last_hidden[batch_idx][ 
                (0 if with_cls else 1):
                (batch_seq_len if with_sep else batch_seq_len - 1)
            ]
            # (batch_seq_len, )
            cls_similarity = F.cosine_similarity(cls_embedding, batch_hidden_states, dim=-1)
            cls_rank = torch.argsort(cls_similarity, descending=True).tolist()

            word_hyperbolic_embeddings = self.mlp(batch_hidden_states)
            sent_hyperbolic_embedding = word_hyperbolic_embeddings[cls_rank[0]]
            for word_idx in cls_rank[1: ]:
                sent_hyperbolic_embedding = Manifold.mobius_add(
                    sent_hyperbolic_embedding, 
                    word_hyperbolic_embeddings[word_idx]
                )
            sent_hyperbolic_embeddings.append(sent_hyperbolic_embedding)
        sent_hyperbolic_embeddings = torch.stack(sent_hyperbolic_embeddings, dim=0)
        return sent_hyperbolic_embeddings # (bs, hidden_len)
                                        # (bs, seq_len)
    def _get_hyperbolic_embedding(self, attention_mask: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        last_hidden: torch.Tensor = outputs.last_hidden_state # (bs, seq_len, hidden_len)
        hidden_states: torch.Tensor = outputs.hidden_states  # Tuple of (bs, seq_len, hidden_len)

        if 'cls' in self.pooler_type:
            cls_embedding = last_hidden[:, 0]
            hyperbolic_sentence_embedding = self.mlp(cls_embedding)
            return hyperbolic_sentence_embedding #(bs, hidden_len)
        elif 'avg' in self.pooler_type: 
            return self._avg_embedding(attention_mask, outputs)
            # return self._avg_embedding_deprecated(attention_mask, outputs)
        else:
            raise NotImplementedError

    def _hyper_forward(self, encoder, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ori_input_ids = input_ids
        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        num_sent = input_ids.size(1)

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

        # Get raw embeddings
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        # Get hyperbolic embeddings
        hyperbolic_embedding: torch.Tensor = self._get_hyperbolic_embedding(attention_mask, outputs)
        hyperbolic_embedding = hyperbolic_embedding.view(batch_size, num_sent, hyperbolic_embedding.shape[-1])

        # Separate representation
        he1, he2 = hyperbolic_embedding[:, 0], hyperbolic_embedding[:, 1]

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            # Dummy vectors for allgather
            he1_list = [torch.zeros_like(he1) for _ in range(dist.get_world_size())]
            he2_list = [torch.zeros_like(he2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=he1_list, tensor=he1.contiguous())
            dist.all_gather(tensor_list=he2_list, tensor=he2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            he1_list[dist.get_rank()] = he1
            he2_list[dist.get_rank()] = he2
            # Get full batch embeddings: (bs x N, hidden)
            he1 = torch.cat(he1_list, 0)
            he2 = torch.cat(he2_list, 0)
        
        hyperbolic_similarity = self.similarity(he1.unsqueeze(1), he2.unsqueeze(0)) # (bs, bs)
        # hyperbolic_distance = (1 - nn.CosineSimilarity(dim=-1)(he1.unsqueeze(1), he2.unsqueeze(0))) / self.config.temp
        labels = torch.arange(hyperbolic_similarity.shape[0]).to(dtype=torch.long, device=self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(hyperbolic_similarity, labels)

        if not return_dict:
            output = (hyperbolic_similarity,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=hyperbolic_similarity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _sentemb_forward(self, encoder, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        return_dict: bool = return_dict if return_dict is not None else self.config.use_return_dict
        paired_data: bool = len(input_ids.shape) == 3

        if paired_data:
            batch_size = input_ids.size(0)
            # Number of sentences in one instance
            # 2: pair instance; 3: pair instance with a hard negative
            num_sent = input_ids.size(1)

            # Flatten input for encoding
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        
        # Get raw embeddings
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        
        # Get hyperbolic embeddings
        hyperbolic_embedding: torch.Tensor = self._get_hyperbolic_embedding(attention_mask, outputs)
        if paired_data:
            hyperbolic_embedding = hyperbolic_embedding.view(batch_size, num_sent, hyperbolic_embedding.shape[-1])

        if not return_dict:
            return (outputs.last_hidden_state, hyperbolic_embedding, outputs.hidden_states)
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=hyperbolic_embedding,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states
        )

    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False
    ) -> Union[Tuple, Dict[str, Any]]:
        forward_fn = self._sentemb_forward if sent_emb else self._hyper_forward
        
        return forward_fn(self.bert if hasattr(self, 'bert') else self.roberta, 
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

class BertForHyper(InitAndForward, BertPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = BertHyperConfig

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        # self.model_args = model_kwargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        super()._hyper_init(config)

class RobertaForHyper(InitAndForward, RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = RobertaHyperConfig

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        # self.model_args = model_kwargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        super()._hyper_init(config)
