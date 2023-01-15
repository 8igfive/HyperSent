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

    def _hyper_init(self, config: Union[BertHyperConfig, RobertaHyperConfig], hierarchy_type: str):
        self.hierarchy_type = hierarchy_type

        self.pooler_type = config.pooler_type
        self.mlp = MLPLayer(config.hidden_size, config.hyperbolic_size)
        self.similarity = Similarity(temp=config.temp)
        self.temp = config.temp

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
            # FIXME
            if sent_hyperbolic_embedding is None:
                print(batch_idx, batch_seq_len)
                print(cls_rank[batch_idx])
                print(word_hyperbolic_embeddings[batch_idx])
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

    def _change_dropout(self, encoder, p):
        for module in encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = p

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
        num_sent = input_ids.size(1) # level num

        if self.hierarchy_type == "dropout":
            # Split input based on level and flatten them.
            levels_input_ids = [input_ids[:, :2].reshape(-1, input_ids.size(-1))]
            for level in range(2, num_sent):
            # levels_input_ids = [] # FIXME: new
            # for level in range(num_sent): # FIXME: new
                levels_input_ids.append(input_ids[:, level])
            levels_attention_mask = [attention_mask[:, :2].reshape(-1, attention_mask.size(-1))]
            for level in range(2, num_sent):
            # levels_attention_mask = [] # FIXME: new
            # for level in range(num_sent):  # FIXME: new  
                levels_attention_mask.append(attention_mask[:, level])
            if token_type_ids is not None:
                levels_token_type_ids = [token_type_ids[:, :2].reshape(-1, token_type_ids.size(-1))]
                for level in range(2, num_sent):
                # levels_token_type_ids = [] # FIXME: new
                # for level in range(num_sent): # FIXME: new  
                    levels_token_type_ids.append(token_type_ids[:, level])
            else:
                levels_token_type_ids = [None] * (num_sent - 1)
                # levels_token_type_ids = [None] * num_sent # FIXME: new
            
            levels_hyperbolic_embedding = [] # List of Tensor(size=(bs, hidden_size))
            levels_outputs = None
            for level in range(num_sent - 1): # first two levels are in a single input
                # change dropout
                # self._change_dropout(encoder, 1 - 0.9**(level + 1))
                self._change_dropout(encoder, 1 - 0.9 * 0.97**level)
            # for level in range(num_sent): # FIXME: new
            #     if level == 0:
            #         self._change_dropout(encoder, 0.1)
            #     else:
            #         self._change_dropout(encoder, 1 - 0.9 * 0.97**(level - 1))
            #     # self._change_dropout(encoder, 1 - 0.95**level) # FIXME: new

                # Get raw embeddings
                outputs = encoder(
                    levels_input_ids[level],
                    attention_mask=levels_attention_mask[level],
                    token_type_ids=levels_token_type_ids[level],
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                    return_dict=True,
                )
                if not return_dict:
                    if not levels_outputs:
                        levels_outputs = [([] if isinstance(output, torch.Tensor) else
                                           tuple([] for _ in output) if isinstance(output, tuple) else
                                           pdb.set_trace()) for output in outputs]
                    for i, value in enumerate(outputs):
                        if isinstance(value, torch.Tensor):
                            levels_outputs[i].append(value)
                        else:
                            for j, item in enumerate(value):
                                levels_outputs[i][j].append(item)
                else:
                    if not levels_outputs:
                        levels_outputs = {key: ([] if isinstance(output, torch.Tensor) else
                                           tuple([] for _ in output) if isinstance(output, tuple) else
                                           pdb.set_trace()) for key, output in outputs.items()}
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            levels_outputs[key].append(value)
                        else:
                            for j, item in enumerate(value):
                               levels_outputs[key][j].append(item) 

                hyperbolic_embedding: torch.Tensor = self._get_hyperbolic_embedding(levels_attention_mask[level], outputs) # (bs * num_sent, hidden_size)
                hyperbolic_embedding = hyperbolic_embedding.reshape(batch_size, -1, hyperbolic_embedding.shape[-1])
                for sublevel in range(hyperbolic_embedding.size(1)):
                    levels_hyperbolic_embedding.append(hyperbolic_embedding[:, sublevel])
            if not return_dict:
                for i, values in enumerate(levels_outputs):
                    if isinstance(values, list):
                        levels_outputs[i] = torch.cat(values, dim=0)
                    else:
                        levels_outputs[i] = tuple(torch.cat(item, dim=0) for item in values)
                levels_outputs = tuple(levels_outputs)
            else:
                for key, values in levels_outputs.items():
                    if isinstance(values, list):
                        levels_outputs[key] = torch.cat(values, dim=0)
                    else:
                        levels_outputs[key] = tuple(torch.cat(item, dim=0) for item in values)

            # Gather all embeddings if using distributed training
            if dist.is_initialized() and self.training:
                for level in range(num_sent):
                    he = levels_hyperbolic_embedding[level]
                    # Dummy vectors for allgather
                    he_list = [torch.zeros_like(he) for _ in range(dist.get_world_size())]
                    
                    # Allgather
                    dist.all_gather(tensor_list=he_list, tensor=he.contiguous())

                    # Since allgather results do not have gradients, we replace the
                    # current process's corresponding embeddings with original tensors
                    he_list[dist.get_rank()] = he

                    levels_hyperbolic_embedding[level] = torch.cat(he_list, dim=0)

            levels_hyperbolic_similarity = []
            loss_fn = nn.CrossEntropyLoss()
            loss = torch.tensor(0.0).to(levels_hyperbolic_embedding[0].device)
            loss_count = 0
            he_base: torch.Tensor = levels_hyperbolic_embedding[0] # (batch_size, hidden_size)
            for level in range(1, num_sent):
                he1: torch.Tensor = levels_hyperbolic_embedding[level - 1] # (batch_size, hidden_size)
                he2: torch.Tensor = levels_hyperbolic_embedding[level] # (batch_size, hidden_size)

                # old l1
                hyperbolic_similarity = self.similarity(he1.unsqueeze(1), he2.unsqueeze(0)) # (bs, bs)
                levels_hyperbolic_similarity.append(hyperbolic_similarity)
                labels = torch.arange(hyperbolic_similarity.shape[0]).to(dtype=torch.long, device=self.device)
                loss += loss_fn(hyperbolic_similarity, labels)

                # new l1
                # he1_tmp = he_base.expand(he_base.shape[0], he_base.shape[0], he_base.shape[1]) # (bs, bs, hs)
                # he1_tmp = torch.diagonal_scatter(he1_tmp, he1.transpose(0, 1)) # (bs, bs, hs)
                # hyperbolic_similarity = self.similarity(he2.unsqueeze(1), he1_tmp)
                # levels_hyperbolic_similarity.append(hyperbolic_similarity)
                # labels = torch.arange(hyperbolic_similarity.shape[0]).to(dtype=torch.long, device=self.device)
                # loss += loss_fn(hyperbolic_similarity, labels)

                # l2
                he1_dist = Manifold.dist0(he1)
                he2_dist = Manifold.dist0(he2)
                loss += torch.logsumexp(
                    (he2_dist - he1_dist) / (self.temp * he1_dist.shape[0]),
                    0
                )

                loss_count += 1

            loss /= loss_count
            levels_hyperbolic_similarity = torch.cat(levels_hyperbolic_similarity, dim=0)

            if not return_dict:
                output = (levels_hyperbolic_similarity,) + levels_outputs[2:]
                return ((loss,) + output) if loss is not None else output
            return SequenceClassifierOutput(
                loss=loss,
                logits=levels_hyperbolic_similarity,
                hidden_states=levels_outputs.get("hidden_states", None),
                attentions=levels_outputs.get("attentions", None),
            )
        else:
            raise NotImplementedError
        
        '''deprecated
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
        '''

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

    def __init__(self, config, hierarchy_type="dropout", *model_args, **model_kwargs):
        super().__init__(config, hierarchy_type=hierarchy_type)
        # self.model_args = model_kwargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        super()._hyper_init(config, hierarchy_type=hierarchy_type)

class RobertaForHyper(InitAndForward, RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = RobertaHyperConfig

    def __init__(self, config, hierarchy_type="dropout", *model_args, **model_kwargs):
        super().__init__(config, hierarchy_type=hierarchy_type)
        # self.model_args = model_kwargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        super()._hyper_init(config, hierarchy_type=hierarchy_type)
