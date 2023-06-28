# import logging
import re
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.utils import logging
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.configuration_bert import BertConfig

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from geoopt.manifolds import PoincareBallExact
from scipy.optimize import root
from typing import Union, Dict, Any, Tuple, Union

logger = logging.get_logger()

Manifold = PoincareBallExact()

class MLPLayer(nn.Module):
    """
    Head for getting sentence/word embedding based on `pooler_type`.
    'post': hyperbolic MLP.
    'pre': euclidean MLP.
    Default to 'post'.
    """
    def __init__(self, euclidean_size: int, hyperbolic_size: int, pooler_type: str):
        super().__init__()
        self.linear = nn.Linear(euclidean_size, hyperbolic_size)
        self.activation = nn.Tanh()
        self.hyper_output = False if 'pre' in pooler_type else True

    def forward(self, features:torch.Tensor, **kwargs) -> torch.Tensor:
        euclidean_features = self.linear(features)
        # euclidean_features = self.activation(euclidean_features) # TODO: MLP is used now, consider only Linear
        hyperbolic_features = Manifold.expmap0(euclidean_features)

        # pdb.set_trace()

        if self.hyper_output:
            return hyperbolic_features
        else:
            return euclidean_features

class Similarity(nn.Module):
    """
    Hyperbolic Similarity: Dosn't calculate temp now.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return -Manifold.dist(x, y, dim=-1)

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
    
    def _hyper_init(self, config: Union[BertHyperConfig, RobertaHyperConfig], hierarchy_type: str, 
                    dropout_change_layers: int):
        self.hierarchy_type = hierarchy_type
        self.dropout_change_layers = dropout_change_layers

        self.pooler_type = config.pooler_type
        self.mlp = MLPLayer(config.hidden_size, config.hyperbolic_size, self.pooler_type)
        self.similarity = Similarity()
        self.temp = config.temp

        self.level_diffs = None
        self.loss_logs = {'loss_all': [], 'loss1': [], 'loss2': [], 'loss3': [], 'loss4': []}

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
            # FIXME
            if sent_hyperbolic_embedding is None:
                # print(batch_idx, batch_seq_len)
                # print(cls_rank[batch_idx])
                # print(word_hyperbolic_embeddings[batch_idx])
                # Nonetype for the batch_seq_len=2, take the [cls] embedding for sent_hyperbolic_embedding
                sent_hyperbolic_embedding = word_hyperbolic_embeddings[batch_idx, 0]
            sent_hyperbolic_embeddings.append(sent_hyperbolic_embedding)

        sent_hyperbolic_embeddings = torch.stack(sent_hyperbolic_embeddings, dim=0)
        return sent_hyperbolic_embeddings # (bs, hidden_len)

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
        prefix_p = re.compile(r'encoder\.layer\.(\d+)\.')
        for name, module in encoder.named_modules():
            if isinstance(module, nn.Dropout):
                res = prefix_p.findall(name)
                if res and int(res[0]) < self.dropout_change_layers:
                    module.p = p

    def _init_level_diffs(self, level_num: int):
        
        def hn2en(hyperbolic_norm: float) -> float:
            return root(lambda x: Manifold.dist0(x.item()).item() - hyperbolic_norm, x0=0.).x.item()
        
        max_hyperbolic_norm = Manifold.dist0(0.996)
        # assume that the first two levels are the same
        euclidean_norms = [hn2en(max_hyperbolic_norm * 0.9 * 0.7 ** level) for level in range(level_num - 1)]
        self.level_diffs = [euclidean_norms[i] - euclidean_norms[i + 1] for i in range(level_num - 2)]

    def _hyper_forward(self, encoder, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pair=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ori_input_ids = input_ids
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1) # level num

        if self.hierarchy_type.count("dropout") or self.hierarchy_type.count("mixup"):
            # init self.level_diffs
            if not self.level_diffs or len(self.level_diffs) != num_sent - 2:
                self._init_level_diffs(num_sent)

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
            # # every level has different dropout
            # for level in range(num_sent):
            #     if level == 0:
            #         self._change_dropout(encoder, 0.1)
            #     else:
            #         self._change_dropout(encoder, 1 - 0.9 * 0.97**(level - 1))
            #     # self._change_dropout(encoder, 1 - 0.95**level)
            
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
                # for levels_outputs
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

                hyperbolic_embedding = self._get_hyperbolic_embedding(levels_attention_mask[level], outputs) # (bs * num_sent, hidden_size)
                hyperbolic_embedding = hyperbolic_embedding.reshape(batch_size, -1, hyperbolic_embedding.shape[-1])
                for sublevel in range(hyperbolic_embedding.size(1)):
                    levels_hyperbolic_embedding.append(hyperbolic_embedding[:, sublevel])
            # for levels_outputs
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
            loss1 = torch.tensor(0.).to(self.device)
            loss2 = torch.tensor(0.).to(self.device)
            loss3 = torch.tensor(0.).to(self.device)
            loss1_count = 0
            loss2_count = 0
            he_base: torch.Tensor = levels_hyperbolic_embedding[0] # (batch_size, hidden_size)
            last_adj_dist: torch.Tensor = None
            adj_dist_diffs = []
            if self.hierarchy_type.count('dropout'):
                for level in range(1, num_sent):
                    he1: torch.Tensor = levels_hyperbolic_embedding[level - 1] # (batch_size, hidden_size)
                    he2: torch.Tensor = levels_hyperbolic_embedding[level] # (batch_size, hidden_size)

                    # old l1: positive&negative from previous level
                    hyperbolic_similarity = self.similarity(he1.unsqueeze(1), he2.unsqueeze(0)) # (bs, bs)
                    levels_hyperbolic_similarity.append(hyperbolic_similarity)
                    labels = torch.arange(hyperbolic_similarity.shape[0]).to(dtype=torch.long, device=self.device)
                    loss1 += loss_fn(hyperbolic_similarity / self.temp, labels)
                    loss1_count += 1

                    '''# new l1: negative from level 0, positive from previous level
                    he1_tmp = he_base.expand(he_base.shape[0], he_base.shape[0], he_base.shape[1]) # (bs, bs, hs)
                    he1_tmp = torch.diagonal_scatter(he1_tmp, he1.transpose(0, 1)) # (bs, bs, hs)
                    hyperbolic_similarity = self.similarity(he2.unsqueeze(1), he1_tmp)
                    levels_hyperbolic_similarity.append(hyperbolic_similarity)
                    labels = torch.arange(hyperbolic_similarity.shape[0]).to(dtype=torch.long, device=self.device)
                    loss += loss_fn(hyperbolic_similarity / self.temp, labels)
                    '''

                    margin_0 = 5e-3 # fixed margin
                    '''# old l2: make previous level embeddings farther from origin than present level embeddings.
                    if level > 1: # first 2 levels can be seen as the same level.
                        he1_dist = Manifold.dist0(he1)
                        he2_dist = Manifold.dist0(he2)
                        loss2 += torch.logsumexp(
                            torch.cat(
                                [torch.zeros((1,), dtype=he1_dist.dtype, device=he1_dist.device), (he2_dist - he1_dist + margin_0)],
                                dim=0
                            # ), 
                            ) / (self.temp),
                            0
                        )
                        loss2_count += 1
                    '''
                    # new l2
                    if level > 1:
                        he1_dist = Manifold.dist0(he1)
                        he2_dist = Manifold.dist0(he2)
                        # loss2 += F.relu(he2_dist - he1_dist + self.level_diffs[level - 2]).mean() # dynamic
                        loss2 += F.relu(he2_dist - he1_dist + margin_0).mean() # static
                        loss2_count += 1

                    '''# l3
                    adj_dist = Manifold.dist(he_base, he2, dim=-1)
                    if level > 1:
                        adj_dist_diffs.append(last_adj_dist - adj_dist)
                    last_adj_dist = adj_dist
                    '''
                margin_1 = 5e-3
                '''# l3 logsumexp
                if adj_dist_diffs:
                    adj_dist_diffs = torch.cat(adj_dist_diffs, dim=-1)
                    loss3 = torch.logsumexp(
                        torch.cat([torch.zeros((adj_dist_diffs.shape[0], 1), dtype=adj_dist_diffs.dtype, device=adj_dist_diffs.device),
                                (adj_dist_diffs + margin_1) / self.temp], dim=-1),
                        dim=-1
                    ).mean()
                '''
                # l3 triplet
                # if adj_dist_diffs:
                #     triplet_score = torch.stack([adj_dist_diffs[l - 1] - adj_dist_diffs[l] + margin_1
                #                                 for l in range(1, len(adj_dist_diffs))], dim=-1)
                #     loss3 = triplet_score.mean(dim=-1).mean() # TODO: use mean(dim=-1) or max(dim=-1)

            if self.hierarchy_type.count('mixup'):
                pass

            loss = torch.tensor(0.).to(device=self.device)
            if loss1_count:
                loss1 /= loss1_count
                self.loss_logs['loss1'].append(loss1.item())
                loss += loss1
            if loss2_count:
                loss2 /= loss2_count
                self.loss_logs['loss2'].append(loss2.item())
                loss += loss2
            if loss3:
                loss += loss3
                self.loss_logs['loss3'].append(loss3.item())
            self.loss_logs['loss_all'].append(loss.item())
            
            levels_hyperbolic_similarity = torch.cat(levels_hyperbolic_similarity, dim=0)

        elif self.hierarchy_type.count("token_cutoff"):
            # flatten input
            flat_input_ids = input_ids.reshape(-1, input_ids.shape[-1])
            flat_attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
            flat_token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else token_type_ids
            
            # get output
            levels_outputs = encoder(
                flat_input_ids,
                attention_mask=flat_attention_mask,
                token_type_ids=flat_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )

            hyperbolic_embedding = self._get_hyperbolic_embedding(flat_attention_mask, levels_outputs) # (bs * num_sent, hidden_size)
            hyperbolic_embedding = hyperbolic_embedding.reshape(batch_size, -1, hyperbolic_embedding.shape[-1])


            # Gather all embeddings if using distributed training
            if dist.is_initialized() and self.training:
                he_list = [torch.zeros_like(hyperbolic_embedding) for _ in range(dist.get_world_size())]
                
                # Allgather
                dist.all_gather(tensor_list=he_list, tensor=hyperbolic_embedding.contiguous())

                # Since allgather results do not have gradients, we replace the
                # current process's corresponding embeddings with original tensors
                he_list[dist.get_rank()] = hyperbolic_embedding

                hyperbolic_embedding = torch.cat(he_list, dim=0)
            # L1
            local_hyperbolic_similarity = self.similarity(hyperbolic_embedding[:, :, None, :], 
                                                          hyperbolic_embedding[:, None, :, :],
                                                         ) # shape of [bs, num_sent, num_sent]
            
            loss1 = torch.tensor(0.).to(self.device)
            loss1_count = 0
            for sent_i in range(batch_size):
                for l_p in loss_pair[sent_i]:
                    # logsigmoid
                    loss1 += -F.logsigmoid((local_hyperbolic_similarity[sent_i, l_p[0], l_p[1]] - 
                                            local_hyperbolic_similarity[sent_i, l_p[2], l_p[3]]) / 750)
                                        #   local_hyperbolic_similarity[sent_i, l_p[2], l_p[3]]) / self.temp) # TODO: decide temp
                    
                    '''# triplet
                    margin_2 = 5e-6
                    loss1 += F.relu(margin_2 + \
                                    local_hyperbolic_similarity[sent_i, l_p[2], l_p[3]] - \
                                    local_hyperbolic_similarity[sent_i, l_p[0], l_p[1]])
                    '''
                    loss1_count += 1
            if loss1_count:
                loss1 /= loss1_count

            # L2
            global_hyperbolic_similarity = self.similarity(hyperbolic_embedding[:, :1], \
                                                           hyperbolic_embedding[None, :, 1],# FIXME: change to 0
                                                          ) # shape of [bs, bs]
            global_hyperbolic_similarity = torch.diagonal_scatter(global_hyperbolic_similarity, 
                                                                  local_hyperbolic_similarity[:, 0, -1])
            # contrastive-like loss for L2
            loss2 = F.cross_entropy(
                        global_hyperbolic_similarity / self.temp, 
                        torch.arange(global_hyperbolic_similarity.shape[0]).to(
                            dtype=torch.long, device=self.device
                        )
                    )

            # loss accumulation
            loss = torch.tensor(0.).to(device=self.device)

            self.loss_logs['loss1'].append(loss1.item())
            loss += loss1 # FIXME

            self.loss_logs['loss2'].append(loss2.item())
            loss += loss2

            self.loss_logs['loss_all'].append(loss.item())

            levels_hyperbolic_similarity = (local_hyperbolic_similarity, global_hyperbolic_similarity)


        if len(self.loss_logs['loss_all']) % 125 == 1:
            self.display_loss('avg')

        if not return_dict:
            output = (levels_hyperbolic_similarity,) + \
                (levels_outputs.get("hidden_states", None), levels_outputs.get("attentions", None))
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=levels_hyperbolic_similarity,
            hidden_states=levels_outputs.get("hidden_states", None),
            attentions=levels_outputs.get("attentions", None),
        )
        

    def _sentemb_forward(self, encoder, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pair=None,
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

    def display_loss(self, mode: Union[int, str]):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if mode == 'avg':
            logger.info("Average loss:\n{}".format(
                {k: sum(v) / len(v) for k, v in self.loss_logs.items() if v}
            ))
        elif isinstance(mode, int):
            logger.info("Loss every {} inputs:\n{}".format(
                mode, {
                    k: [(i_loss, sum(v[: i_index + 1]) / (i_index + 1)) for i_index, i_loss in enumerate(v) if (i_index % mode) == 0] 
                    for k, v in self.loss_logs.items() if v
                }
            ))
        else:
            logger.error(f"Can not display loss in mode {mode}")

    def custom_param_init(self, config: Union[BertHyperConfig, RobertaHyperConfig]):
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f'{config.num_layers} layers of Bert/Roberta are used for trainning.')
        unfreeze_layers = ['pooler'] + [f'layer.{11 - i}' for i in range(config.num_layers)]
        encoder = self.bert if hasattr(self, 'bert') else self.roberta
        for name, param in encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        
        # mlp for hyperbolic
        # nn.init.uniform_(self.mlp.linear.weight, 0, 6e-3)
        # nn.init.uniform_(self.mlp.linear.bias, 0, 6e-3)
        nn.init.normal_(self.mlp.linear.weight, 0, 6e-3)
        nn.init.normal_(self.mlp.linear.bias, 0, 6e-3)

    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            loss_pair=None,
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
                loss_pair=loss_pair,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

class BertForHyper(InitAndForward, BertPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = BertHyperConfig

    def __init__(self, config, hierarchy_type="dropout", dropout_change_layers=12, *model_args, **model_kwargs):
        super().__init__(config, hierarchy_type=hierarchy_type)
        # self.model_args = model_kwargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        super()._hyper_init(config, hierarchy_type=hierarchy_type, dropout_change_layers=dropout_change_layers)

class RobertaForHyper(InitAndForward, RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = RobertaHyperConfig

    def __init__(self, config, hierarchy_type="dropout", dropout_change_layers=12, *model_args, **model_kwargs):
        super().__init__(config, hierarchy_type=hierarchy_type)
        # self.model_args = model_kwargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        super()._hyper_init(config, hierarchy_type=hierarchy_type, dropout_change_layers=dropout_change_layers)
