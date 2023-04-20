# import logging
import os
import sys
import pdb
import json
from pathlib import Path
from packaging import version
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Mapping

from geoopt.manifolds import PoincareBallExact

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    logging,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.debug_utils import DebugOption
from transformers import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock

# logger = logging.getLogger(__name__)
logger = logging.get_logger()

# Manifold
Manifold = PoincareBallExact()

class GenerateEmbeddingCallback(TrainerCallback):
    def _prepare_input(self, args: TrainingArguments, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(args, v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(args, v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=args.device)
            if args.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info('Generating Hyperbolic Embeddings for sentences in train_dataset.')
        
        model = kwargs.pop('model')
        train_dataloader = kwargs.pop('train_dataloader')

        hyperbolic_embeddings = []
        embeddings_num = 0
        for inputs in train_dataloader:
            inputs = self._prepare_input(args=args, data=inputs)
            outputs = model(**inputs, sent_emb=True)
            tmp_embeddings: torch.Tensor = outputs.pooler_output
            tmp_embeddings = tmp_embeddings.view(-1, tmp_embeddings.shape[-1])
            hyperbolic_embeddings.append(tmp_embeddings)
            embeddings_num += tmp_embeddings.shape[0]
            if embeddings_num >= args.dump_embeddings_num:
                break

        hyperbolic_embeddings = torch.cat(hyperbolic_embeddings, dim=0).tolist()
        with open(os.path.join(args.output_dir, 'hyperbolic_embeddings.json'), 'w', encoding='utf8') as fo:
            json.dump(hyperbolic_embeddings, fo)

        logger.info(f'Hyperbolic Embeddings for sentences in train_dataset were saved to {os.path.join(args.output_dir, "hyperbolic_embeddings.json")}')

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics: Dict[str, float] = kwargs.pop('metrics')

        # Determine the new best metric / best model checkpoint
        if metrics is not None and args.metric_for_best_model is not None:
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if args.greater_is_better else np.less
            if (
                state.best_metric is None
                or state.best_model_checkpoint is None
                or operator(metric_value, state.best_metric)
            ):
                control.should_save = True

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info(f'metric_for_best_model: {args.metric_for_best_model}, best_metric: {state.best_metric}, best_model_checkpoint: {state.best_model_checkpoint}')


class HyperTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        # params['similarity'] = lambda x, y: -Manifold.dist(x, y)

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        if eval_senteval_transfer and self.args.eval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        self.model.eval()
        results = se.eval(tasks)
        
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

        metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
        if eval_senteval_transfer and self.args.eval_transfer:
            avg_transfer = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                avg_transfer += results[task]['devacc']
                metrics['eval_{}'.format(task)] = results[task]['devacc']
            avg_transfer /= 7
            metrics['eval_avg_transfer'] = avg_transfer

        self.log(metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    '''
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Override for displaying detailed loss.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            # scaler = self.scaler if self.do_grad_scaling else None
            # loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            # return loss_mb.reduce_mean().detach().to(self.args.device)
            raise NotImplementedError

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1 and len(loss.shape) > 1:
            loss = loss.mean(dim=0)  # mean() to average on multi-gpu parallel training
        # loss of shape=[3]
        loss_all = loss[0]

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss_all = loss_all / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss_all).backward()
        elif self.use_apex:
            with amp.scale_loss(loss_all, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss_all = self.deepspeed.backward(loss_all)
        else:
            loss_all.backward()

        return loss.detach()
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Override for displaying detailed loss.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                # raw_outputs = smp_forward_only(model, inputs)
                # if has_labels:
                #     if isinstance(raw_outputs, dict):
                #         loss_mb = raw_outputs["loss"]
                #         logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                #     else:
                #         loss_mb = raw_outputs[0]
                #         logits_mb = raw_outputs[1:]

                #     loss = loss_mb.reduce_mean().detach().cpu()
                #     logits = smp_nested_concat(logits_mb)
                # else:
                #     loss = None
                #     if isinstance(raw_outputs, dict):
                #         logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                #     else:
                #         logits_mb = raw_outputs
                #     logits = smp_nested_concat(logits_mb)
                raise NotImplementedError
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.detach()
                    if len(loss.shape) > 1:
                        loss = loss.mean(dim=0)

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
        '''