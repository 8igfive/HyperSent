# import logging
import math
import os
import sys
import torch
import collections
import random
import pdb
import json

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, cast

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.utils import logging
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available

from hypersent.models import RobertaForHyper, BertForHyper, RobertaHyperConfig, BertHyperConfig
from hypersent.trainers import HyperTrainer, GenerateEmbeddingCallback
from hypersent.utils import (
    ModelArguments, DataTrainingArguments, OurTrainingArguments, 
    PrepareFeatures, OurDataCollatorWithPadding
)

# logger = logging.getLogger(__name__) # logging
logger = logging.get_logger() # transformers.utils.logging
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args = cast(ModelArguments, model_args)
    data_args = cast(DataTrainingArguments, data_args)
    training_args = cast(OurTrainingArguments, training_args)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    os.makedirs(training_args.output_dir, exist_ok=training_args.overwrite_output_dir)

    # Save configuration
    all_args = dict()
    for tmp_args in [model_args, data_args, training_args]:
        all_args.update({k: v for k, v in tmp_args.__dict__.items() if \
            type(v) in {int, float, str, list, dict, tuple}})
    with open(os.path.join(training_args.output_dir, 'all_args.config'), 'w', encoding='utf8') as fo:
        json.dump(all_args, fo, indent=4)

    # Setup logging
    # logging.basicConfig(
    #     # "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    #     format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    # )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    elif extension == 'jsonl':
        extension = 'json'
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "disable_hyper": model_args.disable_hyper
    }
    if model_args.pooler_type:
        config_kwargs['pooler_type'] = model_args.pooler_type
    if model_args.hyperbolic_size:
        config_kwargs['hyperbolic_size'] = model_args.hyperbolic_size
    if model_args.temp:
        config_kwargs['temp'] = model_args.temp
    if model_args.num_layers:
        config_kwargs['num_layers'] = model_args.num_layers

    assert model_args.model_type, f'Determine model_type within {MODEL_TYPES}'
    config_class: Union[BertHyperConfig, RobertaHyperConfig] = \
        BertHyperConfig if model_args.model_type == 'bert' else RobertaHyperConfig
    if model_args.config_name:
        config = config_class.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = config_class.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = config_class()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    assert model_args.model_name_or_path, "Requires model_name_or_path."
    model_type: Union[RobertaForHyper, BertForHyper] = \
        BertForHyper if model_args.model_type == 'bert' else RobertaForHyper
    model, loading_info = model_type.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        hierarchy_type=training_args.hierarchy_type,
        dropout_change_layers=training_args.dropout_change_layers,
        output_loading_info=True
    )
    if 'mlp.linear.bias' in loading_info['missing_keys']:
        model.custom_param_init(config)

    model.resize_token_embeddings(len(tokenizer))
    
    column_names = datasets["train"].column_names
    # if training_args.do_train:
    prepare_features = PrepareFeatures(column_names, tokenizer, data_args, training_args)
    train_dataset = datasets["train"]
    if data_args.only_aigen:
        train_dataset = train_dataset.filter(lambda raw: raw['split'] == 'aigen')
    train_dataset = train_dataset.map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    
    data_collator = default_data_collator if data_args.pad_to_max_length \
        else OurDataCollatorWithPadding(tokenizer, aigen=('aigen' in training_args.hierarchy_type), 
                                        aigen_sent_num=len(prepare_features.aigen_keys), 
                                        aigen_batch_size=64, combine_training=True)

    trainer = HyperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[GenerateEmbeddingCallback()]
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        resume_from_checkpoint = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else training_args.resume_from_checkpoint
        )
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        model.display_loss(125) # display loss log.

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
