from dataclasses import asdict
from json import dumps
from os import environ
from typing import Iterable, Dict

from simpletransformers.classification import ClassificationModel
from tensorflow import distribute, config, tpu

from modern_talking.matchers import LabelPolicy


def is_running_on_colab():
    return 'COLAB_GPU' in environ


def gpu_count() -> int:
    return int(environ['COLAB_GPU'])


def setup_colab_tpu():
    """
    Setup TPUs for usage in Google Colab, only if running on Colab.
    """
    if not is_running_on_colab() or gpu_count() >= 1:
        # Skip setup because either not running on Colab or on GPU environment.
        return

    # Special resolver for Google Colaboratory.
    resolver = distribute.cluster_resolver.TPUClusterResolver(tpu='')
    config.experimental_connect_to_cluster(resolver)
    tpu.experimental.initialize_tpu_system(resolver)


MODEL_DESCRIPTION_KEYS: Dict[str, str] = {
    "adafactor_beta1": "Adafactor beta1",
    "adafactor_clip_threshold": "Adafactor clip threshold",
    "adafactor_decay_rate": "Adafactor decay rate",
    "adafactor_eps": "Adafactor epsilon",
    "adafactor_relative_step": "Adafactor relative step",
    "adafactor_scale_parameter": "Adafactor scale parameter",
    "adafactor_warmup_init": "Adafactor warmup init",
    "adam_epsilon": "ADAM epsilon",
    "add_cross_attention": "Add cross attention",
    "architectures": "Architectures",
    "attention_probs_dropout_prob":
        "Attention probabilities dropout probability",
    "augment_train_texts": "Augment training texts",
    "bos_token_id": "Beginning-of-stream token ID",
    "chunk_size_feed_forward": "Feed forward chunk size",
    "cosine_schedule_num_cycles": "Cosine schedule number of cycles",
    "do_lower_case": "Make input texts lower case",
    "dynamic_quantize": "Dynamic quantization",
    "early_stopping_consider_epochs": "Early stopping consider epochs",
    "early_stopping_delta": "Early stopping delta",
    "early_stopping_metric": "Early stopping metric",
    "early_stopping_metric_minimize": "Early stopping minimize metric",
    "early_stopping_patience": "Early stopping patience epochs",
    "eos_token_id": "End-of-stream token ID",
    "eval_batch_size": "Validation batch size",
    "evaluate_each_epoch": "Validation after each epoch",
    "finetuning_task": "Fine-tuning task",
    "fp16": "Half-precision floating-point format",
    "gradient_accumulation_steps": "Gradient accumulation steps",
    "hidden_act": "Hidden layer activation function",
    "hidden_dropout_prob": "Hidden layer dropout probability",
    "hidden_size": "Hidden layer size",
    "initializer_range": "Weight initialization standard deviation",
    "intermediate_size": "Intermediate layer size",
    "layer_norm_eps": "Layer normalization epsilon",
    "learning_rate": "Learning rate",
    "local_rank": "Local rank",
    "manual_seed": "Manual seed",
    "max_grad_norm": "Maximum norm for gradients",
    "max_position_embeddings": "Maximum position embeddings",
    "max_seq_length": "Maximum sequence length",
    "missing_train_labels": "Handle missing training labels",
    "model_name": "Model name",
    "model_type": "Model type",
    "num_attention_heads": "Attention heads",
    "num_hidden_layers": "Hidden layers",
    "num_train_epochs": "Training epochs",
    "optimizer": "Optimizer",
    "over_sample_random": "Over-sample randomly",
    "pad_token_id": "Padding token ID",
    "polynomial_decay_schedule_lr_end":
        "Polynomial decay schedule end learning rate",
    "polynomial_decay_schedule_power": "Polynomial decay schedule power",
    "position_embedding_type": "Position embedding type",
    "quantized_model": "Quantized model",
    "regression": "Regression",
    "reprocess_input_data": "Reprocess input data",
    "scheduler": "Scheduler",
    "sep_token_id": "Separation token ID",
    "shuffle_train_data": "Shuffle training data",
    "skip_special_tokens": "Skip special tokens",
    "sliding_window": "Sliding window",
    "special_tokens_list": "Special tokens list",
    "stride": "Stride for sliding window",
    "tie_value": "Tie value",
    "tokenizer_name": "Tokenizer name",
    "tokenizer_type": "Tokenizer type",
    "train_batch_size": "Training batch size",
    "type_vocab_size": "Type vocabulary size",
    "use_early_stopping": "Early stopping",
    "vocab_size": "Vocabulary size",
    "warmup_ratio": "Warmup ratio",
    "warmup_steps": "Warmup steps",
    "weight_decay": "Weight decay",
}


def describe_model_configuration(
        model: ClassificationModel,
        augment: int,
        label_policy: LabelPolicy,
        over_sample: bool,
        shuffle: bool,
) -> str:
    configuration = {}
    configuration.update(asdict(model.args))
    configuration.update(model.config.to_dict())
    configuration["augment_train_texts"] = augment
    configuration["missing_train_labels"] = label_policy.value
    configuration["over_sample_random"] = over_sample
    configuration["shuffle_train_data"] = shuffle

    def transform_value(value: any) -> str:
        if isinstance(value, Iterable) and not isinstance(value, str):
            value = ", ".join(map(str, value))
        return str(value) \
            .replace("eval", "validation") \
            .replace("_", " ") \
            .replace("True", "yes")\
            .replace("False", "no")

    lines = [
        f"{MODEL_DESCRIPTION_KEYS[key]}: {transform_value(value)}"
        for key, value in configuration.items()
        if key in MODEL_DESCRIPTION_KEYS.keys() and value is not None
    ]
    lines = sorted(lines)
    return "\n".join(lines)
