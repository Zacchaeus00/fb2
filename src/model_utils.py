import torch
from transformers.trainer_pt_utils import get_parameter_names


def get_deberta_v2_layers(model, n):
    for layer in model.deberta.encoder.layer[-n:]:
        yield layer


# Re-init script copied from https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
def reinit_pooler(model):
    model.pooler.dense.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.pooler.dense.bias.data.zero_()
    for p in model.pooler.parameters():
        p.requires_grad = True


def reinit_layers(model, n=0):
    if model.config.model_type == "deberta-v2":
        layer_generator = get_deberta_v2_layers(model, n)
    else:
        raise NotImplementedError
    for layer in layer_generator:
        for module in layer.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


# https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/trainer.py#L209
def get_optimizer_grouped_parameters(model, n=0):
    raise NotImplementedError
