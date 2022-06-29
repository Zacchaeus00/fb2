import torch


# Re-init script copied from https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
def reinit_deberta_v2_pooler(model):
    model.pooler.dense.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.pooler.dense.bias.data.zero_()
    for p in model.pooler.parameters():
        p.requires_grad = True


def reinit_deberta_v2_layers(model, n=0):
    for layer in model.deberta.encoder.layer[-n:]:
        for module in layer.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
