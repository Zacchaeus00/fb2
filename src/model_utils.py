import torch
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.deberta.modeling_deberta import StableDropout
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


class Model3(torch.nn.Module):
    def __init__(self, ckpt, use_stable_dropout=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(ckpt)
        dropout_class = StableDropout if use_stable_dropout else torch.nn.Dropout
        self.dropout1 = dropout_class(0.1)
        self.dropout2 = dropout_class(0.2)
        self.dropout3 = dropout_class(0.3)
        self.dropout4 = dropout_class(0.4)
        self.dropout5 = dropout_class(0.5)
        self.classifier = torch.nn.Linear(self.backbone.config.hidden_size, 3)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids)
        sequence_output = outputs[0]
        logits1 = self.classifier(self.dropout1(sequence_output))
        logits2 = self.classifier(self.dropout2(sequence_output))
        logits3 = self.classifier(self.dropout3(sequence_output))
        logits4 = self.classifier(self.dropout4(sequence_output))
        logits5 = self.classifier(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)