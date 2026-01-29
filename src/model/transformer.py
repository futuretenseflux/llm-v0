#transformer.py
import torch.nn as nn
from llm.model.decoder import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim_model, dim_k, num_q_heads, group_size, num_decoder_layers, intermediate_size, eps=1e-6, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(dim_model, dim_k, num_q_heads, group_size, intermediate_size, eps, dropout=dropout) for _ in range(num_decoder_layers)]
        )
        self.rms_norm = nn.RMSNorm(dim_model, eps)
        self.output_head = nn.Linear(dim_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.rms_norm(x)
        logits = self.output_head(x)
        return logits