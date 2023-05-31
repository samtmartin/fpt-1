import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class MaskedSelfAttentionHead(nn.Module):
    """
    Implements a masked self-attention layer using scaled dot-product attention. 
    Converts raw embeddings into 'contextual embeddings', creating a 
    representation for each element in the sequence that incorporates 
    information from the whole sequence.
    """

    def __init__(
            self, 
            head_size: int, 
            num_input_features: int, 
            block_size: int, 
            dropout: float):
        super().__init__()

        self._query = nn.Linear(num_input_features, head_size, bias=False)
        self._key = nn.Linear(num_input_features, head_size, bias=False)
        self._value = nn.Linear(num_input_features, head_size, bias=False)
        self.register_buffer(
            '_tril', torch.tril(torch.ones(block_size, block_size)))
        self._dropout = nn.Dropout(dropout)

    def forward(
            self, 
            x: Tensor) -> Tensor:
        """
        Hidden state `x` is of shape (batch_size,block_size,num_input_features) 
        or (B,T,C). Output `out` is of shape (batch_size,block_size,head_size) 
        or (B,T,h).
        """

        B,T,C = x.shape
        
        # Project raw embeddings into query, key, and value vectors.
        q = self._query(x) # (B,T,h)
        k = self._key(x) # (B,T,h)
        v = self._value(x) # (B,T,h)

        # Compute attention weights. Uses dot product as similarity function to 
        # create attention scores (i.e. how much `q` and `k` relate to each 
        # other) then multiplies scores, which can be arbitrarily large, by 
        # scaling factor to normalize variance (i.e. prevent softmax saturation) 
        # and then softmaxes so that column values sum to 1.
        wei = (q @ k.transpose(-2,-1) * 
               k.shape[-1]**-0.5) # (B,T,h) @ (B,h,T) -> (B,T,T)
        wei = wei.masked_fill(self._tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self._dropout(wei)

        # Update embeddings.
        out = wei @ v # (B,T,T) @ (B,T,h) -> (B,T,h)
        return out

class MultiHeadAttention(nn.Module):
    """
    Implements a multi-head attention layer, representing multiple self-
    attention heads. Having multiple heads allows the model to focus on several 
    aspects of sequence at once (softmax tends to focus on a single aspect of 
    element similarity).
    """

    def __init__(
            self, 
            num_heads: int, 
            head_size: int, 
            num_input_features: int, 
            block_size: int, 
            dropout: float):
        super().__init__()

        self._heads = nn.ModuleList(
            [MaskedSelfAttentionHead
             (head_size, num_input_features, block_size, dropout) for _ in 
             range(num_heads)])
        self._proj = nn.Linear(head_size * num_heads, num_input_features)
        self._dropout = nn.Dropout(dropout)

    def forward(
            self, 
            x: Tensor) -> Tensor:
        # (B,T,head_size*num_heads). Product (i.e. C) equals num_input_features.
        out = torch.cat([h(x) for h in self._heads], dim=-1) 
        out = self._dropout(self._proj(out))
        return out

class FeedFoward(nn.Module):
    """
    Implements a position-wise feed forward layer. This is where most of the 
    model's memorization is hypothesized to happen.
    """

    def __init__(
            self, 
            num_input_features: int, 
            dropout: float):
        super().__init__()
        
        self._sequential = nn.Sequential(
            # Rule of thumb from research is that the hidden size of first layer 
            # should be four times the size of the input embedding dimensions.
            nn.Linear(num_input_features, 4 * num_input_features),
            # GELU activation function is most commonly used.
            nn.GELU(),
            nn.Linear(4 * num_input_features, num_input_features),
            nn.Dropout(dropout),
        )

    def forward(
            self, 
            x: Tensor) -> Tensor:
        return self._sequential(x)

class DecoderBlock(nn.Module):
    """
    Implements a decoder transformer block.
    """

    def __init__(
            self, 
            num_input_features: int, 
            num_heads: int, 
            block_size: int, 
            dropout: float):
        super().__init__()
        
        # `head_size` represents the number of dimensions we are projecting 
        # into. Calculated as a multiple of the embedding dimensions (i.e. 
        # num_input_features), which allows head ouputs to be concatenated 
        # together to produce shape of inputs (i.e. embed dim).
        head_size = num_input_features // num_heads
        self._mha = MultiHeadAttention(num_heads, head_size, num_input_features, 
                                     block_size, dropout)
        self._ffwd = FeedFoward(num_input_features, dropout)
        self._ln1 = nn.LayerNorm(num_input_features)
        self._ln2 = nn.LayerNorm(num_input_features)

    def forward(
            self, 
            x: Tensor) -> Tensor:
        # Deviating from original transformer paper by using pre layer 
        # normalization. Pre-norm tends to be more stable during training and 
        # doesn't require learning rate warm-up.
        x = x + self._mha(self._ln1(x))
        x = x + self._ffwd(self._ln2(x))
        return x

class FPT(nn.Module):
    """
    Implements Financial Projections Transformer 1. FPT-1 is a batch learning, 
    multivariate regression model that predicts future financial performance 
    given historical performance. FPT uses decoder-only transformer architecture 
    (i.e. causal or autoregressive attention).
    """

    def __init__(
            self, 
            num_input_features: int, 
            num_output_features: int, 
            block_size: int, 
            num_heads: int, 
            num_layers: int, 
            dropout: float):
        super().__init__()

        self._position_embedding_table = nn.Embedding(
            block_size, num_input_features)
        self._blocks = nn.Sequential(*[DecoderBlock(
            num_input_features, num_heads, block_size, dropout) for _ in 
            range(num_layers)])
        self._ln = nn.LayerNorm(num_input_features)
        self._linear_mapping = nn.Linear(
            num_input_features, num_output_features)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self, 
            inputs: Tensor) -> Tensor:
        B, T, C = inputs.shape
        
        # Augments `inputs` with position-dependent pattern of values (uses 
        # index to encode position of elements in sequence), allowing attention 
        # head and feed-forward layers to incorporate positional information 
        # during transformations.
        pos_emb = self._position_embedding_table(torch.arange(T)) # (T,C)
        x = inputs + pos_emb # (B,T,C)

        x = self._blocks(x) # (B,T,C)
        x = self._ln(x) # (B,T,C)

        outputs = self._linear_mapping(x) # (B,T,num_output_features)
        return outputs

    def forecast(
            self, 
            inputs: Tensor, 
            num_predictions: int) -> Tensor:
        B, T, C = inputs.shape

        for _ in range(num_predictions):
            # Crop `inputs` to the last 'block size' row(s).
            cropped_inputs = inputs[:, -T:]
            outputs = self(cropped_inputs)
            # Focus only on the last time step.
            outputs = outputs[..., -1, :] # (1,C)
            outputs = outputs.expand(1,-1,-1) # (1,1,C) 
            # Append sampled index to the running sequence
            inputs = torch.cat((inputs, outputs), dim=1) # (B,T+1,C)

        inputs = inputs.squeeze(dim=0) # (T+num_predictions,C)
        return inputs