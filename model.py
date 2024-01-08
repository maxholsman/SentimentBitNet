import torch
import torch.nn as nn
import torch.nn.functional as F

def absmax_quantization(x, bit=8, nl_next=False):
    Qb = 2**(bit - 1)
    
    # find the maximum absolute value in the tensor
    max_val = torch.max(torch.abs(x))
    min_val = torch.min(x)
    
    if nl_next:
        shifted_x = x - min_val
        max_val = torch.max(torch.abs(shifted_x))
        
        scale_factor = Qb / max_val
        x = torch.round(shifted_x * scale_factor)
    else:
        # using the max values, we can calculate the scaling factor for each value in the tensor to map it to the range appropriate range
        scale_factor = Qb / max_val
        
        # now we can quantize the tensor, rounding to the nearest integer
        x = torch.round(x * scale_factor)
    
    dequant = max_val / Qb
    
    return x.to(torch.int8), dequant, max_val, min_val

def absmax_dequantization(x, max_val, nl_next=False, min_val=None, bit=8):
    Qb = 2**(bit - 1)
    
    reverse_scale_factor = max_val / Qb
    
    x = x * reverse_scale_factor
    
    return x.to(torch.float32) # return to float32 which is original precision

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bit=8, nl_next=False, bias=True):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.nl_next = nl_next
        
        if bias is True:
            self.bias = nn.Parameter(torch.randn(self.out_features))
        else:
            self.register_parameter("bias", None)
        
        self.weights = nn.Parameter(torch.randn(self.out_features, self.in_features))
        
        # # print(f"weights: {self.weights.shape}")
        # # Upon initialization, the weights will be randomly initialized using the kaiming uniform method
        # self.parameter_initialization()
        
    # def parameter_initialization(self):
    #     nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        
    def forward(self, x):
        
        input_norm = F.layer_norm(x, (self.in_features,)) # changed to norm over batch
        
        input_quant, dequant, gamma, eta = absmax_quantization(input_norm, nl_next=self.nl_next)
        
        weights = self.weights - self.weights.mean()
        
        weight_quant = torch.sign(weights)
        
        output = torch.matmul(input_quant.float(), weight_quant.t())
            
        beta = torch.norm(self.weights, p=1) / (self.in_features * self.out_features)
        
        if self.nl_next:
            output = (output * dequant + eta) * beta 
        else:
            output = output * dequant * beta
        
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).expand_as(output)
        
        return output

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #simply a mapping between a number and a vector, number represents a word in the vocabulary and the vector represents the word in the embedding space
        
    def forward(self, x):
        # print(f"x type: {x.type()}")
        x = self.embedding(x)
        # print(f"x embedded type: {x.type()}")
        return x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)) # by paper

class PositionalEncoding(nn.Module):
    # Positional Encoding is a way of encoding the position of the word in the sentence. They will be added to the embedding vector to give the model a sense of the word's position in the sentence.
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        
        #first will build a matrix of shape (seq_len, d_model) because we need seq_len number of positional encodings for each word in the sentence
        pe = torch.zeros(seq_len, d_model)
        
        # apply one formula to all the even indices of the matrix, and another formula to all the odd indices of the matrix
        # will therefore create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1) # unsqueeze to get shape (seq_len, 1)
        divterm = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # shape (d_model/2)
        # sin for even, cosine for odd
        # self.pe[:, 0::2] = torch.sin(position * divterm)
        # self.pe[:, 1::2] = torch.cos(position * divterm)
        # for i in range(0, d_model, 2):
        #     self.pe[:, i] = torch.sin(position * divterm[i])
        #     self.pe[:, i+1] = torch.cos(position * divterm[i])
        
        pe[:, 0::2] = torch.sin(position * divterm)
        pe[:, 1::2] = torch.cos(position * divterm)
                
        # add a batch dimension to the positional encoding matrix so that it can be added to the embedding vector
        pe = pe.unsqueeze(0) # shape (1, seq_len, d_model) which now has a batch dimension
        
        # now we will save this positional encoding matrix as a buffer so that it can be used later
        self.register_buffer('pe', pe) # keeping this tensor NOT as a parameter but still as a part of the model, tensor will now be saved in the file with the model
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # add positional encoding to the embedding vector. sp
        x = self.dropout(x)
        return x

# for each item in batch, calculate mean and variance indpenendent of other items in the batch, then find new values using their own mean and variance
# also add and multiply two additional parameters, to give the model the ability to amplify or reduce the importance of a given feature

class LayerNorm(nn.Module): 
    def __init__(self, d_model, eps=10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # alpha is a learnable parameter and will be multiplied
        self.beta = nn.Parameter(torch.zeros(d_model)) # beta is a learnable parameter and will be added
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # mean of each item in the batch, not the whole batch
        std = x.std(dim=-1, keepdim=True) # same thing here
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

class FFBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        
        
        self.w1 = BitLinear(d_model, d_ff, nl_next=True) #includes bias
        self.w2 = BitLinear(d_ff, d_model)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # input: [batch, seq_len, d_model]
        x = self.w1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % num_heads == 0, "d_model not divisible by num_heads" # d_model must be divisible by num_heads
        
        self.d_k = d_model // num_heads # d_k is the dimension of each head
        
        self.wq = BitLinear(d_model, d_model, bias=False)
        self.wk = BitLinear(d_model, d_model, bias=False)
        self.wv = BitLinear(d_model, d_model, bias=False)
        
        self.w0 = BitLinear(d_model, d_model, bias=False)
    
    @staticmethod
    def attention(q, k, v, mask=None, dropout=None):
        d_k = q.shape[-1]
        
        att = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32)) # [batch, num_heads, seq_len, seq_len], only transpose last two dimensions beause want each head dimension to stay the same
        
        if mask is not None:
            mask = mask.unsqueeze(1) # [batch, 1, 1, seq_len]
            # print(f"mask shape: {mask.shape}")
            # print(f"att shape: {att.shape}")
            att.masked_fill_(mask == 0, -1e9)
        
        # att = torch.softmax(att, dim=-1)
        att = att.softmax(dim=-1)
        
        if dropout is not None:
            att = dropout(att)
        
        att_values = torch.matmul(att, v)
        
        return att_values, att
        

    def forward(self, q, k, v, mask=None):
        
        # print datatypes of all inputs
        # print(f"q type: {q.type()}")
        # print(f"k type: {k.type()}")
        # print(f"v type: {v.type()}")
        # print(f"mask type: {mask.type()}")
        
        qprime = self.wq(q) # [batch, seq_len, d_model]
        kprime = self.wk(k)
        vprime = self.wv(v)
        
        # split qprime, kprime, vprime into num_heads
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
        qprime = qprime.view(qprime.shape[0], qprime.shape[1], self.num_heads, self.d_k)
        kprime = kprime.view(kprime.shape[0], kprime.shape[1], self.num_heads, self.d_k)
        vprime = vprime.view(vprime.shape[0], vprime.shape[1], self.num_heads, self.d_k)
        
        # transpose qprime, kprime, vprime to [batch, num_heads, seq_len, d_k] so that each head has full access to the sequence
        qprime = qprime.transpose(1, 2)
        kprime = kprime.transpose(1, 2)
        vprime = vprime.transpose(1, 2)
        
        # calculate attention
        x, self.attention = MultiHeadAttention.attention(qprime, kprime, vprime, mask, self.dropout)  
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) # [batch, seq_len, d_model]
        
        out = self.w0(x)
        
        return out
    
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNorm(d_model) # initialized from class defined above
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.ln(x))) # takes care of both the residual connection and the layer normalization, sublayer will be either the multihead attention or the feed forward block

#now we must construct the entire encoder block

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        self.residual_connection = ResidualConnection(d_model, dropout)
        self.multiheadattention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffblock = FFBlock(d_model, d_ff, dropout)
        
    def forward(self, x, src_mask):
        x = self.residual_connection(x, lambda x: self.multiheadattention(x, x, x, src_mask))
        x = self.residual_connection(x, self.ffblock)
        # print(f"output type inside encoder block: {x.type()}")
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x, src_mask)
            # print(f"output type inside encoder after block {i+1}: {x.type()}")
        # print(f"output type inside encoder: {x.type()}")
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = num_classes
        
        self.projection = BitLinear(d_model, num_classes)

    def forward(self, x):
        #applying log softmax for numerical stability
        return self.projection(x)

class SentimentTransformer(nn.Module):
    def __init__(self, encoder, embedding, pos_encoding, proj_layer, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.pos_encoding = pos_encoding
        self.proj_layer = proj_layer
        self.num_classes = num_classes
    
    def encode(self, x, src_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_mask)
        return x

    def project(self, x):
        x = x.mean(dim=1)
        x = self.proj_layer(x)
        return x
    
    def forward(self, x, src_mask):
        # print(f"input shape: {x.shape}")
        x = self.encode(x, src_mask)
        # print(f"input after encoding shape: {x.shape}")
        x = self.project(x)
        # print(f"input after projection shape: {x.shape}")
        return x

def build_sentiment_transfomer(d_model, vocab_size, seq_len, num_heads, d_ff, num_layers, dropout=0.1, num_classes=2):
    embedding = InputEmbedding(d_model, vocab_size)
    pos_encoding = PositionalEncoding(d_model, seq_len, dropout)
    encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
    proj_layer = ProjectionLayer(d_model, num_classes)
    model = SentimentTransformer(encoder, embedding, pos_encoding, proj_layer, num_classes)
    return model




        
        