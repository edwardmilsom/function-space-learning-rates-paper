import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, widthmult=1.0, noqknorm = False, init_scale=1.0, use_forward_pass_rootL=False, depthmult=1.0, affinetransformations=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.qknorm = not noqknorm

        if self.qknorm:
            if not affinetransformations:
                self.q_norm = nn.LayerNorm(self.d_head, elementwise_affine=False)
                self.k_norm = nn.LayerNorm(self.d_head, elementwise_affine=False)
            else:
                self.q_norm = nn.LayerNorm(self.d_head, elementwise_affine=True)
                self.k_norm = nn.LayerNorm(self.d_head, elementwise_affine=True)
        
        # Initialize projection matrices
        nn.init.kaiming_normal_(self.q_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.k_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.v_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_in', nonlinearity='linear')
        self.out_proj.weight.data *= 1.0 / math.sqrt(widthmult)  # Scale for head concatenation
        self.v_proj.weight.data *= init_scale
        self.out_proj.weight.data *= init_scale
        if not use_forward_pass_rootL:
            self.out_proj.weight.data *= 1.0 / math.sqrt(depthmult) #Ensure block variance scales properly with depth

        if self.qknorm:
            self.q_proj.weight.data *= init_scale
            self.k_proj.weight.data *= init_scale
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.d_head)
        
        if self.qknorm:
            # Apply QK normalization - normalize over feature dimension (d_head)
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, init_scale=1.0, use_forward_pass_rootL=False, depthmult=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        nn.init.kaiming_normal_(self.net[0].weight, mode='fan_in', nonlinearity='linear')
        self.net[0].weight.data *= init_scale
        nn.init.zeros_(self.net[0].bias)
        
        nn.init.kaiming_normal_(self.net[2].weight, mode='fan_in', nonlinearity='relu')
        self.net[2].weight.data *= init_scale
        if not use_forward_pass_rootL:
            self.net[2].weight.data *= 1.0 / math.sqrt(depthmult) #Ensure block variance scales properly with depth
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, depthmult=1.0, widthmult=1.0, noqknorm=False, normtype="postnorm", init_scale=1, use_forward_pass_rootL=False, affinetransformations=False):
        super().__init__()
        
        self.depthmult = depthmult
        self.use_forward_pass_rootL = use_forward_pass_rootL

        self.self_attn = MultiHeadAttention(d_model, num_heads, widthmult, noqknorm=noqknorm, init_scale=init_scale, use_forward_pass_rootL=use_forward_pass_rootL, depthmult=depthmult, affinetransformations=affinetransformations)
        if not affinetransformations:
            self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        else:
            self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
            self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True)
        # self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = FeedForward(d_model, d_ff, init_scale, use_forward_pass_rootL=use_forward_pass_rootL, depthmult=depthmult)
        # self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.normtype = normtype
        
    def forward(self, x, mask=None):
        if self.normtype == "prenorm":
            if self.use_forward_pass_rootL:
                x = x + (1/math.sqrt(self.depthmult)) * self.self_attn(self.norm1(x), mask)
                x = x + (1/math.sqrt(self.depthmult)) * self.ff(self.norm2(x))
            else:
                x = x + self.self_attn(self.norm1(x), mask)
                x = x + self.ff(self.norm2(x))

        elif self.normtype == "postnorm":
            if self.use_forward_pass_rootL:
                x = self.norm1(x + (1/math.sqrt(self.depthmult)) * self.self_attn(x, mask))
                x = self.norm2(x + (1/math.sqrt(self.depthmult)) * self.ff(x))
            else:
                x = self.norm1(x + self.self_attn(x, mask))
                x = self.norm2(x + self.ff(x))
        
        elif self.normtype == "postnormpreres":
            if not self.use_forward_pass_rootL:
                raise ValueError("Should probably not be using postnormpreres without use_forward_pass_rootL=True, otherwise initialisation will not be depth invariant.")
            x = x + (1/math.sqrt(self.depthmult)) * self.norm1(self.self_attn(x, mask))
            x = x + (1/math.sqrt(self.depthmult)) * self.norm2(self.ff(x))

        elif self.normtype == "nonorm":
            if self.use_forward_pass_rootL:
                x = x + (1/math.sqrt(self.depthmult)) * self.self_attn(x, mask)
                x = x + (1/math.sqrt(self.depthmult)) * self.ff(x)
            else:
                x = x + self.self_attn(x, mask)
                x = x + self.ff(x)

        else:
            raise ValueError("Invalid normtype")

        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=2, num_layers=2, d_ff=512, 
                 widthmult=1.0, depthmult=1.0, noqknorm=False, normtype="postnorm", init_scale=1, use_forward_pass_rootL=False, affinetransformations=False):
        super().__init__()
        
        # Apply width scaling
        d_model = int(d_model * widthmult)
        d_ff = int(d_ff * widthmult)
        num_heads = int(num_heads * widthmult)
        
        # Apply depth scaling
        num_layers = int(num_layers * depthmult)
        
        self.d_model = d_model
        self.input_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.input_emb.weight, mean=0.0, std=1.0)
        
        self.register_buffer('pos_encoding', self._init_pos_encoding())
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, depthmult, widthmult, noqknorm, normtype, init_scale, use_forward_pass_rootL, affinetransformations)
            for _ in range(num_layers)
        ])
        
        if not affinetransformations:
            self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        else:
            self.final_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.normtype = normtype

        if not use_forward_pass_rootL:
            raise ValueError("Weight Init root L is broken, do not use this option.")

        if not use_forward_pass_rootL and normtype == "postnormpreres":
            raise ValueError("Should probably not be using postnormpreres without use_forward_pass_rootL=True, otherwise initialisation will not be depth invariant.")
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        nn.init.kaiming_normal_(self.output_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.output_proj.bias)
        
        
    def _init_pos_encoding(self):
        position = torch.arange(1024).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, 1024, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
        
    def create_causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        return mask
        
    def forward(self, x):
        seq_len = x.size(1)
        mask = self.create_causal_mask(seq_len).to(x.device)
        
        x = self.input_emb(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x, mask)
        
        if self.normtype != "nonorm":
            x = self.final_norm(x)
        
        output = self.output_proj(x)
        
        return F.log_softmax(output, dim=-1)
