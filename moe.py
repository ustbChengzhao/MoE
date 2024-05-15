from torch import nn
from torch import softmax
import torch

class Expert(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
    
    def forward(self, x):
        return self.seq(x)
    

class MoE(nn.Module):
    def __init__(self, experts: int, 
                 topk: int, 
                 emb_size: int, 
                 w_importance: int = 0.01):
        super().__init__()
        self.experts = nn.ModuleList([Expert(emb_size) for _ in range(experts)])
        self.topk = topk
        self.gate = nn.Linear(emb_size, experts)
        self.noise = nn.Linear(emb_size, experts)
        self.w_importance = w_importance
        
    def forward(self, x):
        x_shape = x.shape   # [batch, seq_len, emb_size]
        x = x.reshape(-1, x_shape[-1])  # [batch*seq_len, emb_size]
        
        # gate
        gate_logits = self.gate(x)  # [batch*seq_len, experts]
        gete_prob = softmax(gate_logits, dim=-1)  # [batch*seq_len, experts]
        
        # Noisy Top-K Gating
        if self.training:
            noise = torch.randn_like(gate_prob) * nn.functional.softplus(self.noise(x))
            gate_prob = gate_prob + noise
        
        # top expert
        top_weights, top_index = torch.topk(gate_prob, k=self.topk, dim=-1) 
        # top_weights: [batch*seq_len, topk], top_index: [batch*seq_len, topk]
        top_weights = softmax(top_weights, dim=-1)  # [batch*seq_len, topk]
        
        top_weights = top_weights.view(-1) # [batch*seq_len*topk]
        top_index = top_index.view(-1)  # [batch_seq_len*topk]
        
        # expert
        for expert_i, expert in enumerate(self.experts):
            x_expert = x[top_index == expert_i]  # [..., emb_size]
            y_expert = expert(x_expert)  # [..., emb_size]
            
            add_index = (top_index == expert_i).nonzero().flatten()  # [...]
            y = y.index_add(dim=0, index=add_index, source=y_expert)
            
        top_weights = top_weights.view(-1, 1).expand(-1, x.shape[-1])  # [batch*seq_len*topk, emb_size] 
        y = y * top_weights
        y = y.view(-1, self.topk, x.size(-1)) # [batch*seq_len, topk, emb_size]
        y = y.sum(dim=1) # [batch*seq_len, emb_size]
        
        # 负载均衡
        if self.training:
            importance = gate_prob.sum(dim=0)  # [experts] 每个expert的重要性
            importance_loss = self.w_importance * (torch.std(importance) / torch.mean(importance)) ** 2
        else:
            importance_loss = None
        return y.view(x_shape), gate_prob, importance_loss
    
class MNIST_MoE(nn.Module):
    def __init__(self, input_size, experts, topk, emb_size):
        super().__init__()
        self.moe = MoE(experts, topk, emb_size)
        self.emb = nn.Linear(input_size, emb_size)
        self.cls = nn.Linear(emb_size, 10)
    
    def forward(self, x):   # [batch, 28, 28]
        x = x.view(-1, 28*28)  # [batch, 784]
        y = self.emb(x)
        y, gate_prob, importance_loss = self.moe(y)
        return self.cls(y), gate_prob, importance_loss
        