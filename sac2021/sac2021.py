import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sac2021.const import max_n_atoms, num_unique_atoms

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, idx, res_p=0.0, att_p=0.0, bias=True, scale=True):
        super(MultiHeadAttention, self).__init__()

        d_head = d_model // n_heads
        self.n_heads, self.d_head = n_heads, d_head
        self.bias, self.scale = bias, scale
        
        self.gamma_p = nn.Parameter(torch.ones([n_heads]))
        self.gamma_sym = nn.Parameter(torch.ones([n_heads]))
        self.gamma_adj = nn.Parameter(torch.ones([n_heads]))

        self.w_bias = nn.Linear(2, n_heads, bias=False)

        self.att = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)  # LinearNoBias for attention
        # self.gate = nn.Sequential(nn.Linear(d_model, n_heads * d_head), nn.Sigmoid())
        self.ff = nn.Linear(n_heads * d_head, d_model, bias=bias)

        self.drop_att, self.drop_res = nn.Dropout(att_p), nn.Dropout(res_p)
        self.ln = nn.LayerNorm(d_model)

        self.idx = idx

    def forward(self, x, pdist, angle, adj, mask=None):
    # def forward(self, x, pdist, sym, angle, adj, mask=None):
        ff_out = self.ff(self._attention(x, pdist, angle, adj, mask=mask))
        # ff_out = self.ff(self._attention(x, pdist, sym, angle, adj, mask=mask))
        return self.ln(x + self.drop_res(ff_out))

    # def _attention(self, x, pdist, sym, angle, adj, mask):
    def _attention(self, x, pdist, angle, adj, mask):
        # x : bsz x n_atoms x d_model
        bsz, n_atoms = x.size(0), x.size(1)
        # self.att(x) : bsz x n_atoms x (3 * n_heads * d_head)
        wq, wk, wv = torch.chunk(self.att(x), 3, dim=-1) 
        # --> wq, wk, wv : bsz x n_atoms x (n_heads * d_head)
        wq, wk, wv = map(lambda x: x.view(bsz, n_atoms, self.n_heads, self.d_head), (wq, wk, wv))
        # --> wq, wk, wv : bsz x n_atoms x n_heads x d_head
        wq, wk, wv = wq.permute(0, 2, 1, 3), wk.permute(0, 2, 3, 1), wv.permute(0, 2, 1, 3)
        # --> wq : bsz x n_heads x n_atoms x d_head
        # --> wk : bsz x n_heads x d_head x n_atoms
        # --> wv : bsz x n_heads x n_atoms x d_head
        att_score = torch.matmul(wq, wk)
        # --> att_score : bsz x n_heads x n_atoms x n_atoms
        if self.scale:
            att_score.div_(self.d_head ** 0.5)

        # Penalize by pairwise distance
        # pdist : bsz x 2 x n_atoms x n_atoms
        # --> expand to bsz x n_heads x n_atoms x n_atoms
        # --> also expand gamma to have shape bsz x n_heads x n_atoms x n_atoms

        pdist = pdist.unsqueeze(1).expand(bsz, self.n_heads, n_atoms, n_atoms)
        gamma_p = self.gamma_p.unsqueeze(1).unsqueeze(2).expand(self.n_heads, n_atoms, n_atoms).unsqueeze(0).expand(bsz, self.n_heads, n_atoms, n_atoms)
        att_score -= gamma_p * pdist

        # sym = sym.unsqueeze(1).expand(bsz, self.n_heads, n_atoms, n_atoms)
        # gamma_sym = self.gamma_sym.unsqueeze(1).unsqueeze(2).expand(self.n_heads, n_atoms, n_atoms).unsqueeze(0).expand(bsz, self.n_heads, n_atoms, n_atoms)
        # att_score -= gamma_sym * sym

        angle = angle.view(-1, 2) # (bsz * n_atoms * n_atoms) x 2
        angle = self.w_bias(angle) # (bsz * n_atoms * n_atoms) x n_heads
        angle = angle.view(bsz, n_atoms, n_atoms, self.n_heads).permute(0, 3, 1, 2)
        att_score += angle

        adj = adj.unsqueeze(1).expand(bsz, self.n_heads, n_atoms, n_atoms)
        gamma_adj = self.gamma_adj.unsqueeze(1).unsqueeze(2).expand(self.n_heads, n_atoms, n_atoms).unsqueeze(0).expand(bsz, self.n_heads, n_atoms, n_atoms)
        att_score += gamma_adj * adj

        if mask is not None:
            minus_inf = -65504 if att_score.dtype == torch.float16 else -1e9
            att_score = att_score.masked_fill(mask, minus_inf).type_as(att_score)
        att_prob = self.drop_att(F.softmax(att_score, dim=-1))
        # --> att_prob : bsz x n_heads x n_atoms x n_atoms

        att_vec = torch.matmul(att_prob, wv)
        # --> att_vec : bsz x n_heads x n_atoms x d_head
        att_vec = att_vec.permute(0, 2, 1, 3).contiguous().contiguous().view(bsz, n_atoms, -1)
        # --> att_vec : bsz x n_atoms x (h_heads * d_head)

        return att_vec
        # return att_vec * self.gate(att_vec)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(x + self.l2(F.relu(self.l1(x))))

class AttentionBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, idx, res_p=0.0, att_p=0.0, bias=True, scale=True):
        super(AttentionBlock, self).__init__()
        self.self_att = MultiHeadAttention(n_heads, d_model, idx, res_p, att_p, bias, scale)
        self.ff = FeedForward(d_model, d_ff)
    
    # def forward(self, x, pdist, sym, angle, adj, mask=None):
    def forward(self, x, pdist, angle, adj, mask=None):
        # return self.ff(self.self_att(x, pdist, sym, angle, adj, mask))
        return self.ff(self.self_att(x, pdist, angle, adj, mask))

class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, res_p=0.0, att_p=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            AttentionBlock(n_heads, d_model, d_ff, idx=i, res_p=res_p, att_p=att_p) for i in range(n_layers)
        ])

    # def forward(self, x, pdist, sym, angle, adj, mask):
    def forward(self, x, pdist, angle, adj, mask):
        for layer in self.layers:
            # x = layer(x, pdist, sym, angle, adj, mask=mask)
            x = layer(x, pdist, angle, adj, mask=mask)
        return x

class AtomTransformer(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, res_p=0.0, att_p=0.0):
        super(AtomTransformer, self).__init__()
        self.d_model = d_model
        self.transformer = Transformer(n_layers, n_heads, d_model, d_ff, res_p, att_p)

        # atom_embedding_size = d_model - 17
        hyb_embedding_size = 64
        donac_embedding_size = 16
        spin_embedding_size = 16
        atom_embedding_size = d_model - 64 - 16 - 16 - 1 - 1 - 1 - 1 - 2
        assert atom_embedding_size > 0, f'Atom embedding size should be > 0. Now {atom_embedding_size}'
        # 1 for aromaticity
        # 1 for formal charge
        # 1 for totalnumH
        # 1 for total valence
        # 2 for molecule features (NPR1, NPR2)
        self.atom_embedding = nn.Embedding(num_unique_atoms + 1 + 1, atom_embedding_size) # +1 for dummy node
        self.hyb_embedding = nn.Linear(7, hyb_embedding_size, bias=False)
        self.donac_embedding = nn.Linear(2, donac_embedding_size, bias=False)
        self.spin_embedding = nn.Linear(2, spin_embedding_size, bias=False)

        n_head_feature = d_model * 3
        self.gap_head = nn.Sequential(
            # nn.Linear(n_head_feature, 256),
            nn.Linear(n_head_feature, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.s1_head = nn.Sequential(
            nn.Linear(n_head_feature, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.t1_head = nn.Sequential(
            nn.Linear(n_head_feature, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    # def forward(self, atom_idx, hyb, donac, spin, feat, fp, pdist, sym, angle, adj, mask, out_mask, n_atoms):
    def forward(self, atom_idx, hyb, donac, spin, feat, pdist, angle, adj, mask, out_mask, n_atoms):
        a = self.atom_embedding(atom_idx + 1)
        hyb = self.hyb_embedding(hyb)
        donac = self.donac_embedding(donac)
        spin = self.spin_embedding(spin)

        x = torch.cat([feat, a, hyb, donac, spin], dim=-1)
        
        # x : bsz x max_n_atoms x model
        # x = self.transformer(x, pdist, sym, angle, adj, mask)
        x = self.transformer(x, pdist, angle, adj, mask)

        # Take mean and max only for where atom exists.
        _out_mask = out_mask.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2))
        mean_masked_x = x *_out_mask
        max_masked_x = x - (~_out_mask).float() * 10000
        min_masked_x = x + (~_out_mask).float() * 10000

        x_mean = mean_masked_x.sum(axis=1) / n_atoms.view(x.size(0), 1)
        x_max = max_masked_x.max(axis=1).values
        x_min = min_masked_x.min(axis=1).values

        x = torch.cat([x_mean, x_max, x_min], dim=-1)

        return self.gap_head(x), self.s1_head(x), self.t1_head(x)

class AtomTransformerPretrain(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, res_p=0.0, att_p=0.0):
        super(AtomTransformerPretrain, self).__init__()
        self.d_model = d_model
        self.transformer = Transformer(n_layers, n_heads, d_model, d_ff, res_p, att_p)

        # atom_embedding_size = d_model - 17
        hyb_embedding_size = 64
        donac_embedding_size = 16
        spin_embedding_size = 16
        atom_embedding_size = d_model - 64 - 16 - 16 - 1 - 1 - 1 - 1 - 2
        assert atom_embedding_size > 0, f'Atom embedding size should be > 0. Now {atom_embedding_size}'
        # 1 for aromaticity
        # 1 for formal charge
        # 1 for totalnumH
        # 1 for total valence
        # 2 for molecule features (NPR1, NPR2)
        self.atom_embedding = nn.Embedding(num_unique_atoms + 1 + 1, atom_embedding_size) # +1 for dummy node
        self.hyb_embedding = nn.Linear(7, hyb_embedding_size, bias=False)
        self.donac_embedding = nn.Linear(2, donac_embedding_size, bias=False)
        self.spin_embedding = nn.Linear(2, spin_embedding_size, bias=False)

        n_head_feature = d_model * 3

        self.homo_head = nn.Sequential(
            nn.Linear(n_head_feature, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.lumo_head = nn.Sequential(
            nn.Linear(n_head_feature, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    # def forward(self, atom_idx, hyb, donac, spin, feat, pdist, sym, angle, adj, mask, out_mask, n_atoms):
    def forward(self, atom_idx, hyb, donac, spin, pdist, angle, adj, mask, out_mask, n_atoms):
        a = self.atom_embedding(atom_idx + 1)
        hyb = self.hyb_embedding(hyb)
        donac = self.donac_embedding(donac)
        spin = self.spin_embedding(spin)

        # x = torch.cat([feat, a, hyb, donac, spin], dim=-1)
        x = torch.cat([a, hyb, donac, spin], dim=-1)
        
        # x : bsz x max_n_atoms x model
        # x = self.transformer(x, pdist, sym, angle, adj, mask)
        x = self.transformer(x, pdist, angle, adj, mask)

        # Take mean and max only for where atom exists.
        _out_mask = out_mask.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2))
        mean_masked_x = x *_out_mask
        max_masked_x = x - (~_out_mask).float() * 10000
        min_masked_x = x + (~_out_mask).float() * 10000

        x_mean = mean_masked_x.sum(axis=1) / n_atoms.view(x.size(0), 1)
        x_max = max_masked_x.max(axis=1).values
        x_min = min_masked_x.min(axis=1).values

        x = torch.cat([x_mean, x_max, x_min], dim=-1)
        return self.homo_head(x), self.lumo_head(x)

if __name__ == '__main__':
    bsz, n_atoms, d_model = 16, max_n_atoms, 128
    mask = torch.randint(0, 1, [bsz, n_atoms, n_atoms]).bool().unsqueeze(1) # bsz x 1 x n_atoms x n_atoms
    out_mask = torch.randint(0, 1, [bsz, n_atoms])
    num_atoms = torch.randint(1, 10, [bsz])
    pdist = torch.randn([bsz, n_atoms, n_atoms])
    adj = torch.randn([bsz, n_atoms, n_atoms])
    atom_idx = torch.randint(-1, num_unique_atoms - 1, [bsz, n_atoms])
    feat = torch.randn([bsz, n_atoms, 8])

    xyz = torch.randn([bsz, n_atoms, 3])

    att = AtomTransformer(n_layers=6, n_heads=8, d_model=128, d_ff=1024)
    out = att(xyz, atom_idx, feat, pdist, adj, mask, out_mask, num_atoms)
    print(out)

    print('Model size')
    print(sum(param.numel() for param in att.parameters() if param.requires_grad))


    pass
