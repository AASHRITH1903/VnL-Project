import torch.nn as nn
import torch


class VideoSummarizer(nn.Module):

    def __init__(self, embed_dim, nhead, internal_dim=128):
        super(VideoSummarizer, self).__init__()

        self.learnable_query = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.query_projection = nn.Linear(embed_dim, internal_dim)
        self.key_projection = nn.Linear(embed_dim, internal_dim)
        self.value_projection = nn.Linear(embed_dim, internal_dim)
        self.out_projection = nn.Linear(internal_dim, embed_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=internal_dim, num_heads=nhead, batch_first=True)

    def forward(self, visual_embedding):
        '''
        Args:
            visual_embedding: [Tensor] (batch_size, seq_len, embedding_dim)
        '''
        summary1 = torch.mean(visual_embedding, dim=1, keepdim=True) # Shape (batch_size, 1, embedding_dim)

        batch_size = visual_embedding.size(0)
        query = self.learnable_query.expand(batch_size, -1, -1)  # Shape (batch_size, 1, embedding_dim)
        q = self.query_projection(query)
        k = self.key_projection(visual_embedding)   
        v = self.value_projection(visual_embedding)  
        attn, _ = self.cross_attention(q, k, v)  # Shape (batch_size, 1, embedding_dim)
        summary2 = self.out_projection(attn)

        out = summary1 + summary2
        return out