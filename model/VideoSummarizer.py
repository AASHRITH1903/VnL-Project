import torch.nn as nn
import torch


class VideoSummarizer(nn.Module):

    def __init__(self, d_model, nhead, internal_dim=128):
        super(VideoSummarizer, self).__init__()

        self.learnable_query = nn.Parameter(torch.zeros(1, 1, d_model))

        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, 
                                                     kdim=internal_dim, vdim=internal_dim)

    def forward(self, visual_embedding):
        '''
        Args:
            visual_embedding: [Tensor] (batch_size, seq_len, embedding_dim)
        '''
        summary1 = torch.mean(visual_embedding, dim=1, keepdim=True) # Shape (batch_size, 1, embedding_dim)

        batch_size = visual_embedding.size(0)
        query = self.learnable_query.expand(batch_size, -1, -1)  # Shape (batch_size, 1, embedding_dim)
        summary2, _ = self.cross_attention(query, visual_embedding, visual_embedding)  # Shape (batch_size, 1, embedding_dim)

        out = summary1 + summary2

        return out