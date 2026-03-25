import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Text Encoder (Finetunable)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.text_encoder = AutoModel.from_pretrained(config.pretrained_model)
        
        # Freezing Logic
        if not config.finetune_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
        # 2. Structural Component (Entity/Relation Embeddings)
        self.entity_embeddings = nn.Embedding(config.num_entities, config.hidden_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, config.hidden_dim)
        
        # 3. Context Processing (RNN / GWM Core)
        # Note: If fusion output is hidden_dim, LSTM input is hidden_dim.
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim, 
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # 4. Fusion Layer
        self.fusion = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        # 5. Output Projector (Optional but good for matching embeddings)
        self.projector = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def _encode_text(self, input_ids, attention_mask):
        """Forward pass through BERT."""
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :] # [CLS] token
        
    def forward(self, h_batch, r_batch, context_batch):
        """
        Forward pass for a batch of triples.
        h_batch: dict {input_ids, attention_mask, id}
        r_batch: dict {input_ids, attention_mask, id}
        context_batch: dict {input_ids, attention_mask, id}
          - input_ids: (B*K, L)
          - id: (B, K)
        """
        # Encode Head & Relation Text
        h_emb_text = self._encode_text(h_batch['input_ids'], h_batch['attention_mask'])
        r_emb_text = self._encode_text(r_batch['input_ids'], r_batch['attention_mask'])
        
        # Structural Embeddings
        h_struct = self.entity_embeddings(h_batch['id']) # (B, H)
        r_struct = self.relation_embeddings(r_batch['id']) # (B, H)
        
        # Context
        context_ids = context_batch['id'] # (B, K)
        ctx_input_ids = context_batch['input_ids'] 
        ctx_mask = context_batch['attention_mask']

        ctx_emb_text = self._encode_text(ctx_input_ids, ctx_mask) # (B*K, H)
        ctx_struct = self.entity_embeddings(context_ids) # (B, K, H)
        batch_size, k = context_ids.shape
        ctx_emb_text = ctx_emb_text.view(batch_size, k, -1)

        # Fuse Context (Text + Structure)
        ctx_fused_input = torch.cat([ctx_emb_text, ctx_struct], dim=-1)
        ctx_fused = self.fusion(ctx_fused_input) # (B, K, H)
        # Aggregate Context
        ctx_summary = torch.mean(ctx_fused, dim=1) # (B, H)
        
        # Main Fusion
        h_fused = self.fusion(torch.cat([h_emb_text, h_struct], dim=-1)) # (B, H)
        r_fused = self.fusion(torch.cat([r_emb_text, r_struct], dim=-1)) # (B, H)

        # LSTM Context Aggregation
        # Sequence: [Context, Head, Relation] -> Predict Tail
        lstm_input = torch.stack([ctx_summary, h_fused, r_fused], dim=1) # (B, 3, H)
        
        lstm_out, _ = self.lstm(lstm_input)
        query_vector = lstm_out[:, -1, :] # Last hidden state (B, H)
        
        # Project Query
        query_vector = self.projector(query_vector)
        
        # Ensure normalization for cosine similarity / InfoNCE
        query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)
        
        return query_vector

    def encode_target(self, t_batch):
        """
        Encode target/tail entities symmetrically (Fusion of Text + Structure).
        t_batch: dict {input_ids, attention_mask, id}
        Returns: (B, H) normalized fused embedding
        """
        t_emb_text = self._encode_text(t_batch['input_ids'], t_batch['attention_mask'])
        t_struct = self.entity_embeddings(t_batch['id'])
        
        t_fused = self.fusion(torch.cat([t_emb_text, t_struct], dim=-1))
        
        # Apply same projection if desired, or skip. Usually targets are raw fused.
        # But for symmetric comparison, we might want projector too?
        # Standard SimKGC: Only Query has pooling/projection. Keys areraw.
        # But this is GWM, let's keep it symmetric if we want "Symmetric". 
        # Actually usually Keys are just the representation.
        # Let's verify Projector usage. LSTM out -> Project -> Query.
        # Tail -> Fuse -> Target.
        # This is standard.
        
        return torch.nn.functional.normalize(t_fused, p=2, dim=1)

    def compute_loss(self, query_vector, t_fused):
        """
        InfoNCE Loss with In-Batch Negatives.
        query_vector: (B, H) - Normalized query embeddings
        t_fused: (B, H) - Normalized target/tail embeddings (Symmetric Fusion)
        """
        # Cosine Similarity
        # (B, B)
        # score[i, j] = sim(query[i], tail[j])
        scores = torch.mm(query_vector, t_fused.t())
        
        # Temperature
        if hasattr(self.config, 'temperature'):
            scores /= self.config.temperature
        else:
            scores /= 0.07
        
        # Labels: diagonal are positives
        labels = torch.arange(scores.size(0), device=scores.device)
        
        return nn.CrossEntropyLoss()(scores, labels), scores
