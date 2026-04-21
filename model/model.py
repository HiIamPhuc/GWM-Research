import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Structural Component (Entity/Relation Embeddings)
        self.structural_dim = config.structural_dim
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

        # Project structural embeddings to hidden_dim when needed by gating fusion.
        self.structural_projection = None
        if self.structural_dim != config.hidden_dim:
            self.structural_projection = nn.Linear(self.structural_dim, config.hidden_dim)
        
        # 2. Spatial Encoder (GAT for subgraph context)
        # This encodes the K-node subgraph around the head entity into a mental state.
        # The GAT takes fused node embeddings (text + structure) and outputs hidden_dim.
        text_dim = int(getattr(config, 'text_embedding_dim', config.hidden_dim))
        self.gat = GATConv(in_channels=config.hidden_dim, out_channels=config.hidden_dim, heads=1, concat=False)
        
        # State Projectors: Project GAT output to LSTM initial states
        self.h0_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.c0_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 3. Context Processing (RNN / GWM Core)
        # Note: If fusion output is hidden_dim, LSTM input is hidden_dim.
        # NEW: LSTM now receives only [Head_Fused, Relation_Fused] (2 steps instead of 3)
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim, 
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # 4. Fusion Layer
        self.text_projection = nn.Linear(text_dim, config.hidden_dim)
        self.fusion_mode = config.fusion_mode

        # Legacy/default path: concat(text, struct) -> linear
        self.fusion = nn.Linear(text_dim + self.structural_dim, config.hidden_dim)

        # Dynamic gating path: learn sample-wise interpolation between text and structure.
        if self.fusion_mode == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid()
            )

        # Running alpha stats for lightweight diagnostics.
        self.reset_alpha_stats()
        
        # 5. Output Projector (Optional but good for matching embeddings)
        self.projector = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Precomputed text cache loaded from preprocessing artifacts.
        self.cached_entity_text_emb = None
        self.cached_relation_text_emb = None
        self.use_text_cache = False

    def _encode_subgraph_with_gat(self, context_ids, ctx_fused):
        """
        Encode subgraph context using Graph Attention Network.
        
        Args:
            context_ids: (B, K) tensor of context node IDs
            ctx_fused: (B, K, H) fused context embeddings
        
        Returns:
            gat_output: (B, H) aggregated subgraph representation
        """
        B, K, H = ctx_fused.shape
        
        # Create fully connected subgraph (all K nodes connect to each other)
        # This treats the K-node subgraph as a complete graph
        edge_index = []
        for i in range(K):
            for j in range(K):
                if i != j:
                    edge_index.append([i, j])
        
        if len(edge_index) == 0:
            # If K=1, no edges; just aggregate the single node
            return torch.mean(ctx_fused, dim=1)  # (B, H)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=ctx_fused.device).t().contiguous()
        
        # Process each batch item separately
        gat_outputs = []
        for b in range(B):
            # Create a graph data object for this batch item
            graph = Data(x=ctx_fused[b], edge_index=edge_index)  # (K, H) nodes
            
            # Forward through GAT
            node_out = self.gat(graph.x, graph.edge_index)  # (K, H)
            
            # Global mean pooling over nodes to get graph-level representation
            graph_rep = torch.mean(node_out, dim=0)  # (H,)
            gat_outputs.append(graph_rep)
        
        return torch.stack(gat_outputs, dim=0)  # (B, H)
 
    def _load_embedding_tensor(self, source, expected_rows, name):
        if isinstance(source, str):
            loaded = torch.load(source, map_location='cpu')
        elif torch.is_tensor(source):
            loaded = source.detach().cpu()
        else:
            raise TypeError(f"Unsupported {name} cache source: {type(source)}")

        if isinstance(loaded, dict):
            if 'embeddings' in loaded:
                loaded = loaded['embeddings']
            elif 'tensor' in loaded:
                loaded = loaded['tensor']
            else:
                raise ValueError(f"{name} cache dict must contain 'embeddings' or 'tensor'.")

        if not torch.is_tensor(loaded):
            raise TypeError(f"{name} cache must resolve to a torch.Tensor.")

        loaded = loaded.float().contiguous()
        if loaded.dim() != 2:
            raise ValueError(f"{name} cache must be rank-2. Got shape {tuple(loaded.shape)}")
        if loaded.size(0) != expected_rows:
            raise ValueError(
                f"{name} cache row count mismatch. Expected {expected_rows}, got {loaded.size(0)}"
            )
        return loaded

    def load_precomputed_text_embedding_cache(self, entity_source, relation_source, cache_device='cpu'):
        if self.use_text_cache and self.cached_entity_text_emb is not None and self.cached_relation_text_emb is not None:
            return

        cache_device = torch.device(cache_device)

        entity_cache = self._load_embedding_tensor(
            source=entity_source,
            expected_rows=self.entity_embeddings.num_embeddings,
            name='entity',
        ).to(cache_device)

        relation_cache = self._load_embedding_tensor(
            source=relation_source,
            expected_rows=self.relation_embeddings.num_embeddings,
            name='relation',
        ).to(cache_device)

        if entity_cache.size(1) != relation_cache.size(1):
            raise ValueError(
                "Entity and relation text embeddings must share the same embedding dimension. "
                f"Got {entity_cache.size(1)} and {relation_cache.size(1)}"
            )

        expected_text_dim = self.text_projection.in_features
        if entity_cache.size(1) != expected_text_dim:
            raise ValueError(
                "Text embedding dimension mismatch with model config. "
                f"Expected {expected_text_dim}, got {entity_cache.size(1)}"
            )

        self.cached_entity_text_emb = entity_cache
        self.cached_relation_text_emb = relation_cache
        self.use_text_cache = True

        if cache_device.type == 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Backward-compatible alias.
    def build_text_embedding_cache(self, entity_source, relation_source, device='cpu', **_kwargs):
        self.load_precomputed_text_embedding_cache(entity_source, relation_source, cache_device=device)

    def _lookup_cached_text(self, ids, kind='entity'):
        cache = self.cached_entity_text_emb if kind == 'entity' else self.cached_relation_text_emb
        if cache is None:
            raise RuntimeError("Text cache is not built. Call load_precomputed_text_embedding_cache first.")

        original_shape = ids.shape
        flat_ids = ids.reshape(-1)
        if flat_ids.device != cache.device:
            flat_ids = flat_ids.to(cache.device)
        selected = cache.index_select(0, flat_ids)
        if selected.device != ids.device:
            selected = selected.to(ids.device)
        return selected.reshape(*original_shape, -1)

    def _project_structural(self, struct_emb):
        if self.structural_projection is not None:
            return self.structural_projection(struct_emb)
        return struct_emb

    def reset_alpha_stats(self):
        self._alpha_sum = 0.0
        self._alpha_count = 0

    def get_alpha_mean(self, reset=False):
        if self.fusion_mode != 'gated' or self._alpha_count == 0:
            alpha_mean = None
        else:
            alpha_mean = self._alpha_sum / self._alpha_count

        if reset:
            self.reset_alpha_stats()

        return alpha_mean

    def _fuse_modalities(self, text_emb, struct_emb):
        if self.fusion_mode == 'gated':
            text_proj = self.text_projection(text_emb)
            struct_proj = self._project_structural(struct_emb)
            gate_input = torch.cat([text_proj, struct_proj], dim=-1)
            alpha = self.gate(gate_input)
            alpha_detached = alpha.detach()
            self._alpha_sum += alpha_detached.sum().item()
            self._alpha_count += alpha_detached.numel()
            return alpha * text_proj + (1.0 - alpha) * struct_proj

        # Backward-compatible concat fusion
        return self.fusion(torch.cat([text_emb, struct_emb], dim=-1))
        
    def forward(self, h_batch, r_batch, context_batch):
        """
        Forward pass for a batch of triples.
        h_batch: dict {id}
        r_batch: dict {id}
        context_batch: dict {id}
          - id: (B, K)
        
        Ha & Schmidhuber Paradigm:
        - Encode subgraph context with GAT -> (B, H)
        - Project GAT output to LSTM initial states (h_0, c_0)
        - Run LSTM on [Head_Fused, Relation_Fused] with those initial states
        - Use final LSTM hidden state as query vector
        """
        if not self.use_text_cache:
            raise RuntimeError(
                "Text cache is not built. Call load_precomputed_text_embedding_cache before training/inference."
            )

        h_emb_text = self._lookup_cached_text(h_batch['id'], kind='entity')
        r_emb_text = self._lookup_cached_text(r_batch['id'], kind='relation')
        
        # Structural Embeddings
        h_struct = self.entity_embeddings(h_batch['id']) # (B, H)
        r_struct = self.relation_embeddings(r_batch['id']) # (B, H)
        
        # Context
        context_ids = context_batch['id'] # (B, K)
        ctx_emb_text = self._lookup_cached_text(context_ids, kind='entity') # (B, K, H)
        ctx_struct = self.entity_embeddings(context_ids) # (B, K, H)

        # Fuse Context (Text + Structure)
        ctx_fused = self._fuse_modalities(ctx_emb_text, ctx_struct) # (B, K, H)
        
        # SPATIAL ENCODER: Encode subgraph using GAT
        subgraph_rep = self._encode_subgraph_with_gat(context_ids, ctx_fused)  # (B, H)
        
        # Project to LSTM initial states
        h_0_init = torch.tanh(self.h0_projection(subgraph_rep))  # (B, H)
        c_0_init = torch.tanh(self.c0_projection(subgraph_rep))  # (B, H)
        # For multi-layer LSTM, replicate states across layers
        if self.config.num_layers > 1:
            h_0 = h_0_init.unsqueeze(0).repeat(self.config.num_layers, 1, 1)  # (num_layers, B, H)
            c_0 = c_0_init.unsqueeze(0).repeat(self.config.num_layers, 1, 1)  # (num_layers, B, H)
        else:
            h_0 = h_0_init.unsqueeze(0)  # (1, B, H)
            c_0 = c_0_init.unsqueeze(0)  # (1, B, H)
        
        # Main Fusion
        h_fused = self._fuse_modalities(h_emb_text, h_struct) # (B, H)
        r_fused = self._fuse_modalities(r_emb_text, r_struct) # (B, H)

        # TRAJECTORY: Run LSTM with initial state from GAT-encoded subgraph
        # Sequence: [Head_Fused, Relation_Fused] initialized with (h_0, c_0) from subgraph
        lstm_input = torch.stack([h_fused, r_fused], dim=1) # (B, 2, H)
        
        lstm_out, (h_n, c_n) = self.lstm(lstm_input, (h_0, c_0))
        query_vector = lstm_out[:, -1, :] # Last hidden state (B, H)
        
        # Project Query
        query_vector = self.projector(query_vector)
        
        # Ensure normalization for cosine similarity / InfoNCE
        query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)
        
        return query_vector

    def encode_target(self, t_batch):
        """
        Encode target/tail entities symmetrically (Fusion of Text + Structure).
        t_batch: dict {id}
        Returns: (B, H) normalized fused embedding
        """
        if not self.use_text_cache:
            raise RuntimeError(
                "Text cache is not built. Call load_precomputed_text_embedding_cache before training/inference."
            )

        t_emb_text = self._lookup_cached_text(t_batch['id'], kind='entity')
        t_struct = self.entity_embeddings(t_batch['id'])
        
        t_fused = self._fuse_modalities(t_emb_text, t_struct)
        
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
