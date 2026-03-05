"""
REFLEX: Enhanced Protein-Protein Interaction Prediction with Hierarchical Fusion

This model introduces a novel hierarchical fusion architecture for PPI prediction,
combining graph-based protein representations with multi-stage feature integration.

Core Architecture:
1. Pretrained Protein Embedder: Pre-configured VQ-VAE codebook + ESM-2 (in ppi_data.py)
2. Adaptive Hyperbolic Projector: Lorentz manifold projection with multi-view GCN
3. Hierarchical Attribute Extractor (HAE): Three-stage gating + adaptive residual gating
4. Auxiliary Generation Regularization: VAE-based sequence generation as regularizer

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PPIGEN import (
    HIPPI_VAE_Latent as LSE_VAE_Latent,
    TransformerEncoder,
    PositionalEncoding,
    Decoder,
    HyperbolicGCN,
    HyperbolicDecoder,
)
import src.mainfold
import torch_geometric
from src.base.model import BaseModel


# =====================================
# Module (b): Adaptive Hyperbolic Projector
# =====================================

class AdaptiveHyperbolicProjector(nn.Module):
    """
    Adaptive Hyperbolic Projector
    
    Projects PPI network onto a Lorentz hyperbolic manifold, preserving the
    inherent hierarchy through multi-view graph convolution and tangent space
    refinement. Uses a learnable scalar parameter to dynamically adjust 
    the magnitude of feature vectors in the tangent space.
    
    Architecture per view:
    1. Hyperbolic GCN: Aggregates structural information on the manifold
    2. Tangent Projection: Maps features back to Euclidean space via log map
    3. GIN Refinement: Captures local isomorphism features in the tangent plane
    
    Final representation concatenates scale-adjusted initial embeddings
    with all view-specific outputs.
    
    Dual-Modal Input:
        - embed1: VQ-VAE structure encoding [N, 512]
        - encode1: ESM2 sequence encoding [N, 1280]
    """
    
    def __init__(
        self,
        input_dim,
        esm_dim=1280,
        args=None,
        act='relu',
        layer_num=2,
        radius=None,
        dropout=0.0,
        if_bias=True,
        use_att=0,
        local_agg=0,
        class_num=7,
        in_len=512,
        device=None,
    ):
        super(AdaptiveHyperbolicProjector, self).__init__()
        self.models = nn.ModuleList()
        self.layer_num = layer_num
        self.class_num = class_num
        self.in_len = in_len
        self.input_dim = input_dim
        self.esm_dim = esm_dim
        self.hyper_dim = int(self.input_dim / 2)
        
        # Dual-modal fusion: VQ-VAE (512) + ESM2 (1280) -> input_dim (512)
        self.dual_modal_fusion = nn.Sequential(
            nn.Linear(input_dim + esm_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim)
        )
        
        # Learnable tangent space scaling factor
        self.tangent_scale = nn.Parameter(torch.ones(1))
        
        self.manifold = src.mainfold.Hyperboloid()
        self.device = device

        dims = [self.input_dim] + ([self.hyper_dim] * layer_num)
        if self.manifold.name == 'Hyperboloid':
            dims[0] += 1

        n_curvatures = len(dims) + 1
        self.radius = radius
        if radius is None:
            self.curvatures = nn.ParameterList([
                nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)
            ])
        else:
            self.curvatures = [torch.tensor([radius]).to(device) for _ in range(n_curvatures)]

        act_fn = getattr(torch.nn.functional, act)
        acts = [act_fn] * layer_num

        for c in range(class_num):
            graph_layers = []
            i = 0
            c_in, c_out = self.curvatures[i + 1], self.curvatures[i + 2]
            in_dim, out_dim = dims[i], dims[i + 1]
            graph_layers.append(
                HyperbolicGCN(
                    self.manifold, in_dim, out_dim, c_in, c_out,
                    dropout, acts[i], if_bias, use_att, local_agg
                )
            )
            graph_layers.append(
                HyperbolicDecoder(
                    dims[-2], dims[-1],
                    if_bias, dropout, self.curvatures[-1]
                )
            )
            graph_layers.append(
                torch_geometric.nn.models.GIN(
                    dims[-1], dims[-1], 1, out_dim,
                    act='tanh', norm=nn.BatchNorm1d(dims[-1])
                )
            )
            self.models.append(nn.Sequential(*graph_layers))

        self.output_dim = dims[0] + class_num * dims[-1]

    def forward(self, data):
        # Dual-modal features
        struct_feat = data.embed1   # VQ-VAE structure: [N, 512]
        seq_feat = data.encode1     # ESM2 sequence: [N, 1280]
        
        # Fuse dual modalities
        # f1 = self.dual_modal_fusion(torch.cat([struct_feat, seq_feat], dim=-1))  # [N, 512]
        f1 = struct_feat
        
        sparse_adj = data.sparse_adj1
        edges = data.edge1

        o = torch.zeros_like(f1)
        f1 = torch.cat([o[:, 0:1], f1], dim=1)

        output = [f1]

        # Tangent space projection with learnable scaling
        x_tan = self.manifold.proj_tan0(f1, self.curvatures[0])
        x_tan = x_tan * self.tangent_scale  # Learnable tangent scaling
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        for i, m in enumerate(self.models):
            tmp = x_hyp
            tmp, _ = m[0]((tmp, sparse_adj[i]))
            tmp = m[1].forward(tmp)
            tmp = m[2](tmp, edges[i])
            output.append(tmp)

        x = torch.cat(output, dim=1)
        return x


# =====================================
# Internal: Three-Stage Gating (building block of HAE)
# =====================================

class ThreeStageGating(nn.Module):
    """
    Three-Stage Hierarchical Gating Architecture
    
    Internal building block of the Hierarchical Attribute Extractor (HAE).
    Progressively integrates features from coarse to fine granularity using
    Gated Transformation Blocks.
    
    Architecture:
    - Stage I (Coarse): Captures first-order pairwise interactions via Hadamard product
    - Stage II (Intermediate): Refines features through residual gated block
    - Stage III (Fine): Projects to output space with full expressiveness
    
    Args:
        protein_dim (int): Dimension of input protein embeddings (d_h)
        intermediate_dim (int): Hidden dimension (d_h/2)
        dropout (float): Dropout probability
    """
    def __init__(self, protein_dim, intermediate_dim, dropout=0.1):
        super().__init__()
        
        # Stage 1: Coarse-level interaction capture
        self.coarse_gate = nn.Sequential(
            nn.Linear(protein_dim * 2, intermediate_dim),
            nn.Sigmoid()
        )
        self.coarse_transform = nn.Sequential(
            nn.Linear(protein_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm_coarse = nn.LayerNorm(intermediate_dim)
        
        # Stage 2: Intermediate-level feature refinement
        self.intermediate_gate = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.Sigmoid()
        )
        self.intermediate_transform = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm_intermediate = nn.LayerNorm(intermediate_dim)
        
        # Stage 3: Fine-level feature integration
        self.fine_gate = nn.Sequential(
            nn.Linear(intermediate_dim, protein_dim),
            nn.Sigmoid()
        )
        self.fine_transform = nn.Sequential(
            nn.Linear(intermediate_dim, protein_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm_fine = nn.LayerNorm(protein_dim)
        
    def forward(self, protein1, protein2):
        """
        Forward pass through hierarchical fusion stages
        
        Args:
            protein1, protein2: [batch_size, protein_dim]
            
        Returns:
            fused_features: [batch_size, protein_dim]
        """
        # Stage 1: Coarse-level fusion
        # Compute interaction via element-wise multiplication
        pairwise_interaction = protein1 * protein2
        concatenated_input = torch.cat([protein1, protein2], dim=-1)
        
        # Apply gated transformation
        gate_coarse = self.coarse_gate(concatenated_input)
        features_coarse = self.coarse_transform(pairwise_interaction)
        stage1_output = self.norm_coarse(gate_coarse * features_coarse)
        
        # Stage 2: Intermediate-level refinement with residual
        gate_intermediate = self.intermediate_gate(stage1_output)
        features_intermediate = self.intermediate_transform(stage1_output)
        # Residual connection for information preservation
        stage2_output = self.norm_intermediate(stage1_output + gate_intermediate * features_intermediate)
        
        # Stage 3: Fine-level integration
        gate_fine = self.fine_gate(stage2_output)
        features_fine = self.fine_transform(stage2_output)
        final_output = self.norm_fine(features_fine * gate_fine)
        
        return final_output


# =====================================
# Internal: Adaptive Residual Gating (building block of HAE)
# =====================================

class AdaptiveResidualGating(nn.Module):
    """
    Adaptive Residual Gating
    
    Internal building block of the Hierarchical Attribute Extractor (HAE).
    Dynamically interpolates between original and fused features using a
    learnable gating mechanism, inspired by Residual Learning and Highway Networks.
    
    Mechanism:
        o = ReLU(W_o * [h_i; h_j] + b_o)
        beta = Sigmoid(W_beta * [o; f_HAE] + b_beta)
        f_final = LN(beta * f_HAE + (1 - beta) * o)
    
    Args:
        protein_dim (int): Dimension of protein embeddings (d)
        dropout (float): Dropout probability
    """
    def __init__(self, protein_dim, dropout=0.1):
        super().__init__()
        
        # Original feature pathway
        self.original_pathway = nn.Sequential(
            nn.Linear(protein_dim * 2, protein_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Adaptive integration gate
        self.integration_gate = nn.Sequential(
            nn.Linear(protein_dim * 2, protein_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(protein_dim)
        
    def forward(self, protein1, protein2, fused_features):
        """
        Adaptively integrate original and fused features
        
        Args:
            protein1, protein2: [batch_size, protein_dim] - Original embeddings
            fused_features: [batch_size, protein_dim] - Fused features from HGF
            
        Returns:
            integrated_features: [batch_size, protein_dim]
        """
        # Process original features
        original_concat = torch.cat([protein1, protein2], dim=-1)
        original_processed = self.original_pathway(original_concat)
        
        # Compute adaptive weights
        gate_input = torch.cat([original_processed, fused_features], dim=-1)
        integration_weights = self.integration_gate(gate_input)
        
        # Weighted combination
        integrated = integration_weights * fused_features + (1 - integration_weights) * original_processed
        integrated = self.layer_norm(integrated)
        
        return integrated


# =====================================
# Module (c): Hierarchical Attribute Extractor (HAE)
# =====================================

class HierarchicalAttributeExtractor(nn.Module):
    """
    Hierarchical Attribute Extractor (HAE)
    
    Employs a three-stage gating mechanism to progressively integrate interaction
    features, dynamically synthesizing information across varying spatial and
    energetic scales. Then applies adaptive residual gating to interpolate
    between original and fused features.
    
    Contains:
    - ThreeStageGating: Progressive coarse-to-fine feature fusion
    - AdaptiveResidualGating: Learnable interpolation between original and fused features
    
    Args:
        protein_dim (int): Dimension of protein embeddings from hyperbolic projector
        intermediate_dim (int): Hidden dimension (typically protein_dim // 2)
        dropout (float): Dropout probability
        use_adaptive_gating (bool): Whether to apply adaptive residual gating
        disable_fusion (bool): Ablation flag to use simple baseline fusion
    """
    def __init__(self, protein_dim, intermediate_dim, dropout=0.1,
                 use_adaptive_gating=True, disable_fusion=False):
        super().__init__()
        self.disable_fusion = disable_fusion
        self.use_adaptive_gating = use_adaptive_gating
        
        if disable_fusion:
            # Ablation baseline: Simple concatenation + MLP
            self.baseline_fusion = nn.Sequential(
                nn.Linear(protein_dim * 2, protein_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(protein_dim)
            )
        else:
            # Three-stage gating (Stage I, II, III)
            self.three_stage_gating = ThreeStageGating(
                protein_dim=protein_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout
            )
            # Adaptive residual gating (beta-gating)
            if use_adaptive_gating:
                self.adaptive_gating = AdaptiveResidualGating(
                    protein_dim=protein_dim,
                    dropout=dropout
                )
    
    def forward(self, protein1, protein2):
        """
        Args:
            protein1, protein2: [batch_size, protein_dim] - Protein pair embeddings from projector
            
        Returns:
            f_final: [batch_size, protein_dim] - Integrated representation
        """
        if self.disable_fusion:
            return self.baseline_fusion(
                torch.cat([protein1, protein2], dim=-1)
            )
        
        # Three-stage hierarchical gating: f_HAE
        f_hae = self.three_stage_gating(protein1, protein2)
        
        # Adaptive residual gating: f_final = LN(beta * f_HAE + (1 - beta) * o)
        if self.use_adaptive_gating:
            f_final = self.adaptive_gating(protein1, protein2, f_hae)
        else:
            f_final = f_hae
        
        return f_final


# =====================================
# Module (d): Auxiliary Generation Regularization
# =====================================

class AuxiliaryGenerationRegularization(nn.Module):
    """
    Auxiliary Generation Regularization
    
    Introduces an auxiliary sequence reconstruction task to regularize
    learned representations, ensuring fine-grained residue-level information
    is preserved during deep feature transformation.
    
    Architecture:
    - Variational encoder (reparameterization trick) with KL warmup annealing
    - Transformer-based autoregressive decoder with causal masking
    - Token embedding + sinusoidal positional encoding
    - Cross-entropy reconstruction loss on shifted predictions
    
    Args:
        input_dim (int): Dimension of fused representation (d)
        vocab_size (int): ESM-2 alphabet size (|Sigma| = 33)
        hidden_dim (int): Latent / decoder hidden dimension (d_z)
        ff_dim (int): Feed-forward dimension in Transformer
        heads (int): Number of attention heads
        layers (int): Number of Transformer layers
        max_len (int): Maximum sequence length (L)
        pad_value (int): Padding token index
        sos_value (int): Start-of-sequence token index
        eos_value (int): End-of-sequence token index
    """
    def __init__(self, input_dim, vocab_size, hidden_dim, ff_dim, heads, layers,
                 max_len, pad_value, sos_value=None, eos_value=None):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Variational latent space (reparameterization trick)
        self.variational_encoder = LSE_VAE_Latent(
            in_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # Memory encoder for conditioning the decoder
        self.memory_encoder = TransformerEncoder(
            dim=hidden_dim, ff_dim=ff_dim, num_head=heads, num_layer=layers
        )
        self.positional_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        self.memory_segment_encoding = nn.Parameter(torch.randn(hidden_dim))
        
        # Autoregressive Transformer decoder with causal masking
        self.sequence_decoder = Decoder(
            dim=hidden_dim, ff_dim=ff_dim, num_head=heads, num_layer=layers
        )
        
        # Token embedding and prediction head
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.token_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )
        torch.nn.init.zeros_(self.token_predictor[3].bias)
        
        # Special tokens
        self.pad_token = pad_value
        self.start_token = sos_value if sos_value is not None else 0
        self.end_token = eos_value if eos_value is not None else pad_value
        
        # KL divergence warmup annealing schedule
        self.kl_warmup_iterations = 5000
        self.kl_weight_max = 0.01
        self.register_buffer('training_step', torch.tensor(0))
    
    def compute_kl_weight(self):
        """KL weight with linear warmup: lambda_KL(t) = min(1, t / T_warmup)"""
        if not self.training:
            return self.kl_weight_max
        current_step = self.training_step.item()
        if current_step < self.kl_warmup_iterations:
            return self.kl_weight_max * (current_step / self.kl_warmup_iterations)
        return self.kl_weight_max
    
    def encode_memory(self, latent_code):
        """Encode latent code into memory for the decoder."""
        memory_input = latent_code + self.memory_segment_encoding
        memory_input = memory_input.unsqueeze(0)  # [1, batch, dim]
        memory_mask = memory_input.new_zeros(memory_input.shape[1], memory_input.shape[0]).bool()
        encoded_memory = self.memory_encoder(memory_input, src_key_padding_mask=memory_mask)
        return encoded_memory, memory_mask
    
    def forward(self, fused_representation, target_seq=None):
        """
        Training forward: compute generation and KL losses.
        
        Args:
            fused_representation: [batch_size, input_dim] - f_final from HAE
            target_seq: [batch_size, seq_len] - Tokenized target sequence
            
        Returns:
            generated_scores: [batch_size, seq_len, vocab_size] or None
            reconstruction_loss: Cross-entropy loss on shifted predictions
            kl_divergence_loss: Weighted KL divergence
        """
        # Reparameterization: mu, log_sigma -> z
        latent_code, kl_loss_raw = self.variational_encoder(fused_representation)
        
        # KL warmup annealing
        kl_weight = self.compute_kl_weight()
        kl_divergence_loss = kl_loss_raw * kl_weight
        
        if self.training:
            self.training_step += 1
        
        # Encode memory for decoder conditioning
        memory, memory_mask = self.encode_memory(latent_code)
        
        if target_seq is not None:
            batch_size, seq_length = target_seq.shape
            
            # Causal mask for autoregressive generation
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=target_seq.device),
                diagonal=1
            )
            
            # Embed target tokens with positional encoding
            target_embedded = self.token_embedding(target_seq)
            target_embedded = target_embedded.permute(1, 0, 2).contiguous()  # [seq, batch, dim]
            target_embedded = self.positional_encoder(target_embedded)
            
            # Decode
            decoder_output = self.sequence_decoder(
                target_embedded,
                memory,
                x_mask=causal_mask,
                mem_padding_mask=memory_mask,
            )
            decoder_output = decoder_output.permute(1, 0, 2).contiguous()  # [batch, seq, dim]
            
            # Predict tokens
            generated_scores = self.token_predictor(decoder_output)
            
            # Reconstruction loss: CE(Y_hat[1:L-1], S[2:L], ignore=PAD)
            shifted_scores = generated_scores[:, :-1, :].contiguous()
            shifted_targets = target_seq[:, 1:].contiguous()
            
            batch_size, seq_len_minus_1, vocab_size = shifted_scores.size()
            shifted_scores_flat = shifted_scores.view(-1, vocab_size)
            shifted_targets_flat = shifted_targets.view(-1)
            
            reconstruction_loss = F.cross_entropy(
                shifted_scores_flat,
                shifted_targets_flat,
                ignore_index=self.pad_token,
            )
        else:
            generated_scores = None
            reconstruction_loss = torch.tensor(0.0, device=latent_code.device)
        
        return generated_scores, reconstruction_loss, kl_divergence_loss
    
    def generate_sequence(self, memory, memory_mask, random_sampling=False, return_scores=False):
        """Autoregressive sequence generation from memory."""
        batch_size = memory.shape[1]
        device = memory.device
        
        generated = torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=device)
        
        if return_scores:
            score_history = []
        
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(self.max_len - 1):
            current_length = generated.size(1)
            
            token_embedded = self.token_embedding(generated)
            positional_embedded = self.positional_encoder.pe[:current_length].transpose(0, 1)
            sequence_embedded = token_embedded + positional_embedded.squeeze(1)
            sequence_embedded = sequence_embedded.permute(1, 0, 2)  # [seq, batch, dim]
            
            causal_mask = torch.triu(
                torch.ones(current_length, current_length, dtype=torch.bool, device=device),
                diagonal=1
            )
            
            decoder_output = self.sequence_decoder(
                sequence_embedded,
                memory,
                x_mask=causal_mask,
                mem_padding_mask=memory_mask
            )
            
            last_hidden_state = decoder_output[-1]
            next_token_logits = self.token_predictor(last_hidden_state)
            
            if return_scores:
                score_history.append(next_token_logits)
            
            if random_sampling:
                temperature = 0.9
                probabilities = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, 1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            is_finished |= (next_token == self.end_token) | (next_token == self.pad_token)
            if is_finished.all():
                break
        
        generated_sequences = generated[:, 1:]
        
        if return_scores:
            return generated_sequences, torch.stack(score_history, dim=1)
        return generated_sequences
    
    def generate(self, fused_representation, random_sampling=False, return_latent=False):
        """Generate sequences from fused representation."""
        latent_code, _ = self.variational_encoder(fused_representation)
        memory, memory_mask = self.encode_memory(latent_code)
        generated_sequences = self.generate_sequence(
            memory, memory_mask, random_sampling=random_sampling, return_scores=False
        )
        if return_latent:
            return generated_sequences, latent_code.detach().cpu().numpy()
        return generated_sequences


# =====================================
# Main Model - REFLEX
# =====================================

class REFLEX(BaseModel):
    """
    REFLEX: Enhanced PPI Prediction with Hierarchical Fusion
    
    Architecture (following paper methodology):
    (a) Pretrained Protein Embedder: Pre-configured (VQ-VAE codebook + ESM-2, in ppi_data.py)
    (b) Adaptive Hyperbolic Projector: Lorentz manifold projection with multi-view GCN
    (c) Hierarchical Attribute Extractor (HAE): Three-stage gating + adaptive residual gating
    (d) Auxiliary Generation Regularization: VAE-based sequence reconstruction regularizer
    + Prediction & Optimization: Linear classifier with BCE loss
    
    Ablation Options:
    - disable_hierarchical_fusion: Use simple concatenation baseline in HAE
    - disable_generation_task: Remove auxiliary generation regularization
    
    Args:
        input_dim (int): Input protein feature dimension
        vocab_size (int): Vocabulary size for sequence generation (ESM-2 alphabet)
        pad_value (int): Padding token index
        class_num (int): Number of PPI interaction types (default: 7)
        hidden_dim (int): Hidden dimension for generation modules (default: 376)
        use_adaptive_integration (bool): Enable adaptive residual gating in HAE (default: True)
        disable_hierarchical_fusion (bool): Ablation flag for HAE (default: False)
        disable_generation_task (bool): Ablation flag for generation (default: False)
    """
    
    def __init__(
        self,
        input_dim,
        vocab_size,
        pad_value,
        sos_value=None,
        eos_value=None,
        act='relu',
        layer_num=2,
        radius=None,
        dropout=0.0,
        if_bias=True,
        use_att=0,
        local_agg=0,
        class_num=7,
        in_len=512,
        device=None,
        hidden_dim=376,
        ff_dim=1024,
        heads=8,
        layers=4,
        max_len=4096,
        use_adaptive_integration=True,
        disable_hierarchical_fusion=False,
        disable_generation_task=False,
        args=None,
        **kwargs
    ):
        super(REFLEX, self).__init__(args)
        
        # Model configuration
        self.device = device
        self.max_len = max_len
        self.latent_dim = hidden_dim
        self.num_interaction_types = class_num
        self.vocab_size = vocab_size
        self.disable_generation_task = disable_generation_task
        
        # ========== Module (b): Adaptive Hyperbolic Projector ==========
        self.hyperbolic_projector = AdaptiveHyperbolicProjector(
            input_dim=input_dim,
            esm_dim=1280,  # ESM2 embedding dimension
            args=self.args,
            act=act,
            layer_num=layer_num,
            radius=radius,
            dropout=dropout,
            if_bias=if_bias,
            use_att=use_att,
            local_agg=local_agg,
            class_num=class_num,
            in_len=in_len,
            device=device,
        )
        self.protein_embedding_dim = self.hyperbolic_projector.output_dim
        
        # ========== Module (c): Hierarchical Attribute Extractor ==========
        self.attribute_extractor = HierarchicalAttributeExtractor(
            protein_dim=self.protein_embedding_dim,
            intermediate_dim=self.protein_embedding_dim // 2,
            dropout=dropout,
            use_adaptive_gating=use_adaptive_integration,
            disable_fusion=disable_hierarchical_fusion,
        )
        if disable_hierarchical_fusion:
            print("[REFLEX] Ablation mode: HAE using baseline fusion")
        else:
            print("[REFLEX] Using Hierarchical Attribute Extractor (HAE)")
            if use_adaptive_integration:
                print("[REFLEX]   with adaptive residual gating")
        
        # ========== Prediction: PPI Interaction Classifier ==========
        classifier_input_dim = self.protein_embedding_dim
        self.interaction_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim // 2, classifier_input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim // 4, self.num_interaction_types),
        )
        
        # ========== Module (d): Auxiliary Generation Regularization (Optional) ==========
        self.pad_token = pad_value
        self.start_token = sos_value if sos_value is not None else 0
        self.end_token = eos_value if eos_value is not None else pad_value
        
        if not self.disable_generation_task:
            print("[REFLEX] Enabling Auxiliary Generation Regularization")
            self.generation_regularizer = AuxiliaryGenerationRegularization(
                input_dim=self.protein_embedding_dim,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                heads=heads,
                layers=layers,
                max_len=max_len,
                pad_value=pad_value,
                sos_value=sos_value,
                eos_value=eos_value,
            )
        else:
            print("[REFLEX] Auxiliary Generation Regularization disabled")
    
    def _extract_pair_embeddings(self, data, edge_id):
        """Extract protein pair embeddings from graph data."""
        protein_embeddings = self.hyperbolic_projector(data)
        edge_index = data.edge2
        node_indices = edge_index[:, edge_id]
        
        if node_indices.dim() == 1:
            protein1 = protein_embeddings[node_indices[0]].unsqueeze(0)
            protein2 = protein_embeddings[node_indices[1]].unsqueeze(0)
        else:
            protein1 = protein_embeddings[node_indices[0]]
            protein2 = protein_embeddings[node_indices[1]]
        
        return protein1, protein2
    
    def forward(self, data, edge_id, target_seq=None):
        """
        Forward pass for PPI prediction and optional sequence generation.
        
        Args:
            data: PyG Data object with protein graph
            edge_id: Edge indices to predict
            target_seq: [batch_size, seq_len] Target sequences for generation (optional)
            
        Returns:
            interaction_logits: [batch_size, num_interaction_types]
            generated_scores: [batch_size, seq_len, vocab_size] or None
            reconstruction_loss: Cross-entropy reconstruction loss
            kl_divergence_loss: Weighted KL divergence loss
        """
        # ========== Step 1: Adaptive Hyperbolic Projector ==========
        protein1_embedding, protein2_embedding = self._extract_pair_embeddings(data, edge_id)
        
        # ========== Step 2: Hierarchical Attribute Extractor ==========
        fused_representation = self.attribute_extractor(protein1_embedding, protein2_embedding)
        
        # ========== Step 3: Prediction ==========
        interaction_logits = self.interaction_classifier(fused_representation)
        
        # ========== Step 4: Auxiliary Generation Regularization (Optional) ==========
        if self.disable_generation_task:
            kl_divergence_loss = torch.tensor(0.0, device=interaction_logits.device)
            reconstruction_loss = torch.tensor(0.0, device=interaction_logits.device)
            generated_scores = None
        else:
            generated_scores, reconstruction_loss, kl_divergence_loss = \
                self.generation_regularizer(fused_representation, target_seq)
        
        return interaction_logits, generated_scores, reconstruction_loss, kl_divergence_loss
    
    def generate(self, data, edge_id, random_sampling=False, return_latent=False):
        """
        Generate sequences for protein pairs.
        
        Args:
            data: PyG Data object
            edge_id: Edge indices
            random_sampling: Whether to use random sampling
            return_latent: Whether to return latent codes
            
        Returns:
            generated_sequences: [batch_size, max_len] or None if generation disabled
            latent_codes: [batch_size, latent_dim] (optional)
        """
        if self.disable_generation_task:
            if return_latent:
                return None, None
            return None
        
        # Encode proteins and extract hierarchical attributes
        protein1_embedding, protein2_embedding = self._extract_pair_embeddings(data, edge_id)
        fused_representation = self.attribute_extractor(protein1_embedding, protein2_embedding)
        
        return self.generation_regularizer.generate(
            fused_representation, random_sampling=random_sampling, return_latent=return_latent
        )
