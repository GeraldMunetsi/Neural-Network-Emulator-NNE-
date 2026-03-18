"""
step0_model.py — SIR Epidemic Emulator

Architecture overview

1. StandardRFF          : [tau,gamma,rho] to 128-dim Fourier embedding                     
2. Fusion MLP           : 128 to latent_dim (learns nonlinear combinations)
3. S decoder            : monotone-decreasing B-spline via cumprod retention
4. g(t) decoder         : free B-spline + sigmoid then g ∈ (0,1)
5. Conservation         : I = (N-S)·g,  R = (N-S)·(1-g)
                          then S+I+R = N exactly, I≥0, R≥0 always

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import BSpline

#OPTION 1
#  STANDARD FOURIER FEATURES (the basic)
# Creating the frequency matrix (W) myself (not randomly sampling fron Gausian, more interpretable, )
#So I have tried different ways to create W :
#For example  W = base_frequency * torch.stack([k,k**2,k**3], dim=1) with normilization :  W = W / W.norm(dim=1, keepdim=True)
#Another example :  W = base_frequency * torch.stack([k,k**2,k**4], dim=1), with normilization :  W = W / W.norm(dim=1, keepdim=True)
#Still not producing good results when i train the model.
#Please advise 

class StandardFourierFeatures(nn.Module):
    """
    Standard (deterministic) Fourier Features.

    Frequencies are fixed harmonics rather than random samples.

    Maps:
        (batch, n_params) to (batch, 2 × n_fourier)
    """

    def __init__(self, n_params=3, n_fourier=64, base_frequency=3): # base_frequency rescales W to control the frequency of oscillations in the embedding:

        # Deterministic frequency indices
        k = torch.arange(1, n_fourier + 1).float() # k indices

        # deterministic frequency matrix
        W = torch.zeros(n_fourier, 3)
     
        W[:,0] = k        
        W[:,1] = 2*k     
        W[:,2] = 4*k   
        W = W / W.norm(dim=1, keepdim=True) # normilizatrion
        W = base_frequency * W # multipying the frequency matrix with the base frequence
        
        # Register as non-trainable buffer
        self.register_buffer("W", W)

        self.output_dim = 2 * n_fourier # 128 cosine and sine embeddings that i have included in phi

#OPTION 2
#Standard Random Fourier features
#Sampling from the frequency Matrix W from Gaussion distribution
# Critically, because each row of W is an independent random vector in ℝ³, the projection z_k = τ·w₁ + γ·w₂ + ρ·w₃ assigns genuinely distinct weights to all three parameters at every frequency, enabling the embedding to represent the multiplicative interaction τ/γ that determines R₀ — something no additive deterministic frequency scheme can achieve without explicit cross-parameter interaction terms
# Working well!

class StandardRFF(nn.Module):
    def __init__(self, n_params=3, n_fourier=64, sigma=1.0):
        super().__init__()
        W = torch.randn(n_fourier, n_params) * sigma  # each row is a random 3D vector
        self.register_buffer('W', W)
        self.scale = (2.0 / n_fourier) ** 0.5        # RBF kernel normalisation
        self.output_dim = 2 * n_fourier

    def forward(self, x):
        z   = x @ self.W.T
        phi = self.scale * torch.cat([torch.cos(z), torch.sin(z)], dim=1)
        return phi    

    def forward(self, x):
        """
        x : (batch, n_params)
        """

        z = x @ self.W.T  # linear combination

        phi = torch.cat(
            [torch.cos(z), torch.sin(z)],# concartinate along the feature dimension
            dim=1
        )

        return phi



# 2. B-SPLINE LAYER 

class BSplineLayer(nn.Module):
    """
    Differentiable B-spline evaluation layer

    Converts n_knots control-point coefficients into n_timepoints smooth values
    using a pre-computed (frozen) basis matrix B of shape (n_timepoints, n_knots).

    output[b, t] = Σ coeffs[b, k] × B[t, k]= coeffs @ B.T   (batch matrix multiply)
    The basis matrix is computed once at initilization using scipy and registered as a
    buffer (moves to my CPU automatically, never trained).
    """
    def __init__(self, n_knots, n_timepoints, degree= 3):
        super().__init__()
        self.n_knots      = n_knots
        self.n_timepoints = n_timepoints

        # Build clamped B-spline knot vector
        # Clamped = curve passes exactly through first and last control points
        internal = np.linspace(0, 1, n_knots - degree + 1)
        knots    = np.concatenate([
            np.zeros(degree),
            internal,
            np.ones(degree),
        ]) #Repetition at ends ensures the curve touches the first and last control points

        # Evaluate all basis functions at each timestep
        t_eval  = np.linspace(0, 1, n_timepoints)
        B_np    = np.zeros((n_timepoints, n_knots))
        for k in range(n_knots):
            c = np.zeros(n_knots)
            c[k] = 1.0
            B_np[:, k] = BSpline(knots, c, degree)(t_eval)

        self.register_buffer('B', torch.tensor(B_np, dtype=torch.float32))

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeffs : (batch, n_knots)
        Returns:
            curve  : (batch, n_timepoints)
        """
        return coeffs @ self.B.T    # (batch, n_timepoints)


# 3. TEMPORAL DECODER

class TemporalDecoder(nn.Module):
    """
    Decodes latent vector z and unnormilized rho into SIR trajectories.

    S compartment
  
    Predicted via a monotone-decreasing B-spline.
    Uses cumprod of sigmoid retention rates so S can only ever go down:
        S_coeffs[k] = S₀ × r₁ × r₂ × ... × rₖ,   each rᵢ ∈ (0,1)
    Guarantees: S(t) ≤ S(0) = N(1-ρ) for all t.

    g(t) function
    g(t) = I(t) / (N - S(t)) = fraction of ever-infected still infectious.
    Predicted as a FREE B-spline , then we sigmoid it such that g ∈ (0,1).
    First spline coefficient is pinned to 20.0 so sigmoid(20) ≈ 1 → g(0) ≈ 1.

    I and R

    I(t) = (N - S(t)) × g(t)≥ 0 always (both factors ≥ 0)
    R(t) = (N - S(t)) × (1 - g(t))≥ 0 always (g < 1)
    S + I + R = S + (N-S)·[g+(1-g)] = N, exact conservation

    Args:
        latent_dim : dimension of input z from fusion MLP
        n_knots : number of B-spline control points
        n_timepoints : output timesteps (50)
        total_population: N (10000) 
        hidden_dim: hidden size for retention and g networks
    """
    def __init__(
        self,
        latent_dim     : int,
        n_knots        : int   = 12,
        n_timepoints   : int   = 50,
        total_population: int  = 10000,
        hidden_dim     : int   = 64,
    ):
        super().__init__()
        self.N             = float(total_population)
        self.n_knots       = n_knots
        self.n_timepoints  = n_timepoints

        # S decoder 
        # Predicts (n_knots - 1) retention rates (first rate is always 1.0)
        self.predict_S_retention = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_knots - 1),
        )

        # g(t) decoder 
        # Predicts (n_knots - 1) free coefficients; first is pinned to 8.0
        self.predict_g_coeffs = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_knots - 1),
        )

        # B-spline layers (shared basis) 
        self.spline_S = BSplineLayer(n_knots, n_timepoints)
        self.spline_g = BSplineLayer(n_knots, n_timepoints)

    def forward(self, z: torch.Tensor, rho_raw: torch.Tensor) -> tuple:
        """
        Args:
            z       : (batch, latent_dim)  — from fusion MLP
            rho_raw : (batch,)             — ρ ∈ [0.001, 0.010] # proportion of individuals initially infected

        Returns:
            S_pred, I_pred, R_pred  each (batch, n_timepoints)
        """
        batch_size = z.size(0)
        device     = z.device

        # S₀ = N·(1-ρ) 
        S_0 = ((1.0 - rho_raw) * self.N).unsqueeze(1)   # (batch, 1)

        # S: monotone-decreasing spline 
        # retention_rates ∈ (0,1) — each is the fraction of S remaining
        retention_raw   = self.predict_S_retention(z)         # (batch, n_knots-1)
        retention_rates = torch.sigmoid(retention_raw)         # (0,1)

        # First rate = 1.0 (S starts at S₀, no drop at t=0)
        ones        = torch.ones(batch_size, 1, device=device)
        all_rates   = torch.cat([ones, retention_rates], dim=1)  # (batch, n_knots)

        # Cumulative product: S_coeffs[k] = S₀ × r₁ × ... × rₖ
        # All coefficients ≤ S₀, monotone decreasing
        cum_product = torch.cumprod(all_rates, dim=1)            # (batch, n_knots)
        S_coeffs    = S_0 * cum_product                          # (batch, n_knots)
        S_pred      = self.spline_S(S_coeffs)                    # (batch, T)

        # g(t): fraction of ever-infected still infectious 
       
        g_free      = self.predict_g_coeffs(z)                   # (batch, n_knots-1)
        g_coeff_0   = torch.full((batch_size, 1), 20.0, device=device)

        #Why did i choose the value 20.0
        # g(t): fraction of ever-infected still infectious
        #Recall I(t) = (N - S(t)) · g(t), R(t) = (N - S(t)) · (1 - g(t))
        # At t=0, I(0) = N·ρ  (only the seed fraction is infected),R(0) = 0          (nobody has recovered yet)
        # At t=0, R(0) = (N - S₀) · (1 - g(0)) = 0
        #Since (N - S₀) = N·ρ > 0 always, the only way R(0)=0 is satisfied is if:
        # g(0) = 1   exactly
        # g is not the raw spline output, g(t) = sigmoid(h(t))
        # so pinning to 1.0 would give me , g(0) = sigmoid(1.0) = 0.731 thus R(0) ≠ 0
        # So i need  c such that sigmoid(c) ≈ 1
        # so i choose c=8 then sigmoid(20.0) = 0.999999979 
        #R(0) = (N·ρ) · (1 - 0.999999979)= 50 · 0.0003 = 0.00000105 people , basically zero
      
        g_coeffs    = torch.cat([g_coeff_0, g_free], dim=1)      # (batch, n_knots)
        g_spline    = self.spline_g(g_coeffs)                    # (batch, T)
        g           = torch.sigmoid(g_spline)                    # (0,1) strict

        # I and R from conservation 
        # N - S(t) = everyone who has ever left S = I(t) + R(t)
        ever_infected = self.N - S_pred                          # ≥ 0 always

        I_pred = ever_infected * g                               # (batch, T) ≥ 0
        R_pred = ever_infected * (1.0 - g)                      # (batch, T) ≥ 0

        # Conservation check (for debugging, not needed for correctness):
        # S_pred + I_pred + R_pred = S + (N-S)·g + (N-S)·(1-g)= S + (N-S)·1 = N  

        return S_pred, I_pred, R_pred


# 4. FULL MODEL


class HybridSIREmulator(nn.Module):
    """
    Full SIR emulator.

    Pipeline:
        params_norm (batch,3)
            ↓  StandardRFF
        phi (batch, 2×n_fourier = 128)
            ↓  Fusion MLP
        z   (batch, latent_dim)
            ↓  TemporalDecoder  [+ rho_raw]
        S_pred, I_pred, R_pred  (batch, n_timepoints)
    """
    def __init__(self, config: dict):
        super().__init__()

        n_params         = config.get('n_params',          3)
        n_fourier        = config.get('n_fourier',        64)
        sigma            = config.get('sigma',           1.0)
        fusion_hidden    = config.get('fusion_hidden',   128)
        latent_dim       = config.get('latent_dim',       64)
        n_knots          = config.get('n_knots',          12)
        n_timepoints     = config.get('n_timepoints',     50)
        total_population = config.get('total_population', 10000)
        decoder_hidden   = config.get('decoder_hidden',   64)
        dropout          = config.get('dropout',         0.1)

        rff_out = 2 * n_fourier   # 128

        # 1. Standard RFF 
        self.rff = StandardRFF(
            n_params  = n_params,
            n_fourier = n_fourier,
            sigma     = sigma,
        )

        # 2. Fusion MLP 
        # 128 → fusion_hidden → latent_dim
        # LayerNorm stabilises training after the unnormalised RFF output
        self.fusion = nn.Sequential(
            nn.Linear(rff_out, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, latent_dim),
            nn.ReLU(),
        )

        # 3. Temporal decoder 
        self.temporal_decoder = TemporalDecoder(
            latent_dim      = latent_dim,
            n_knots         = n_knots,
            n_timepoints    = n_timepoints,
            total_population= total_population,
            hidden_dim      = decoder_hidden,
        )

        self.n_timepoints = n_timepoints

    def forward(self, data, n_timesteps: int = None) -> torch.Tensor:
        """
        Args:
            data        : BatchWrapper with fields .params_norm and .rho_raw
            n_timesteps : ignored (kept for API compatibility); always uses
                          self.n_timepoints from config

        Returns:
            predictions : (batch, n_timepoints, 3)  — [S, I, R] stacked
        """
        params_norm = data.params_norm    # (batch, 3)
        rho_raw     = data.rho_raw        # (batch,)

        # Fourier embedding
        phi = self.rff(params_norm)       # (batch, 128)

        # Latent vector
        z   = self.fusion(phi)            # (batch, latent_dim)

        # SIR trajectories
        S_pred, I_pred, R_pred = self.temporal_decoder(z, rho_raw)

        # Stack to (batch, T, 3) for loss function compatibility
        return torch.stack([S_pred, I_pred, R_pred], dim=2)

    def get_component_params(self) -> dict:
        """Return parameter counts per component (useful for logging)."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        rff_frozen = sum(
            b.numel() for b in self.rff.buffers()
        )

        return {
            'rff_trainable'   : count(self.rff),
            'rff_frozen'      : rff_frozen,
            'fusion'          : count(self.fusion),
            'temporal_decoder': count(self.temporal_decoder),
            'total'           : count(self),
        }



# 5. FACTORY FUNCTION
def create_hybrid_mlp_model(config: dict) -> HybridSIREmulator:
    """
    Build and return the SIR emulator from a config dict.

    Minimal config:
        config = {
            'n_params'        : 3,
            'n_fourier'       : 64,
            'sigma'           : 1.0,
            'fusion_hidden'   : 128,
            'latent_dim'      : 64,
            'n_knots'         : 12,
            'n_timepoints'    : 50,
            'total_population': 10000,
            'decoder_hidden'  : 64,
            'dropout'         : 0.1,
        }
    """
    model = HybridSIREmulator(config)
    return model


