

import torch
import numpy as np
from scipy.stats import nbinom, norm
from typing import Dict, Union
import time
import matplotlib.pyplot as plt

class BivariateForecastEvaluator:
    """
    A class for evaluating probabilistic forecasts of bivariate distributions,
    particularly with negative binomial marginals linked by a Gaussian copula.
    
    Implements the following evaluation metrics:
    - Energy Score (ES)
    - Multivariate Continuous Ranked Probability Score (MCRPS)
    """
    
    def __init__(self, device: torch.device = torch.device("cpu")):
        """
        Initialize the evaluator.
        
        Args:
            device: Torch device to use for computations (CPU or CUDA)
        """
        self.device = device
        
    def evaluate_forecast(self, 
                          r1_pred: torch.Tensor,
                          p1_pred: torch.Tensor,
                          r2_pred: torch.Tensor,
                          p2_pred: torch.Tensor,
                          rho_pred: torch.Tensor,
                          y_true: torch.Tensor,
                          num_samples_es: int = 200,
                          num_samples_mcrps: int = 1000,
                          verbose: bool = True) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate a probabilistic forecast using multiple metrics.
        
        Args:
            r1_pred, p1_pred: Parameters for the first NB marginal (shape: batch_size)
            r2_pred, p2_pred: Parameters for the second NB marginal (shape: batch_size)
            rho_pred: Correlation parameter for the Gaussian copula (shape: batch_size)
            y_true: True observations (shape: batch_size, 2)
            num_samples_es: Number of samples to draw for Energy Score calculation
            num_samples_mcrps: Number of samples for MCRPS calculation
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if verbose:
            print("Evaluating forecast...")
            start_time = time.time()
        
        n_observations = y_true.shape[0]
        
        # Input validation
        if not (r1_pred.shape == (n_observations,) and 
                p1_pred.shape == (n_observations,) and
                r2_pred.shape == (n_observations,) and 
                p2_pred.shape == (n_observations,) and
                rho_pred.shape == (n_observations,)):
            raise ValueError("Parameter tensors must have shape (n_observations,)")
        
        if y_true.shape != (n_observations, 2):
            raise ValueError(f"y_true must have shape ({n_observations}, 2)")
        
        # Calculate Energy Score
        if verbose:
            print(f"Calculating Energy Score with {num_samples_es} samples per forecast...")
        energy_score = self._calculate_energy_score(
            r1_pred, p1_pred, r2_pred, p2_pred, rho_pred, y_true, num_samples_es)
        
        # Calculate MCRPS
        if verbose:
            print(f"Calculating MCRPS with {num_samples_mcrps} samples per forecast...")
        mcrps = self._calculate_mcrps(
            r1_pred, p1_pred, r2_pred, p2_pred, rho_pred, y_true, num_samples_mcrps)
        
        if verbose:
            end_time = time.time()
            print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        
        # Combine results
        results = {
            'energy_score': energy_score,
            'mcrps': mcrps
        }
        
        return results
    
    def _calculate_energy_score(self, 
                              r1: torch.Tensor, 
                              p1: torch.Tensor, 
                              r2: torch.Tensor, 
                              p2: torch.Tensor, 
                              rho: torch.Tensor, 
                              y_true: torch.Tensor, 
                              num_samples: int) -> float:
        """
        Calculate Energy Score (ES) using Monte Carlo simulation.
        
        Args:
            r1, p1, r2, p2, rho: Distribution parameters
            y_true: True observations
            num_samples: Number of samples for MC estimation
            
        Returns:
            Mean Energy Score
        """
        n_obs = len(r1)
        
        # Generate samples for each forecast
        r1_samples = r1.repeat_interleave(num_samples)
        p1_samples = p1.repeat_interleave(num_samples)
        r2_samples = r2.repeat_interleave(num_samples)
        p2_samples = p2.repeat_interleave(num_samples)
        rho_samples = rho.repeat_interleave(num_samples)
        
        # Generate samples using a helper function to sample from bivariate NB with Gaussian copula
        samples = self._sample_bivariate_nb_gaussian_copula(
            n_obs * num_samples, r1_samples, p1_samples, r2_samples, p2_samples, rho_samples
        )
        
        # Reshape samples: (n_obs, num_samples, 2)
        samples = samples.reshape(n_obs, num_samples, 2)
        
        # Move y_true to device and expand for vectorized distance calculation
        y_true_dev = y_true.to(self.device).unsqueeze(1)  # (n_obs, 1, 2)
        
        # Calculate first term: E_F ||X - y||
        diff_xy = samples - y_true_dev
        norm_xy = torch.norm(diff_xy, dim=2)  # Euclidean distance
        term1 = torch.mean(norm_xy, dim=1)  # Mean over samples
        
        # Calculate second term: 0.5 * E_F ||X - X'|| (pairwise differences)
        samples1 = samples.unsqueeze(2)  # (n_obs, num_samples, 1, 2)
        samples2 = samples.unsqueeze(1)  # (n_obs, 1, num_samples, 2)
        diff_xx = samples1 - samples2  # (n_obs, num_samples, num_samples, 2)
        norm_xx = torch.norm(diff_xx, dim=3)  # (n_obs, num_samples, num_samples)
        term2 = 0.5 * torch.mean(norm_xx, dim=(1, 2))  # Mean over all pairs
        
        # Energy score = term1 - term2
        es_per_obs = term1 - term2
        mean_es = torch.mean(es_per_obs).item()
        
        return mean_es
    
    def _calculate_mcrps(self, 
                         r1: torch.Tensor, 
                         p1: torch.Tensor, 
                         r2: torch.Tensor, 
                         p2: torch.Tensor, 
                         rho: torch.Tensor, 
                         y_true: torch.Tensor, 
                         num_samples: int) -> float:
        """
        Calculate Multivariate Continuous Ranked Probability Score (MCRPS).
        
        Args:
            r1, p1, r2, p2, rho: Distribution parameters
            y_true: True observations
            num_samples: Number of samples for MC estimation
            
        Returns:
            MCRPS value
        """
        n_obs = len(r1)
        
        # Generate samples for each forecast
        r1_samples = r1.repeat_interleave(num_samples)
        p1_samples = p1.repeat_interleave(num_samples)
        r2_samples = r2.repeat_interleave(num_samples)
        p2_samples = p2.repeat_interleave(num_samples)
        rho_samples = rho.repeat_interleave(num_samples)
        
        # Generate samples using a helper function to sample from bivariate NB with Gaussian copula
        samples = self._sample_bivariate_nb_gaussian_copula(
            n_obs * num_samples, r1_samples, p1_samples, r2_samples, p2_samples, rho_samples
        )
        
        # Reshape samples: (n_obs, num_samples, 2)
        samples = samples.reshape(n_obs, num_samples, 2)
        
        # Calculate the MCRPS
        mcrps = 0.0
        for i in range(n_obs):
            for j in range(2):  # Iterate over the two variables (dimensions)
                # Get the true value for the j-th dimension
                true_value = y_true[i, j]
                # Calculate the CDF of the forecasted distribution for the true value
                cdf_values = np.mean(samples[i, :, j] <= true_value)
                mcrps += (cdf_values - 0.5) ** 2
        
        return mcrps / n_obs
    
    def _sample_bivariate_nb_gaussian_copula(self,
                                           batch_size: int,
                                           r1: torch.Tensor,
                                           p1: torch.Tensor,
                                           r2: torch.Tensor,
                                           p2: torch.Tensor,
                                           rho: torch.Tensor) -> torch.Tensor:
        """
        Sample from bivariate negative binomial with Gaussian copula.
        
        Args:
            batch_size: Number of samples to generate
            r1, p1, r2, p2: NB parameters
            rho: Correlation parameter
            
        Returns:
            Tensor of samples with shape (batch_size, 2)
        """
        # Ensure inputs are on the correct device
        r1 = r1.to(self.device)
        p1 = p1.to(self.device)
        r2 = r2.to(self.device)
        p2 = p2.to(self.device)
        rho = rho.to(self.device)
        
        # Generate correlated Gaussian samples
        z1 = torch.randn(batch_size, device=self.device)
        z2 = rho * z1 + torch.sqrt(1 - rho**2) * torch.randn(batch_size, device=self.device)
        
        # Transform to uniform via normal CDF
        normal = torch.distributions.Normal(0, 1)
        u1 = normal.cdf(z1)
        u2 = normal.cdf(z2)
        
        # Clamp values for numerical stability
        u1 = torch.clamp(u1, min=1e-6, max=1-1e-6)
        u2 = torch.clamp(u2, min=1e-6, max=1-1e-6)
        
        # Apply inverse CDF to get NB samples
        x1_np = nbinom.ppf(u1.cpu().numpy(), r1.cpu().numpy(), p1.cpu().numpy())
        x2_np = nbinom.ppf(u2.cpu().numpy(), r2.cpu().numpy(), p2.cpu().numpy())
        
        # Convert back to torch tensors
        x1 = torch.from_numpy(x1_np).float().to(self.device)
        x2 = torch.from_numpy(x2_np).float().to(self.device)
        
        # Combine into pairs
        samples = torch.stack([x1, x2], dim=1)
        
        return samples
    
    def plot_results(self, results: Dict[str, Union[float, np.ndarray]], title: str = "Forecast Evaluation") -> None:
        """
        Plot evaluation results.
        
        Args:
            results: Dictionary of evaluation results
            title: Title for the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Display key metrics in a text box
        plt.subplot(2, 1, 1)
        plt.axis('off')
        metrics_text = f"""
        Energy Score: {results['energy_score']:.4f} (lower is better)
        MCRPS: {results['mcrps']:.4f} (lower is better)
        """
        plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


