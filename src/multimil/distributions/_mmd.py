import torch
from typing import Optional, List

class MMD(torch.nn.Module):
    """Maximum mean discrepancy.

    Parameters
    ----------
    kernel_type : str
        Indicates if to use Gaussian kernel. One of
        * ``'gaussian'`` - use Gaussian kernel
        * ``'not gaussian'`` - do not use Gaussian kernel.
    """

    def __init__(self, kernel_type: str = "gaussian"):
        super().__init__()
        self.kernel_type = kernel_type

    def gaussian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gamma: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Apply Gaussian kernel.

        Parameters
        ----------
        x : torch.Tensor
            Tensor from the first distribution.
        y : torch.Tensor
            Tensor from the second distribution.
        gamma : Optional[List[float]]
            List of gamma parameters.

        Returns
        -------
        torch.Tensor
            Gaussian kernel between ``x`` and ``y``.
        """

        # Check if x and y have the same shape
        if x.shape != y.shape:
            raise ValueError(f"Input tensors x and y must have the same shape, but got {x.shape} and {y.shape}")

        if gamma is None:
            gamma = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                     1, 5, 10, 15, 20, 25, 30, 35, 100,
                     1e3, 1e4, 1e5, 1e6]

        # Convert gamma to a torch.Tensor (ensure it's on the correct device)
        gamma = torch.as_tensor(gamma, device=x.device, dtype=x.dtype)
        D = torch.cdist(x, y).pow(2).unsqueeze(-1)

        # This computes the exponential for all gamma values in one operation and then averages over the gamma dimension.
        gamma = gamma.view(1, 1, -1)
        K = torch.exp(-D * gamma).mean(dim=-1)

        return K

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Adapted from
        Title: scarches
        Date: 9th Octover 2021
        Code version: 0.4.0
        Availability: https://github.com/theislab/scarches/blob/63a7c2b35a01e55fe7e1dd871add459a86cd27fb/scarches/models/trvae/losses.py
        Citation: Gretton, Arthur, et al. "A Kernel Two-Sample Test", 2012.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with shape ``(batch_size, z_dim)``.
        y : torch.Tensor
            Tensor with shape ``(batch_size, z_dim)``.

        Returns
        -------
        torch.Tensor
            MMD between ``x`` and ``y``.
        """
        # In case there is only one sample in a batch belonging to one of the groups, then skip the batch
        if len(x) == 1 or len(y) == 1:
            return torch.tensor(0.0)

        # Resampling logic to ensure x and y have the same shape NEW (03.10.2024)
        if x.shape[0] > y.shape[0]:
            indices = torch.randperm(x.shape[0])[:y.shape[0]]
            x = x[indices]
        elif y.shape[0] > x.shape[0]:
            indices = torch.randperm(y.shape[0])[:x.shape[0]]
            y = y[indices]

        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
