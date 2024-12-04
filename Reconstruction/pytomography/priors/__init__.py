r"""This module contains classes/functionality for encorporating priors in statistical reconstruction algorithms. Under the modification :math:`L(\tilde{f}, f) \to L(\tilde{f}, f)e^{-\beta V(f)}`, the log-liklihood becomes :math:`\ln L(\tilde{f},f) - \beta V(f)`. Typically, the prior has a form :math:`V(f) = \sum_{r,s} w_{r,s} \phi(f_r,f_s)`. In this expression, :math:`r` represents a voxel in the object, :math:`s` represents a voxel nearby to voxel :math:`r`, and :math:`w_{r,s}` represents a weighting between the voxels."""

from .prior import Prior
from .nearest_neighbour import NearestNeighbourPrior, QuadraticPrior, LogCoshPrior, RelativeDifferencePrior, NeighbourWeight, EuclideanNeighbourWeight, AnatomyNeighbourWeight, TopNAnatomyNeighbourWeight