from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform
from scipy.ndimage import map_coordinates
from torch.nn.functional import grid_sample

class DVFMotionTransform(Transform):
	def __init__(
		self,
		dvf_forward: torch.Tensor | None = None,
		dvf_backward: torch.Tensor | None = None,
		)-> None:
		"""Object to object transform that uses a deformation vector field to deform an object. 

		Args:
			dvf_forward (torch.Tensor[Lx,Ly,Lz,3] | None, optional): Vector field correspond to forward transformation. If None, then no transformation is used. Defaults to None.
			dvf_backward (torch.Tensor[Lx,Ly,Lz,3] | None, optional): Vector field correspond to backward transformation. If None, then no transformation is used. Defaults to None. Defaults to None.
		"""
		self.dvf_forward = dvf_forward.to(pytomography.device).to(pytomography.dtype)
		self.dvf_backward = dvf_backward.to(pytomography.device).to(pytomography.dtype)
		super(DVFMotionTransform, self).__init__()  ## go to the _init_ in Class Transform
  
	def _get_old_coordinates(self):
		"""Obtain meshgrid of coordinates corresponding to the object

		Returns:
			torch.Tensor: Tensor of coordinates corresponding to input object
		"""
		dim_x, dim_y, dim_z = self.object_meta.shape
		coordinates=torch.stack(torch.meshgrid(torch.arange(dim_x),torch.arange(dim_y), torch.arange(dim_z), indexing='ij')).permute((1,2,3,0)).to(pytomography.device).to(pytomography.dtype)
		return coordinates

	def _get_new_coordinates(self, old_coordinates: torch.Tensor, DVF: torch.Tensor):
		"""Obtain the new coordinates of each voxel based on the DVF.

		Args:
			old_coordinates (torch.Tensor): Old coordinates of each voxel
			DVF (torch.Tensor): Deformation vector field.

		Returns:
			_type_: _description_
		"""
		dimensions = torch.tensor(self.object_meta.shape).to(pytomography.device)
		new_coordinates = old_coordinates + DVF
		new_coordinates = 2/(dimensions-1)*new_coordinates - 1 
		return new_coordinates
		
	def _apply_dvf(self, DVF: torch.Tensor, object_i: torch.Tensor):
		"""Applies the deformation vector field to the object

		Args:
			DVF (torch.Tensor): Deformation vector field
			object_i (torch.Tensor): Old object.

		Returns:
			torch.Tensor: Deformed object.
		"""
		old_coordinates = self._get_old_coordinates()
		new_coordinates = self._get_new_coordinates(old_coordinates, DVF)
		return torch.nn.functional.grid_sample(object_i.unsqueeze(0), new_coordinates.unsqueeze(0).flip(dims=[-1]), align_corners=True)[0]

	def forward( 
		self,
		object_i: torch.Tensor,
	)-> torch.Tensor:
		"""Forward transform of deformation vector field

		Args:
			object_i (torch.Tensor): Original object.

		Returns:
			torch.Tensor: Deformed object corresponding to forward transform.
		"""
		if self.dvf_forward is None:
			return object_i
		else:
			return self._apply_dvf(self.dvf_forward, object_i)
	
	def backward( 
		self,
		object_i: torch.Tensor,
	)-> torch.Tensor:
		"""Backward transform of deformation vector field

		Args:
			object_i (torch.Tensor): Original object.

		Returns:
			torch.Tensor: Deformed object corresponding to backward transform.
		"""
		if self.dvf_backward is None:
			return object_i
		else:
			return self._apply_dvf(self.dvf_backward, object_i)