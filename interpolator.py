import torch
import torch.nn.functional as F


def _unstructured_interpolation_3d(values: torch.Tensor, points: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
	"""
	Sample `values` at given `points` using the interpolation `mode` of choice.

	:param values: the given tensor of shape 1 x C x H x W x D from which new values are interpolated from.
	:param points: coordinates of the given points of shape N_p x 3 at which new values are sampled.
	:param mode: used in F.grid_sample. Default: 'bilinear'.
	:return: sampled values of shape N_p x C

	Note: internally, `points` are first reshaped N_p x 3 -> 1 x 1 x 1 x N_p x 3.
	Then, F.grid_sample produces `output` of shape 1 x C x 1 x 1 x N_p.
	We finally obtained the sampled values by reshaping 1 x C x 1 x 1 x N_p -> N_p x C.

	Note: coordinates of the given `point` should be in the implicit coordinate system implied by `values`,
	i.e., $(x, y, z) \in [0, H] \times [0, W] \times [0, D] \subseteq \mathbb{R}^3$.
	Coordinates of `points` are normalized to [-1, 1].
	"""

	input: torch.Tensor = values
	grid: torch.Tensor = points[None, None, None, ...]
	grid[..., 0] = 2 * grid[..., 0] / (values.shape[2] - 1) - 1
	grid[..., 1] = 2 * grid[..., 1] / (values.shape[3] - 1) - 1
	grid[..., 2] = 2 * grid[..., 2] / (values.shape[4] - 1) - 1

	output: torch.Tensor = F.grid_sample(
		input,
		grid,
		mode=mode,
		padding_mode='border',
		align_corners=True,
	)

	return torch.squeeze(output, dim=(0, 2, 3)).permute(1, 0)


def _unstructured_interpolation_2d(values: torch.Tensor, points: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
	"""
	Sample `values` at given `points` using the interpolation `mode` of choice.

	:param values: the given tensor of shape 1 x C x H x W from which new values are interpolated from.
	:param points: coordinates of the given points of shape N_p x 2 at which new values are sampled.
	:param mode: used in F.grid_sample. Default: 'bilinear'.
	:return: sampled values of shape N_p x C

	Note: internally, `points` are first reshaped N_p x 2 -> 1 x 1 x N_p x 2.
	Then, F.grid_sample produces `output` of shape 1 x C x 1 x N_p.
	We finally obtained the sampled values by reshaping 1 x C x 1 x N_p -> N_p x C.

	Note: coordinates of the given `point` should be in the implicit coordinate system implied by `values`,
	i.e., $(x, y) \in [0, H] \times [0, W] \subseteq \mathbb{R}^2$.
	Coordinates of `points` are normalized to [-1, 1].
	"""

	input: torch.Tensor = values
	grid: torch.Tensor = points[None, None, ...]
	grid[..., 0] = 2 * grid[..., 0] / (values.shape[2] - 1) - 1
	grid[..., 1] = 2 * grid[..., 1] / (values.shape[3] - 1) - 1

	output: torch.Tensor = F.grid_sample(
		input,
		grid,
		mode=mode,
		padding_mode='border',
		align_corners=True,
	)

	return torch.squeeze(output, dim=(0, 2)).permute(1, 0)


def unstructured_interpolation(values: torch.Tensor, points: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
	"""
	Sample `values` at given `points` using the interpolation `mode` of choice.

	:param values: the given tensor of shape 1 x C x H x W (x D) from which new values are interpolated from.
	:param points: coordinates of the given points of shape N_p x dim at which new values are sampled.
	:param mode: used in F.grid_sample. Default: 'bilinear'.
	:return: sampled values of shape N_p x C

	Note: internally, `points` are first reshaped N_p x dim -> 1 x 1 (x 1) x N_p x dim.
	Then, F.grid_sample produces `output` of shape 1 x C x 1 (x 1) x N_p.
	We finally obtained the sampled values by reshaping 1 x C x 1 (x 1) x N_p -> N_p x C.

	Note: coordinates of the given `point` should be in the implicit coordinate system implied by `values`.
	Coordinates of `points` are normalized to [-1, 1].
	"""

	assert values.dim() in [4, 5], f'Unsupported dimension of `values`: {values.dim()}'

	assert points.shape[1] in [2, 3], f'Unsupported dimension of `points`: {points.shape[1]}'

	assert points.shape[1] == values.dim() - 2, \
		f'Incompatible dimension of `values` and `points`: {values.dim()} and {points.shape[1]}'

	if values.dim() == 5:
		return _unstructured_interpolation_3d(values, points, mode)
	elif values.dim() == 4:
		return _unstructured_interpolation_2d(values, points, mode)


if __name__ == '__main__':
	values: torch.Tensor = torch.tensor(
		[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
	).unsqueeze(0).unsqueeze(0).cuda()
	points: torch.Tensor = torch.tensor(
		[[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
	).cuda()

	results: torch.Tensor = unstructured_interpolation(values, points)
	print(results)

	values: torch.Tensor = torch.tensor(
		[[1.0, 2.0], [3.0, 4.0]]
	).unsqueeze(0).unsqueeze(0).cuda()
	points: torch.Tensor = torch.tensor(
		[[0.5, 0.5], [0.0, 0.5], [1.0, 0.5]]
	).cuda()

	results: torch.Tensor = unstructured_interpolation(values, points)
	print(results)
