# Unstructured Interpolation
Given a 3D array of data, with an implicitly defined coordinate system, i.e., voxel coordinates,
and given a set of points in this coordinate system (not necessarily integer), 
I want to **interpolate the data at these given points**, which is also known as **unstructured interpolation**.

In PyTorch, `torch.nn.functional.grid_sample` is the easiest way to do this. 
However, exploiting the power of this function for our task is not straightforward.

## Usage
For a given 2D/3D array of data `values` of shape 1CHW(D) and a set of points `points` of shape ND,
run the following code to interpolate the data at the given points:

```python
output = unstructured_interpolation(values, points)
```

The returned tensor will be of shape NC.
