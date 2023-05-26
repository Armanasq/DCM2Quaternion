# DCM2Quaternion

The presented code implements a function called `dcm2quat` that performs a conversion operation between a batch of 3x3 rotation matrices and quaternions. The conversion is done using an analytical solution derived from the properties of quaternion algebra.

The input to the function is a NumPy array with shape `(batch_size, 3, 3)` containing the rotation matrices for each element in the batch. The output is a NumPy array with shape `(batch_size, 4)` containing the corresponding quaternions in the format `(w, x, y, z)`.

To perform the conversion, the function first initializes an empty NumPy array of size `(batch_size, 4)`, which will store the resulting quaternions. It then computes the trace of each rotation matrix in the batch and creates a mask to separate cases where the trace is positive or negative.

Next, the function calculates the scaling factor `S` for each element in the batch using the trace and the mask computed before. The individual components of the resulting quaternions are then calculated using vectorized operations based on the formula that relates rotation matrices and quaternions.

Finally, the function normalizes the resulting quaternions to unit length and returns the resulting array.

The equations used in the `dcm2quat` function are as follows:

Let `R` be a batch of 3x3 rotation matrices, with shape `(batch_size, 3, 3)`, and let `q` be a batch of quaternions in the format `(w, x, y, z)`, with shape `(batch_size, 4)`.

1. Compute the trace of each rotation matrix in the batch:
```
tr = np.trace(R, axis1=1, axis2=2)
```

2. Create a mask to separate cases where the trace is positive or negative:
```
mask = tr > 0
```

3. Compute the scaling factor `S` for each element in the batch using the trace and the mask:
```
S = np.sqrt(np.where(mask, tr + 1.0, np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))) * 2
```

4. Calculate the individual components of the resulting quaternions using vectorized operations based on the formula that relates rotation matrices and quaternions:
```
qx = np.where(mask, (R[:, 2, 1] - R[:, 1, 2]) / S, (R[:, 0, 1] + R[:, 1, 0]) / np.maximum(S, np.finfo(float).eps))
qy = np.where(mask, (R[:, 0, 2] - R[:, 2, 0]) / S, (R[:, 0, 2] + R[:, 2, 0]) / np.maximum(S, np.finfo(float).eps))
qz = np.where(mask, (R[:, 1, 0] - R[:, 0, 1]) / S, (R[:, 1, 2] + R[:, 2, 1]) / np.maximum(S, np.finfo(float).eps))
qw = np.where(mask, 0.25 * S, 0.25 * S)
```

5. Normalize the resulting quaternions to unit length:
```
q_norm = np.linalg.norm([qw, qx, qy, qz], axis=0)
q = np.stack([qw, qx, qy, qz], axis=1) / q_norm[:, np.newaxis]
```

Overall,

 the `dcm2quat` function provides an efficient and numerically stable way to perform the conversion between rotation matrices and quaternions for large batches of data.
