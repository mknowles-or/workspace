perception::UniformGridMap<float> ComputeGridmapGradient(
    const perception::UniformGridMap<float>& gridmap)
{
  // Create x and y sobel kernels.
  Matrix3d sobel_x, sobel_y;
  sobel_x.row(0) << 1, 2, 1;
  sobel_x.row(1) << 0, 0, 0;
  sobel_x.row(2) << -1, -2, -1;

  sobel_y.row(0) << 1, 0, -1;
  sobel_y.row(1) << 2, 0, -2;
  sobel_y.row(2) << 1, 0, -1;

  // Apply the operator with convolution.
  const perception::UniformGridMap<float>& gradient_x =
      perception::ConvolveGridMap(gridmap, sobel_x);

  const perception::UniformGridMap<float>& gradient_y =
      perception::ConvolveGridMap(gridmap, sobel_y);

  // Compute the magnitude of gradient.
  perception::UniformGridMap<float> magnitude(gridmap, 0.0f);
  for (int i = 0; i < gridmap.GetNumXCells(); ++i) {
    for (int j = 0; j < gridmap.GetNumYCells(); ++j) {
      float* cell = magnitude.GetMutableCell(i, j);
      *cell = Vector2f(gradient_x.GetCell(i, j), gradient_y.GetCell(i, j)).norm();
    }
  }

  return magnitude;
}
