/**
 * @brief Generate a 2D point density map.
 */
perception::UniformGridMap<float> GenerateDensityMap(
    const perception::UniformGridMap<perception::ScalarAccumulator>& height_accumulator_map)
{
  perception::UniformGridMap<float> density_map(height_accumulator_map, 0.0f);

  for (int xi = 0; xi < density_map.GetNumXCells(); ++xi) {
    for (int yi = 0; yi < density_map.GetNumYCells(); ++yi) {
      const Vector2i cell_coord = Vector2i(xi, yi);

      // If height values were recorded at this cell, take max.
      if (!height_accumulator_map.GetCell(cell_coord).IsEmpty()) {
        const float density_at_cell =
            static_cast<float>(height_accumulator_map.GetCell(cell_coord).GetSize());
        *density_map.GetMutableCell(cell_coord) = density_at_cell;
      }
    }
  }
  return density_map;
}
