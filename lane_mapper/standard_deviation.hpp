/**
 * @brief Compute the standard deviation of values in a vector.
 */
template <typename T>
T StandardDeviation(const std::vector<T>& v)
{
  T mean = 0;
  for (const T& vi : v) {
    mean += vi;
  }
  mean /= static_cast<T>(v.size());

  T ssd = 0;
  for (const T& vi : v) {
    ssd += pow(vi - mean, 2);
  }
  return std::sqrt(ssd / static_cast<T>(v.size()));
}
