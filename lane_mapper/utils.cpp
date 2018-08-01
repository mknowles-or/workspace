/**
 * @brief Fit a spline to a set of points using inliers as the only scoring criteria.
 */
curves2d::ASplinePtr FitSplineRansac(const std::vector<Vector2d>& points, bool* success,
                                     const int N, const float inlierDist = 0.2,
                                     const float successPct = 0.8, const int maxIters = 100);

curves2d::ASplinePtr FitSplineRansac(const std::vector<Vector2d>& points, bool* success,
                                     const int N, const float inlierDist, const float successPct,
                                     const int maxIters)
{
  float bestNumInliers = 0;
  ori::curves2d::ASplinePtr bestSpline;

  std::vector<int> samples;
  for (int iter = 0; iter < maxIters; iter++) {
    samples.clear();
    SampleUniformSpacing(N, 0, points.size(), &samples);

    curves2d::ASplinePtr candidate = curves2d::ASpline::Create();
    for (const int& idx : samples) {
      candidate->Add(points[idx]);
    }

    int numInliers = 0;

    for (const Vector2d& pt : points) {
      Vector2d closest =
          curves2d::CurveUtilWrapper(candidate, 0.1).ProjectPointOntoCurve(pt, false).curve_at_t;
      float dist = (closest - pt).norm();

      if (dist < inlierDist) {
        numInliers++;
      }
    }
    // Update best spline if improved.
    if (numInliers > bestNumInliers) {
      bestNumInliers = numInliers;
      bestSpline = candidate;

      // Early stopping condition.
      if (bestNumInliers > (successPct * points.size())) {
        *success = true;
        return bestSpline;
      }
    }
  }

  return bestSpline;
}