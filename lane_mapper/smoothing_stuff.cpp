template <typename T>
T NormalPdf(T x, T m, T s)
{
  static const T inv_sqrt_2pi = 0.3989422804014327;
  T a = (x - m) / s;
  return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
}

double ScoreWaypoints(const VecVector2d& waypoints, const size_t /*updated_idx*/,
                      const LaneModel& lane_model, const double width_mean, const double width_std)
{
  // Create candidate spline.
  const curves2d::ASplinePtr center_spline = curves2d::ASpline::Create(waypoints);
  const curves2d::CurveUtilWrapperPtr center_wrapper =
      curves2d::CurveUtilWrapper::Create(center_spline, kDefaultCurveSamplingDistance);

  double score = 1.0;
  const double scoring_spacing = 1.0;  // TODO: milo

  // For each detected spline, compute the "probability" of the lane given the spline.
  for (const LaneFitResult& fit : lane_model.LeftLanelines()) {
    double avg_half_width = 0;
    const VecVector2d sample_points =
        curves2d::SampleUniformSpacing(fit.spline, scoring_spacing).points;

    for (const Vector2d pt : sample_points) {
      avg_half_width += (pt - center_wrapper->ProjectPointOntoCurve(pt).curve_at_t).norm();
    }
    avg_half_width /= (double)sample_points.size();

    score *= NormalPdf(avg_half_width, 0.5 * width_mean, 0.5 * width_std);
  }
  for (const LaneFitResult& fit : lane_model.RightLanelines()) {
    double avg_half_width = 0;
    const VecVector2d sample_points =
        curves2d::SampleUniformSpacing(fit.spline, scoring_spacing).points;

    for (const Vector2d pt : sample_points) {
      avg_half_width += (pt - center_wrapper->ProjectPointOntoCurve(pt).curve_at_t).norm();
    }
    avg_half_width /= (double)sample_points.size();

    score *= NormalPdf(avg_half_width, width_mean, width_std);
  }

  return score;
}

VecVector2d LaneDetector::RefineWaypoints(const LaneModel& lane,
                                          const VecVector2d& initial_waypoints,
                                          const double nominal_width,
                                          const LaneFittingParams& /*fitting_params*/)
{
  const size_t max_refinement_iters = 2;
  VecVector2d waypoints_estimate = initial_waypoints;
  curves2d::ParametricCurveConstPtr centerline_estimate =
      curves2d::ASpline::Create(initial_waypoints);

  double search_dist_left = 0.5;
  double search_dist_right = 0.5;
  double search_step = 0.1;
  const double width_std = 0.4;

  for (size_t ii = 0; ii < max_refinement_iters; ++ii) {
    // Recreate the centerline estimate using the current waypoints estimate to get better normals.
    centerline_estimate = curves2d::ASpline::Create(waypoints_estimate);

    for (size_t jj = 0; jj < waypoints_estimate.size(); ++jj) {
      double best_score = 0;

      // Get the normal at waypoints.
      const Vector2d normal = centerline_estimate->Normal(jj).normalized();
      VecVector2d test_waypoints = waypoints_estimate;

      for (double offset = -1.0 * search_dist_left; offset <= search_dist_right;
           offset += search_step) {
        const Vector2d test_waypoint = waypoints_estimate.at(jj) + offset * normal;
        test_waypoints.at(jj) = test_waypoint;

        const double score = ScoreWaypoints(test_waypoints, jj, lane, nominal_width, width_std);

        // If this waypoint position is the best so far, update the estimate.
        if (score > best_score) {
          best_score = score;
          waypoints_estimate.at(jj) = test_waypoint;
        }
      }
    }
    // Search at a finer resolution, but with lower bounds.
    search_dist_left /= 2.0;
    search_dist_right /= 2.0;
    search_step /= 2.0;
  }
  return waypoints_estimate;
}
