CurbFeatureSequence ExtendFeaturesContour(const CvGridMap1f& heightmap_cv,
                                          CurbFeatureSequence input_features,
                                          const CvGridMap1b& roadmask_cv)
{
  assert(heightmap_cv.GetNumXCells() == roadmask_cv.GetNumXCells());
  assert(heightmap_cv.GetNumYCells() == roadmask_cv.GetNumYCells());

  // Extract contours.
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(roadmask_cv.Image(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  // Draw the location of all existing features.
  cv::Mat1b features_image = cv::Mat::zeros(roadmask_cv.Image().size(), CV_8UC1);
  for (const CurbFeature& feat : input_features) {
    const cv::Point& feat_point_img = heightmap_cv.GlobalToCvPoint(feat.point);
    features_image.at<uint8_t>(feat_point_img) = kFeaturePixelValue;
  }
  // Output features.
  CurbFeatureSequence output_features;

  // For each contour, check if it intersects any curb features.
  for (size_t ci = 0; ci < contours.size(); ++ci) {
    const std::vector<cv::Point>& contour = contours.at(ci);

    // Make an image of contour points for fast lookup.
    cv::Mat1b contour_set = cv::Mat::zeros(heightmap_cv.Image().size(), CV_8UC1);
    cv::drawContours(contour_set, contours, ci, cv::Scalar(kContourPixelValue));

    // Find the features on this contour with minimum and maximum t value.
    int features_on_contour = 0;
    cv::Point min_t_point;
    cv::Point max_t_point;
    double min_t = std::numeric_limits<double>::max();
    double max_t = std::numeric_limits<double>::min();

    // For each input feature, see if it lies on the contour.
    for (size_t fi = 0; fi < input_features.size(); ++fi) {
      const cv::Point& feat_point_img = heightmap_cv.GlobalToCvPoint(input_features.at(fi).point);
      if (contour_set.at<uint8_t>(feat_point_img) == kContourPixelValue) {
        ++features_on_contour;
        const double t_observed = input_features.at(fi).t_observed;
        if (t_observed < min_t) {
          min_t_point = feat_point_img;
          min_t = t_observed;
        }
        if (t_observed > max_t) {
          max_t_point = feat_point_img;
          max_t = t_observed;
        }
      }
    }

    // Need 2 features on the contour to do traversal below.
    if (features_on_contour < 2) {
      continue;
    }

    // Find the min/max features on the contour.
    int min_t_contour_idx = -1;
    int max_t_contour_idx = -1;

    for (size_t point_idx = 0; point_idx < contour.size(); ++point_idx) {
      if (contour.at(point_idx) == min_t_point) {
        min_t_contour_idx = point_idx;
      }
      if (contour.at(point_idx) == max_t_point) {
        max_t_contour_idx = point_idx;
      }
    }

    assert(min_t_contour_idx != -1 && max_t_contour_idx != -1);

    // Figure out which direction to traverse the contour.
    int num_features_forward = 0;
    int num_features_reverse = 0;

    // Iterate forward.
    std::vector<cv::Point> forward_points;
    int fidx = min_t_contour_idx;
    while (fidx != max_t_contour_idx) {
      forward_points.emplace_back(contour.at(fidx));
      if (features_image.at<uint8_t>(contour.at(fidx)) == kFeaturePixelValue) {
        ++num_features_forward;
      }
      ++fidx;  // Increment and wrap around.
      if (fidx >= static_cast<int>(contour.size())) {
        fidx = 0;
      }
    }
    // Iterate in reverse.
    std::vector<cv::Point> reverse_points;
    int ridx = min_t_contour_idx;
    while (ridx != max_t_contour_idx) {
      reverse_points.emplace_back(contour.at(ridx));
      if (features_image.at<uint8_t>(contour.at(ridx)) == kFeaturePixelValue) {
        ++num_features_reverse;
      }
      --ridx;  // Decrement and wrap around.
      if (ridx < 0) {
        ridx = contour.size() - 1;
      }
    }

    // Choose the iter direction that hits the most features and add.
    std::vector<cv::Point>* best_points = &forward_points;
    if (num_features_reverse > num_features_forward) {
      best_points = &reverse_points;
    }

    for (const cv::Point& pt : *best_points) {
      CurbFeature feat;
      feat.point = heightmap_cv.ImageToGlobal(Vector2i(pt.y, pt.x));
      feat.height = heightmap_cv.Image().at<uint8_t>(pt);
      output_features.emplace_back(feat);
    }
  }

  return output_features;
}
