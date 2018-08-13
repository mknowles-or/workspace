std::vector<CurbFeatureSequence> ExtractFeaturesFloodFill(const CvGridMap1f& heightmap_cv,
                                                          const CvGridMap1b& road_surface_mask,
                                                          const RoadCurveSegment& segment,
                                                          const CurbFeatureOptions& /*options*/)
{
  // Output features.
  CurbFeatureSequence features_left;
  CurbFeatureSequence features_right;

  const curves2d::ASplinePtr& spline = segment.curve;

  // Make a mask for floodfill boundaries.
  cv::Mat1b mask =
      cv::Mat::zeros(heightmap_cv.GetNumXCells() + 2, heightmap_cv.GetNumYCells() + 2, CV_8UC1);

  const Vector2d& start = spline->Evaluate(spline->GetMinT());
  const Vector2d& end = spline->Evaluate(spline->GetMaxT());
  const Vector2d& normal_start = spline->Normal(spline->GetMinT()).normalized();
  const Vector2d& normal_end = spline->Normal(spline->GetMaxT()).normalized();

  // Get image line endpoints.
  const Vector2d s1 = start - 50 * normal_start;
  const Vector2d s2 = start + 50 * normal_start;
  const Vector2d e1 = end - 50 * normal_end;
  const Vector2d e2 = end + 50 * normal_end;
  const Vector2i& s1p = heightmap_cv.GlobalToImage(s1);
  const Vector2i& s2p = heightmap_cv.GlobalToImage(s2);
  const Vector2i& e1p = heightmap_cv.GlobalToImage(e1);
  const Vector2i& e2p = heightmap_cv.GlobalToImage(e2);

  // Draw lines at the endpoints of the spline as boundaries.
  cv::line(mask, cv::Point(s1p.y(), s1p.x()), cv::Point(s2p.y(), s2p.x()), 255);
  cv::line(mask, cv::Point(e1p.y(), e1p.x()), cv::Point(e2p.y(), e2p.x()), 255);
  const cv::Point seed_point = heightmap_cv.GlobalToCvPoint(spline->Evaluate(0.5 * (spline->GetMaxT() - spline->GetMinT())));
 
  cv::circle(mask, seed_point, 5, 255);
  cv::imshow("mask", mask);

  cv::floodFill(road_surface_mask.Image(), mask, seed_point, cv::Scalar(1), /*rect=*/0,
                cv::Scalar(0), cv::Scalar(0), kFloodFillFlags);

  cv::imshow("input_image", road_surface_mask.Image());

  cv::imshow("floodfill_result", mask);
  cv::waitKey(0);

  const std::vector<CurbFeatureSequence> features = { features_left, features_right };
  return features;
}


const Vector2d& p_start = spline->Evaluate(spline->GetMinT());
  const Vector2d& p_end = spline->Evaluate(spline->GetMaxT());
  const Vector2d& normal_start = spline->Normal(spline->GetMinT()).normalized();
  const Vector2d& normal_end = spline->Normal(spline->GetMaxT()).normalized();

  // Get image line endpoints.
  const Vector2d s1 = p_start - 50 * normal_start;
  const Vector2d s2 = p_start + 50 * normal_start;
  const Vector2d e1 = p_end - 50 * normal_end;
  const Vector2d e2 = p_end + 50 * normal_end;
  const cv::Point& sm = heightmap_cv.GlobalToCvPoint(p_start);
  const cv::Point& em = heightmap_cv.GlobalToCvPoint(p_end);
  const cv::Point& s1p = heightmap_cv.GlobalToCvPoint(s1);
  const cv::Point& s2p = heightmap_cv.GlobalToCvPoint(s2);
  const cv::Point& e1p = heightmap_cv.GlobalToCvPoint(e1);
  const cv::Point& e2p = heightmap_cv.GlobalToCvPoint(e2);

  
  // // Find all contours on the left.
  // for (int i = 0; i < it_start_left.count, ++i, ++it_start_left) {
  //   const cv::Point& pos = it_start_left.pos();
  //   if (intersections.at<uint8_t>(pos) > 0) {
  //     const size_t index = static_cast<uint8_t>(intersections.at<uint8_t>(pos));
  //     left_contours.emplace_back(contours.at(idx));
  //     left_start_points.emplace_back(pos);
  //   }
  // }

  // // Find all contours on the right.
  // for (int i = 0; i < it_start_left.count, ++i, ++it_start_left) {
  //   const cv::Point& pos = it_start_left.pos();
  //   if (intersections.at<uint8_t>(pos) > 0) {
  //     const size_t index = static_cast<uint8_t>(intersections.at<uint8_t>(pos));
  //     left_contours.emplace_back(contours.at(idx));
  //     left_start_points.emplace_back(pos);
  //   }
  // }

  // // Draw each intersection between a contour and the projection lines with value 255.
  // for (const std::vector<cv::Point>& cnt : left_contours) {

  // }
