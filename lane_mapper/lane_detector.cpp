#include "workstation/mapping/lane_mapper/lane_detector.hpp"

#include "mc/common/ocommon/stopwatch.hpp"

namespace ori {
namespace mapping {

#define OLOG_GROUP "lane_detector"

using Contour = std::vector<cv::Point>;
using ContourVector = std::vector<std::vector<cv::Point>>;

LaneDetector::LaneDetector(const Params& params) : params_(params)
{
  OINFO("Initialized Lane Detector!");
}

float PointDist(const cv::Point2f& a, const cv::Point2f& b)
{
  return std::sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

void LaneDetector::MaskRoadRoi(const cv::Mat1b& input_mat, cv::Mat1b* output_mat)
{
  int w = input_mat.cols;
  int h = input_mat.rows;

  // Points in clockwise order.
  int numPoints = 6;
  const cv::Point roiPoly[6] = {
      cv::Point2f(w / 3, h / 2), cv::Point2f(w - w / 3, h / 2),
      cv::Point2f(w, h - 250),   cv::Point2f(w, h),
      cv::Point2f(0, h),         cv::Point2f(0, h - 250),
  };
  cv::Mat1b mask = cv::Mat::zeros(input_mat.size(), CV_8UC1);
  cv::fillConvexPoly(mask, roiPoly, numPoints, 255);
  *output_mat = (input_mat & mask);
  // cv::namedWindow("debug", cv::WINDOW_NORMAL);
  // cv::imshow("debug", *output_mat);
  // cv::waitKey(0);
  return;
}

void LaneDetector::BirdsEyeTransform(const cv::Mat1b& input_mat, cv::Mat1b* output_mat)
{
  int w = input_mat.cols;
  int h = input_mat.rows;

  int heightMult = 4;

  // Corner order: TL, TR, BR, BL.
  // TODO: these numbers are all guesses, need to use camera extrinsics...
  // Computing a homography between the front left camera and a simulated
  // overhead camera might be more appropriate.
  const cv::Point2f srcRect[] = {
      cv::Point2f(760, h / 2), cv::Point2f(w - 760, h / 2), cv::Point2f(w, h - 150),
      cv::Point2f(0, h - 150),
  };

  // Get the maximum width and maximum height from source.
  float widthBottom = PointDist(srcRect[2], srcRect[3]);
  float widthTop = PointDist(srcRect[0], srcRect[1]);
  float heightLeft = PointDist(srcRect[0], srcRect[3]);
  float heightRight = PointDist(srcRect[1], srcRect[2]);
  float maxHeight = std::max(heightLeft, heightRight);
  float maxWidth = std::max(widthBottom, widthTop);

  const cv::Point2f dstRect[]{cv::Point2f(0, 0), cv::Point2f(maxWidth - 1, 0),
                              cv::Point2f(maxWidth - 1, heightMult * maxHeight - 1),
                              cv::Point2f(0, heightMult * maxHeight - 1)};

  // Contour cnt(srcRect, srcRect+4);
  // ContourVector cntVect = {cnt};
  // cv::Mat debug = input_mat.clone();
  // cv::drawContours(debug, cntVect, -1, cv::Scalar(0, 255, 0), 8);
  // cv::namedWindow("debug", cv::WINDOW_NORMAL);
  // cv::imshow("debug", debug);

  cv::Mat tf = cv::getPerspectiveTransform(srcRect, dstRect);

  // cv::Mat warped;
  cv::warpPerspective(input_mat, *output_mat, tf, cv::Size(maxWidth, heightMult * maxHeight),
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT);

  // cv::namedWindow("warped", cv::WINDOW_NORMAL);
  // cv::imshow("warped", warped);
  // cv::waitKey(1);
  return;
}

void LaneDetector::ApplyColorFilter(const cv::Mat3b& input_mat, cv::Mat1b* output_mat)
{
  // Split the HLS image into separate channels.
  cv::split(input_mat, channels);
  left_mat_hue = channels[0];
  left_mat_light = channels[1];
  left_mat_sat = channels[2];

  int yellow_hue_center = 39;

  left_mat_yellow_mask = (left_mat_hue >= (yellow_hue_center - 15))
                         & (left_mat_hue <= (yellow_hue_center + 15)) & (left_mat_light >= 85)
                         & (left_mat_light <= 200) & (left_mat_sat >= 115) & (left_mat_sat <= 255);

  left_mat_white_mask = (left_mat_light >= 175);

  *output_mat = (left_mat_yellow_mask | left_mat_white_mask);
  return;
}

void LaneDetector::ApplyEdgeFilter(const cv::Mat1b& input_mat, cv::Mat1b* output_mat)
{
  int ratio = 2;
  int ksize = 11;
  int sobel_ksize = 3;
  int low_thresh = 15;

  cv::Mat edges_mat;
  cv::blur(input_mat, edges_mat, cv::Size(ksize, ksize));
  cv::Canny(edges_mat, edges_mat, low_thresh, low_thresh * ratio, sobel_ksize);

  ContourVector contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(edges_mat, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE,
                   cv::Point(0, 0));

  std::vector<std::vector<cv::Point>> bboxes;
  bboxes.resize(contours.size());

  for (size_t ii = 0; ii < contours.size(); ii++) {
    cv::Point2f pts[4];
    cv::minAreaRect(contours[ii]).points(pts);

    bboxes[ii].resize(4);
    for (size_t jj = 0; jj < 4; jj++) {
      bboxes[ii][jj] = cv::Point(pts[jj]);
    }
  }

  // cv::Mat drawing_rects = cv::Mat::zeros(edges_mat.size(), CV_8UC3);
  // cv::drawContours(drawing_rects, bboxes, -1, cv::Scalar(0, 0, 255), -1, 8);

  // Draw the contours as a binary mask.
  // Make sure output is allocated.
  if (output_mat->rows <= 0 || output_mat->cols <= 0) {
    *output_mat = cv::Mat::zeros(input_mat.size(), CV_8UC1);
  }
  cv::drawContours(*output_mat, bboxes, -1, 255, -1);
  return;
}

void LaneDetector::ApplyMatchedFilter(const cv::Mat& input_mat, const math::Plane& ground_plane,
                                      const oimage::CameraIntrinsics& intrinsics,
                                      cv::Mat* filtered_x, cv::Mat* filtered_y)
{
  // Initialize output mats.
  *filtered_x = cv::Mat::zeros(input_mat.size(), CV_32F);
  *filtered_y = cv::Mat::zeros(input_mat.size(), CV_32F);

  Matrix3d K;
  K << intrinsics.fx, 0, intrinsics.cx, 0, intrinsics.fy, intrinsics.cy, 0, 0, 1;

  // Compute the filter size at bottom of image (largest).
  Matrix3d Kinv = K.inverse();
  Vector3d X_h = Kinv * Vector3d(0.5 * input_mat.cols, input_mat.rows, 1.0);
  double z_cam = -1 * ground_plane.D() / (X_h.transpose() * ground_plane.Normal());
  int lane_half_width_px = std::floor(0.5 * params_.lane_width * intrinsics.fx / z_cam);

  int maxRowIndex = input_mat.rows;  // Exclusive.
  int validFilterRows =
      std::ceil(-2 * ground_plane.D() / (params_.lane_width * intrinsics.fx * Kinv(0, 0)));
  int minRowIndex = maxRowIndex - validFilterRows;  // Inclusive.

  // TODO: assert that validFilterRows >= 1!

  while (lane_half_width_px >= 1) {
    // Find the range of rows for the next kernel size.
    minRowIndex = maxRowIndex - validFilterRows;

    cv::Mat filter_x = cv::Mat::ones(cv::Size(4 * lane_half_width_px + 1, 1), CV_32F);
    cv::Mat filter_y = cv::Mat::ones(cv::Size(1, 4 * lane_half_width_px + 1), CV_32F);

    filter_x.colRange(0, lane_half_width_px).setTo(-1);
    filter_x.colRange(3 * lane_half_width_px + 1, 4 * lane_half_width_px + 1).setTo(-1);

    filter_y.rowRange(0, lane_half_width_px).setTo(-1);
    filter_y.rowRange(3 * lane_half_width_px + 1, 4 * lane_half_width_px + 1).setTo(-1);

    // Apply x filter only along the row.
    cv::filter2D(input_mat.rowRange(minRowIndex, maxRowIndex),
                 filtered_x->rowRange(minRowIndex, maxRowIndex), CV_32F, filter_x,
                 cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    // Apply the y filter along a strip.
    cv::filter2D(input_mat.rowRange(minRowIndex, maxRowIndex),
                 filtered_y->rowRange(minRowIndex, maxRowIndex), CV_32F, filter_y,
                 cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    // Move on to the next kernel size, and shift the rows to apply filter to.
    maxRowIndex = minRowIndex;
    lane_half_width_px -= 1;
  }

  return;
}

void LaneDetector::GetInterestPoints(const cv::Mat& input_mat, std::vector<cv::Point>* ipoints,
                                     const int axis, const float threshold)
{
  // Axis 0 is rows, axis 1 is cols.
  int iterLimit = axis ? input_mat.cols : input_mat.rows;
  double minVal, maxVal;
  int minIdx, maxIdx;

  // Get maximum for each row/col depending on axis.
  for (int ii = 0; ii < iterLimit; ii++) {
    cv::Mat submat = axis ? input_mat.colRange(ii, ii + 1) : input_mat.rowRange(ii, ii + 1);
    cv::minMaxIdx(submat, &minVal, &maxVal, &minIdx, &maxIdx);

    if (maxVal > threshold) {
      cv::Point ip = axis ? cv::Point(ii, maxIdx) : cv::Point(maxIdx, ii);
      ipoints->push_back(ip);
    }
  }

  return;
}

void LaneDetector::GetiPointOrientations(const std::vector<cv::Point>& ipoints,
                                        const cv::Mat& F_xx, const cv::Mat& F_yy,
                                        const cv::Mat& F_xy, const cv::Mat& F_yx,
                                        std::vector<Vector2d>* orientations)
{
  orientations->resize(ipoints.size());

  // Eigen::Matrix3d hessian;
  // for (size_t ii = 0; ii < ipoints.size(); ii++) {
  //   cv::Point pt = ipoints[ii];
  //   hessian << F_xx.at<double>(pt), F_xy.at<double>(pt), F_yx.at<double>(pt), F_yy.at<double>(pt);
  //   Eigen::EigenSolver<Matrix2d> solver(hessian, true);

  //   Vector2d eigVals = solver.eigenvalues();
  //   Matrix2d eigVects = solver.eigenvectors();

  //   // float maxEigVal = (eigVals(0) > eigVals(1)) ? eigVals[0] : eigVals[1];
  //   Vector2d maxEigVect = (eigVals(0) > eigVals(1)) ? eigVects.col(0) : eigVects.col(1);
  //   (*orientations)[ii] = maxEigVect;
  // }
  for (size_t ii = 0; ii < ipoints.size(); ii++) {
    cv::Point pt = ipoints[ii];
    float fxx = F_xx.at<double>(pt);
    float fxy = F_xy.at<double>(pt);
    float fyx = F_yx.at<double>(pt);
    float fyy = F_yy.at<double>(pt);

    // https://math.stackexchange.com/questions/395698/fast-way-to-calculate-eigen-of-2x2-matrix-using-a-formula
    float maxEigVal = 0.5 * ((fxx + fyy) + std::sqrt(pow(fxx+fyy, 2) - 4*(fxx*fyy - fxy*fyx)));
    Vector2d maxEigVect;
    maxEigVect << fxy, (maxEigVal - fxx);
    (*orientations)[ii] = maxEigVect;
  }

  return;
}

void ComputeHessian(const cv::Mat input_mat, cv::Mat* F_xx, cv::Mat* F_yy,
                    cv::Mat* F_xy, cv::Mat* F_yx, cv::Mat* F_x, cv::Mat* F_y)
{
  // Compute 2nd spatial derivatives.
  cv::Sobel(input_mat, *F_x, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(input_mat, *F_y, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(*F_x, *F_xx, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(*F_x, *F_xy, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(*F_y, *F_yy, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(*F_y, *F_yx, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);

  return;
}

void LaneDetector::ExtractLaneFeatures(const oimage::OImagePtrVector& input_oimages,
                                       const math::Plane& ground_plane,
                                       const oimage::CameraCalibrationPtrVectorPtr& stereo_calib,
                                       std::vector<curves2d::CubicHermiteSpline>* /*lane_features*/)
{
  // Convert to cv::Mat in HLS format.
  left_mat_bgr = oimage::CVWrapOImage(input_oimages[0]);
  cv::cvtColor(left_mat_bgr, left_mat_mono, cv::COLOR_BGR2GRAY);

  // Remove parts of image outside ROI in place.
  // TODO: use the horizon line instead of hardcoded trapezoid.
  // MaskRoadRoi(left_mat_mono, &left_mat_mono);

  // Apply the matched filter to extract horizontal and vertical lanes.
  cv::Mat matched_filter_x, matched_filter_y;
  cv::Mat left_mat_mono_float;
  left_mat_mono.convertTo(left_mat_mono_float, CV_32FC1);
  ApplyMatchedFilter(left_mat_mono_float, ground_plane, (*stereo_calib)[0]->intrinsics,
                     &matched_filter_x, &matched_filter_y);

  cv::Mat matched_filter_response;
  cv::addWeighted(matched_filter_x, 0.5, matched_filter_y, 0.5, 0.0, matched_filter_response);

  // cv::imshow("matchedx", matched_filter_x);
  // cv::imshow("matchedy", matched_filter_y);
  cv::imshow("matched", matched_filter_response);

  // Get local maxima in horizontal and vertical images.
  std::vector<cv::Point> ipoints;
  // double globMinVal, globMaxVal;
  double ipointThresh = 500;

  // cv::minMaxLoc(matched_filter_x, &globMinVal, &globMaxVal);
  GetInterestPoints(matched_filter_x, &ipoints, 0, ipointThresh);

  // cv::minMaxLoc(matched_filter_y, &globMinVal, &globMaxVal);
  GetInterestPoints(matched_filter_y, &ipoints, 1, ipointThresh);

  printf("Num ipoints: %zu\n", ipoints.size());

  // Blur and compute Hessian.
  cv::blur(matched_filter_response, matched_filter_response, cv::Size(11, 11));
  cv::Mat F_x, F_y, F_xx, F_yy, F_xy, F_yx;
  ComputeHessian(matched_filter_response, &F_xx, &F_yy, &F_xy, &F_yx, &F_x, &F_y);

  std::vector<Vector2d> orientations;
  GetiPointOrientations(ipoints, F_xx, F_yy, F_xy, F_yx, &orientations);

  cv::Mat debug = cv::Mat::zeros(matched_filter_response.size(), CV_8UC1); // matched_filter_response.clone();
  for (size_t ii = 0; ii < ipoints.size(); ii++) {

    Vector2d normal = (orientations[ii](0) < orientations[ii](1)) ?
      orientations[ii] / orientations[ii](0) : orientations[ii] / orientations[ii](1);

    normal *= 100;
    cv::Point normal_scaled((int)normal(0), (int)normal(1));
    cv::line(debug, ipoints[ii], ipoints[ii] + normal_scaled, 255, 1);
  }
  // for (auto& ip : ipoints) {
  //   cv::circle(debug, ip, 3, 255);
  // }
  cv::imshow("debug", debug);

  cv::waitKey(1);

  // Store the direction at each interest point...

  // Distance transform image...

  // Fit CH splines to interest points...
  return;
}

// void LaneDetector::ProcessImages(const oimage::OImagePtrVector& input_oimages,
//                                  oimage::OImagePtr detection_oimage)
// {
//   // Convert oimages to BGR format.
//   oimage::OImagePtrVector input_oimages_bgr = oimage::ConvertImages(
//     input_oimages, oimage::FourCC::BGR, allocator_);

//   // Rectify the images (note: size will change).
//   oimage::OImagePtrVector rectified_oimages_bgr;
//   rectifier_->RectifyImages(input_oimages_bgr, &rectified_oimages_bgr, allocator_);

//   // Convert to cv::Mat in HLS format.
//   left_mat_bgr = oimage::CVWrapOImage(rectified_oimages_bgr[0]);
//   cv::cvtColor(left_mat_bgr, left_mat_hls, cv::COLOR_BGR2HLS);

//   cv::Mat1b left_mat_mono;
//   cv::cvtColor(left_mat_bgr, left_mat_mono, cv::COLOR_BGR2GRAY);

//   // TODO: transform "works" but might not be useful in the long run...
//   // BirdsEyeTransform(left_mat_mono, &left_mat_mono);

//   // Remove parts of image outside ROI (road trapezoid).
//   MaskRoadRoi(left_mat_mono, &left_mat_mono);

//   // cv::imshow("left_mono_road", left_mat_mono);

//   cv::Mat matched_filter_x, matched_filter_y;
//   ApplyMatchedFilter(left_mat_mono, &matched_filter_x, &matched_filter_y);

//   cv::Mat1b roi_mask;
//   ApplyEdgeFilter(left_mat_mono, &roi_mask);

//   // cv::namedWindow("edge_filtered", cv::WINDOW_NORMAL);
//   // cv::imshow("edge_filtered", roi_mask);

//   // Apply the mask to HLS image and then filter by color.
//   cv::Mat3b left_mat_hls_roi;
//   left_mat_hls.copyTo(left_mat_hls_roi, roi_mask);
//   // cv::namedWindow("hls_masked", cv::WINDOW_NORMAL);
//   // cv::imshow("hls_masked", left_mat_hls_roi);

//   ApplyColorFilter(left_mat_hls_roi, &roi_mask);

//   // ApplyColorFilter(left_mat_hls, &roi_mask);
//   // ApplyEdgeFilter(roi_mask, &roi_mask);

//   cv::namedWindow("final", cv::WINDOW_AUTOSIZE);
//   cv::imshow("final", roi_mask);
//   cv::waitKey(1);

//   // TODO: assign the output to something useful
//   detection_oimage = oimage::CVWrapMat(&roi_mask, oimage::FourCC::GRAY);
//   return;
// }

}  // namespace mapping
}  // namespace ori