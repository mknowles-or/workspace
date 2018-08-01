// Copyright [2018] Optimus Ride Inc.

#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz/viz3d.hpp>

#include "mc/common/curves2d/curve_util_wrapper.hpp"
#include "mc/common/curves2d/intersection.hpp"
#include "mc/common/ocommon/olog.hpp"
#include "mc/common/omath/intersection.hpp"
#include "mc/common/omath/plane.hpp"
#include "mc/common/omath/rotations.hpp"
#include "workstation/mapping/lane_mapper/lane_detector.hpp"
#include "workstation/mapping/lane_mapper/lane_fitting.hpp"

namespace ori {
namespace mapping {

#define OLOG_GROUP "lane_detector"

/**
 * @brief Convenience function for setting pixel color in a Mat3b.
 */
cv::Vec3b Scalar2Vec3b(const cv::Scalar& color)
{
  return cv::Vec3b(color(0), color(1), color(2));
}

void DrawSpline(cv::Mat3b img, const curves2d::ASplinePtr& spline, const LocalizationMapPtr& map,
                const Vector2i& min_cell_coord, const double cell_size_meters = 0.2,
                const cv::Scalar& controlPtColor = cv::viz::Color::yellow(),
                const cv::Scalar& lineColor = cv::viz::Color::blue())
{
  curves2d::CurveUtilWrapper wrapper(spline, cell_size_meters);
  curves2d::CurvePointSamples samples = wrapper.GetSamples();
  std::vector<Vector2d> controlPoints = spline->GetControlPoints();

  // Draw line along spline using sampled points.
  for (const Vector2d& pt : samples.points) {
    cv::Point2i imgPt = Global2ImageCoords(pt, map, min_cell_coord);
    img.at<cv::Vec3b>(imgPt) = Scalar2Vec3b(lineColor);
  }

  // Mark the control points.
  for (const Vector2d& pt : controlPoints) {
    cv::Point2i imgPt = Global2ImageCoords(pt, map, min_cell_coord);
    cv::circle(img, imgPt, 2, controlPtColor, -1);
  }

  return;
}

void DrawLane(cv::Mat3b img, const LaneModel& lane, const LocalizationMapPtr& map,
              const Vector2i& min_cell_coord)
{
  // Draw left lane points in red.
  std::pair<VecVector3d, VecVector2d> left_flattened = lane.GetLeftPointsFlattened();
  for (const Vector2d& pt : left_flattened.second) {
    cv::Point2i img_pt = Global2ImageCoords(pt, map, min_cell_coord);
    cv::circle(img, img_pt, 1, cv::viz::Color::lime());
  }

  // Draw right lane points in blue.
  std::pair<VecVector3d, VecVector2d> right_flattened = lane.GetRightPointsFlattened();
  for (const Vector2d& pt : right_flattened.second) {
    cv::Point2i img_pt = Global2ImageCoords(pt, map, min_cell_coord);
    cv::circle(img, img_pt, 1, cv::viz::Color::teal());
  }

  return;
}

void DrawLaneFit(cv::Mat3b img, const LaneFitResult& fit, const LocalizationMapPtr& map,
                 const Vector2i& min_cell_coord)
{
  // Control points in blue, splines in yellow.
  DrawSpline(img, fit.spline, map, min_cell_coord, 0.2, cv::viz::Color::yellow(),
             cv::viz::Color::blue());

  // Outliers in red, inliers in green.
  for (const Vector2d& pt : fit.inliers) {
    cv::circle(img, Global2ImageCoords(pt, map, min_cell_coord), 1, cv::viz::Color::green());
  }
  for (const Vector2d& pt : fit.outliers) {
    cv::circle(img, Global2ImageCoords(pt, map, min_cell_coord), 1, cv::viz::Color::red());
  }

  return;
}

void DrawSegments(cv::Mat3b img, const std::vector<TrajectorySegment>& segments,
                  const std::vector<FrameUpdate>& trajectory, const LocalizationMapPtr& map,
                  const Vector2i& min_cell_coord)
{
  std::vector<cv::Scalar> colors = {cv::viz::Color::pink(), cv::viz::Color::yellow(),
                                    cv::viz::Color::teal()};

  int ctr = 0;
  for (const TrajectorySegment& segment : segments) {
    cv::Scalar color = segment.type == LaneType::Curve ? colors[2] : colors[ctr % 2];
    ctr++;
    for (int ii = segment.start; ii <= segment.end; ii++) {
      RigidTransform global_from_body = trajectory[ii].transform;
      cv::Point2i pt = Body2ImageCoords(Vector3d(0, 0, 0), global_from_body, map, min_cell_coord);
      cv::circle(img, pt, 1, color);
    }
  }

  return;
}

double EstimateWidth(const VecVector2d& points, const VecVector3d& poses)
{
  double total = 0;
  if (points.size() != poses.size()) {
    std::runtime_error("EstimateWidth: poses and points should have same size");
  }
  for (size_t ii = 0; ii < points.size(); ii++) {
    total += (points[ii] - poses[ii].head<2>()).norm();
  }
  return (total / (double)points.size());
}

template <typename T>
std::vector<T> SubsetByIndices(const std::vector<T>& input, const std::vector<size_t>& indices)
{
  std::vector<T> output;
  for (const size_t& idx : indices) {
    output.push_back(input.at(idx));
  }
  return output;
}

void LaneDetector::PreprocessMapImage(const cv::Mat1b& input_mat, cv::Mat1b* output_mat)
{
  *output_mat = cv::Mat::zeros(input_mat.size(), CV_8UC1);

  return;
}

void LaneDetector::SubdivideTrajectory(const std::vector<FrameUpdate>& trajectory,
                                       const double local_spline_length, const int pts_per_spline,
                                       const double straight_max_curvature,
                                       std::vector<TrajectorySegment>* segments)
{
  segments->clear();

  TrajectorySegment currentSegment(0, 0, LaneType::Straight);

  // Stores a small piece of spline along the trajectory.
  curves2d::ASplinePtr localSpline = curves2d::ASpline::Create();

  Vector2d prevPosition;
  for (size_t ii = 0; ii < trajectory.size(); ii++) {
    RigidTransform global_from_body = trajectory[ii].transform;
    Vector2d currPosition = Body2GlobalCoords(Vector3d(0, 0, 0), global_from_body);

    if (ii == 0) {
      prevPosition = currPosition;
      continue;
    }
    // If we've moved enough, add another control point.
    float dist = (currPosition - prevPosition).norm();

    if (dist > (local_spline_length / pts_per_spline) || localSpline->Empty()) {
      localSpline->Add(currPosition);
      prevPosition = currPosition;

      double avgCurvature = 0;
      double length = curves2d::CurveUtilWrapper(localSpline, 0.101).GetCurveArcLength();

      if (length >= local_spline_length) {
        // Compute average curvature on local spline.
        int tctr = 0;
        for (double t = localSpline->GetMinT(); t < localSpline->GetMaxT();
             t += length / pts_per_spline) {
          avgCurvature += localSpline->Curvature(t);
          tctr++;
        }
        avgCurvature = fabs(avgCurvature / tctr);

        // Either continue the current segment or start a new one, based on curvature.
        if (currentSegment.type == LaneType::Straight) {
          if (avgCurvature < straight_max_curvature) {
            currentSegment.end = ii;
            currentSegment.length += length;
          } else {
            segments->push_back(currentSegment);
            currentSegment = TrajectorySegment(segments->back().end + 1, ii, LaneType::Curve);
          }
        } else if (currentSegment.type == LaneType::Curve) {
          if (avgCurvature >= straight_max_curvature) {
            currentSegment.end = ii;
            currentSegment.length += length;
          } else {
            segments->push_back(currentSegment);
            currentSegment = TrajectorySegment(segments->back().end + 1, ii, LaneType::Straight);
          }
        }
        // Start a new spline.
        localSpline = curves2d::ASpline::Create();
      }
    }
  }
  segments->push_back(currentSegment);

  return;
}

bool PolylinePolylineIntersect(const VecVector2d& poly1, const VecVector2d poly2, Vector2d* point,
                               double min_angle = 0.0)
{
  for (size_t ii = 0; ii < poly1.size() - 1; ii++) {
    for (size_t jj = 0; jj < poly2.size() - 1; jj++) {
      Vector2d v1 = (poly1[ii + 1] - poly1[ii]);
      Vector2d v2 = (poly2[jj + 1] - poly2[jj]);
      double angle = std::acos(v1.dot(v2) / (v1.norm() * v2.norm()));

      if (fabs(angle) >= min_angle) {
        bool doIntersect = math::LineSegmentLineSegmentIntersect(poly1[ii], poly1[ii + 1],
                                                                 poly2[jj], poly2[jj + 1], point);
        if (doIntersect) {
          return true;
        }
      }
    }
  }
  return false;
}

void LaneDetector::SplitAtIntersections(const std::vector<Vector3d>& poses,
                                        const std::vector<TrajectorySegment>& input_segments,
                                        const double /*truncate_dist*/,
                                        std::vector<TrajectorySegment>* output_segments)
{
  output_segments->clear();
  double spacing = 50;  // meters

  std::vector<VecVector2d> polylines;
  std::vector<TrajectorySegment> straight_segments;

  // Get all straight segments and break up into polylines.
  for (const TrajectorySegment& segment : input_segments) {
    if (segment.type == LaneType::Straight) {
      straight_segments.push_back(segment);
      VecVector2d poly;
      poly.push_back(poses[segment.start].head<2>());  // Include first point.

      // Add a new endpoint every spacing interval.
      double dist = 0;
      for (int ii = segment.start + 1; ii < segment.end; ii++) {
        dist += (poses[ii].head<2>() - poly.back()).norm();
        if (dist > spacing) {
          poly.push_back(poses[ii].head<2>());
          dist = 0;
        }
      }
      poly.push_back(poses[segment.end].head<2>());  // Include last point.
      polylines.push_back(poly);
    }
  }

  size_t thisPolyIdx = 0;

  // Iterate through all segments (including curves) to maintain ordering.
  for (const TrajectorySegment& segment : input_segments) {
    if (segment.type == LaneType::Straight) {
      std::vector<int> splitIndices;
      splitIndices.push_back(segment.start);

      // Find all points where segment needs to be split.
      for (size_t ii = 0; ii < polylines.size(); ii++) {
        if (ii != thisPolyIdx) {  // Don't compare against self.
          Vector2d intersectPt;

          // Gets only the first intersection point.
          bool doIntersect = PolylinePolylineIntersect(polylines[thisPolyIdx], polylines[ii],
                                                       &intersectPt, 3.14159 / 6);

          if (doIntersect) {
            // Find the nearest pose index to the intersection point on both polylines.
            int splitIndex = segment.start;
            double bestDist = (poses[splitIndex].head<2>() - intersectPt).norm();

            for (int p1 = segment.start; p1 <= segment.end; p1++) {
              double dist = (poses[p1].head<2>() - intersectPt).norm();
              if (dist < bestDist) {
                bestDist = dist;
                splitIndex = p1;
              }
            }
            splitIndices.push_back(splitIndex);
          }
        }
      }

      splitIndices.push_back(segment.end);

      // Want pieces of the segment to be in direction of travel.
      std::sort(splitIndices.begin(), splitIndices.end());
      for (size_t ii = 0; ii < splitIndices.size() - 1; ii++) {
        output_segments->push_back(
            TrajectorySegment(splitIndices[ii], splitIndices[ii + 1], LaneType::Straight));
      }
      thisPolyIdx++;
    } else {
      output_segments->push_back(segment);
    }
  }

  return;
}

void LaneDetector::SplitMaxLength(const VecVector3d& poses,
                                  const std::vector<TrajectorySegment>& input_segments,
                                  const double max_segment_length,
                                  std::vector<TrajectorySegment>* output_segments)
{
  output_segments->clear();

  for (const TrajectorySegment& segment : input_segments) {
    double dist = 0;
    for (int ii = segment.start; ii < segment.end; ii++) {
      dist += (poses[ii].head<2>() - poses[ii + 1].head<2>()).norm();
    }
    // Don't allow more divisions than num points - 1.
    int divisions = std::max(
        1, std::min(segment.end - segment.start, static_cast<int>(dist / max_segment_length)));

    // Guaranteed to be at least 1.
    int pointsPerDivision = (segment.end - segment.start) / divisions;

    int start = segment.start;
    while (start < segment.end) {
      output_segments->push_back(
          TrajectorySegment(start, std::min(start + pointsPerDivision, segment.end), segment.type));
      start += pointsPerDivision;
    }
  }

  return;
}

void LaneDetector::ExtractObservationsMultiple(
    const std::vector<TrajectorySegment>& segments, const LocalizationMapPtr& map,
    const cv::Mat1b& map_image, const std::vector<FrameUpdate>& trajectory,
    const Vector2i& min_cell_coord, const double min_search_distance_left,
    const double max_search_distance_left, const double min_search_distance_right,
    const double max_search_distance_right, std::vector<LaneModel>* lanes)
{
  lanes->clear();
  for (const TrajectorySegment& segment : segments) {
    LaneModel lane(segment.type);
    ExtractObservations(segment, map, map_image, trajectory, min_cell_coord, min_search_distance_left,
      max_search_distance_left, min_search_distance_right, max_search_distance_right, &lane);
    lanes->push_back(lane);
  }

  return;
}

void LaneDetector::ExtractObservations(
    const TrajectorySegment& segment, const LocalizationMapPtr& map,
    const cv::Mat1b& map_image, const std::vector<FrameUpdate>& trajectory,
    const Vector2i& min_cell_coord, const double min_search_distance_left,
    const double max_search_distance_left, const double min_search_distance_right,
    const double max_search_distance_right, LaneModel* lane)
{
  const float cell_size_meters = static_cast<float>(map->GetMapOptions().cell_size);

  LaneModel currentLane(segment.type);

  Vector2d prevPosition;
  for (int ii = segment.start; ii <= segment.end; ii++) {
    const RigidTransform global_from_body = trajectory[ii].transform;

    const Vector3d xyz_left_body_min(0, -1 * max_search_distance_left, 0);
    const Vector3d xyz_left_body_max(0, -1 * min_search_distance_left, 0);
    const Vector3d xyz_right_body_min(0, min_search_distance_right, 0);
    const Vector3d xyz_right_body_max(0, max_search_distance_right, 0);

    // Skip poses if too close together.
    Vector2d currPosition = Body2GlobalCoords(Vector3d(0, 0, 0), global_from_body);
    float dist = (currPosition - prevPosition).norm();
    if (dist < 5 * cell_size_meters) {
      continue;
    }
    prevPosition = currPosition;

<<<<<<< Updated upstream
      // Left search endpoints.
      cv::Point2i left_coords_img_min =
          Body2ImageCoords(xyz_left_body_min, global_from_body, map, min_cell_coord);
      cv::Point2i left_coords_img_max =
          Body2ImageCoords(xyz_left_body_max, global_from_body, map, min_cell_coord);

      // Right search endpoints.
      cv::Point2i right_coords_img_min =
          Body2ImageCoords(xyz_right_body_min, global_from_body, map, min_cell_coord);
      cv::Point2i right_coords_img_max =
          Body2ImageCoords(xyz_right_body_max, global_from_body, map, min_cell_coord);

      // Get (x, y, theta) pose of vehicle.
      Vector2d position_body_from_global = Body2GlobalCoords(Vector3d(0, 0, 0), global_from_body);

      float yaw_body_from_global = math::YawFromQuaterniond(global_from_body.Inverse().Rotation());
      Vector3d pose_body_from_global(position_body_from_global.x(), position_body_from_global.y(),
                                     (double)yaw_body_from_global);

      const size_t idx = currentLane.AddPose(pose_body_from_global);

      // Get the first POI to the left and right.
      cv::LineIterator it_left(map_image, left_coords_img_min, left_coords_img_max, 8);
      cv::LineIterator it_right(map_image, right_coords_img_min, right_coords_img_max, 8);

      for (int ii = 0; ii < it_left.count; ii++, it_left++) {
        if (map_image.at<uchar>(it_left.pos()) > 0) {
          currentLane.AddLeftPoint(idx, Image2GlobalCoords(it_left.pos(), map, min_cell_coord));
        }
      }
      for (int ii = 0; ii < it_right.count; ii++, it_right++) {
        if (map_image.at<uchar>(it_right.pos()) > 0) {
          currentLane.AddRightPoint(idx, Image2GlobalCoords(it_right.pos(), map, min_cell_coord));
        }
=======
    // Left search endpoints.
    cv::Point2i left_coords_img_min =
        Body2ImageCoords(xyz_left_body_min, global_from_body, map, min_cell_coord);
    cv::Point2i left_coords_img_max =
        Body2ImageCoords(xyz_left_body_max, global_from_body, map, min_cell_coord);

    // Right search endpoints.
    cv::Point2i right_coords_img_min =
        Body2ImageCoords(xyz_right_body_min, global_from_body, map, min_cell_coord);
    cv::Point2i right_coords_img_max =
        Body2ImageCoords(xyz_right_body_max, global_from_body, map, min_cell_coord);

    // Get (x, y, theta) pose of vehicle.
    Vector2d position_body_from_global = Body2GlobalCoords(Vector3d(0, 0, 0), global_from_body);

    float yaw_body_from_global = math::YawFromQuaterniond(global_from_body.Inverse().Rotation());
    Vector3d pose_body_from_global(position_body_from_global.x(), position_body_from_global.y(),
                                   (double)yaw_body_from_global);

    const uint32_t idx = currentLane.AddPose(pose_body_from_global);

    // Get left POI.
    std::vector<cv::Point2i> poiLeft;
    GetBresenhamPoints(left_coords_img_min, left_coords_img_max, &poiLeft, true);
    // Get right POI.
    std::vector<cv::Point2i> poiRight;
    GetBresenhamPoints(right_coords_img_min, right_coords_img_max, &poiRight, true);

    // Get left and right observation(s).
    for (size_t ii = poiLeft.size(); ii > 0; ii--) {
      if (map_image.at<uchar>(poiLeft[ii - 1]) > 0) {
        currentLane.AddLeftPoint(idx, Image2GlobalCoords(poiLeft[ii - 1], map, min_cell_coord));
        break;
      }
    }
    for (size_t ii = 0; ii < poiRight.size(); ii++) {
      if (map_image.at<uchar>(poiRight[ii]) > 0) {
        currentLane.AddRightPoint(idx, Image2GlobalCoords(poiRight[ii], map, min_cell_coord));
        break;
>>>>>>> Stashed changes
      }
    }
    *lane = currentLane;
  }

  return;
}

void LaneDetector::ConvertTrajectoryGlobal(const std::vector<FrameUpdate>& trajectory,
                                           std::vector<Vector3d>* poses)
{
  poses->clear();

  for (size_t ii = 0; ii <= trajectory.size(); ii++) {
    const RigidTransform global_from_body = trajectory[ii].transform;

    // Get (x, y, theta) pose of vehicle.
    Vector2d position_body_from_global = Body2GlobalCoords(Vector3d(0, 0, 0), global_from_body);

    double yaw_body_from_global = math::YawFromQuaterniond(global_from_body.Inverse().Rotation());
    Vector3d pose_body_from_global(position_body_from_global.x(), position_body_from_global.y(),
                                   yaw_body_from_global);

    poses->push_back(pose_body_from_global);
  }

  return;
}

void LaneDetector::FitLane(const double default_offset_left, const double default_offset_right,
                           const LaneFittingParams& params, LaneModel* lane)
{
  // Straight: fit lane lines with RANSAC.
  if (lane->GetType() == LaneType::Straight) {
    std::pair<VecVector3d, VecVector2d> left_flattened =
        lane->GetLeftPointsFlattened(default_offset_left);
    std::pair<VecVector3d, VecVector2d> right_flattened =
        lane->GetRightPointsFlattened(default_offset_right);

    // Compute arc length of left and right segments to determine # control points.
    VecVector2d poses_left_2d, poses_right_2d;
    for (const Vector3d& pose : left_flattened.first) {
      poses_left_2d.push_back(pose.head<2>());
    }
    for (const Vector3d& pose : right_flattened.first) {
      poses_right_2d.push_back(pose.head<2>());
    }

    double left_length = 0;
    double right_length = 0;
    if (poses_left_2d.size() >= 2) {
      curves2d::ASplinePtr left_pose_spline = curves2d::ASpline::Create(poses_left_2d);
      curves2d::CurveUtilWrapper left_wrap(left_pose_spline, 0.101);
      left_length = left_wrap.GetCurveArcLength();
    }
    if (poses_right_2d.size() >= 2) {
      curves2d::ASplinePtr right_pose_spline = curves2d::ASpline::Create(poses_right_2d);
      curves2d::CurveUtilWrapper right_wrap(right_pose_spline, 0.101);
      right_length = right_wrap.GetCurveArcLength();
    }
    // Calculate # control points.
    const size_t N_left = std::max(params.min_control_points_straight,
                                   (int)(left_length / params.control_point_spacing_straight));
    const size_t N_right = std::max(params.min_control_points_straight,
                                    (int)(right_length / params.control_point_spacing_straight));

    if (left_flattened.second.size() >= N_left) {
      bool lsuccess = false;
      LaneFitResult lfit = FitSplineRansac(
          left_flattened.second, left_flattened.first, &lsuccess, N_left, params.ransac_inlier_dist,
          params.ransac_inlier_weight, params.ransac_tangent_weight, params.ransac_width_weight,
          params.ransac_max_iters, params.ransac_success_score);

      // Use inliers to estimate half width.
      lfit.half_width =
          EstimateWidth(lfit.inliers, SubsetByIndices(left_flattened.first, lfit.inlier_indices));
      lane->SetLeftFitResult(lfit);
    } else {
      OINFO("Not enough points to fit left spline.");
    }

    if (right_flattened.second.size() >= N_right) {
      bool rsuccess = false;
      LaneFitResult rfit = FitSplineRansac(
          right_flattened.second, right_flattened.first, &rsuccess, N_right,
          params.ransac_inlier_dist, params.ransac_inlier_weight, params.ransac_tangent_weight,
          params.ransac_width_weight, params.ransac_max_iters, params.ransac_success_score);

      // Use inliers to estimate half width.
      rfit.half_width =
          EstimateWidth(rfit.inliers, SubsetByIndices(right_flattened.first, rfit.inlier_indices));
      lane->SetRightFitResult(rfit);
    } else {
      OINFO("Not enough points to fit right spline");
    }
    // Curve
  } else {
  }

  return;
}

void LaneDetector::DetectLanes(const LocalizationMapPtr& map,
                               const std::vector<FrameUpdate>& trajectory,
                               const LaneDetectorParams& detector_params,
                               const LaneFittingParams& ransac_params,
                               std::vector<LaneModel>* lanes)
{
  lanes->clear();
  cv::namedWindow("debug", cv::WINDOW_NORMAL);

  // Convert trajectory into poses in global frame.
  std::vector<Vector3d> poses;
  ConvertTrajectoryGlobal(trajectory, &poses);

  // Subdivide the trajectory based on curvature.
  std::vector<TrajectorySegment> segments;
  SubdivideTrajectory(trajectory, detector_params.local_spline_length,
                      detector_params.pts_per_spline, detector_params.straight_max_curvature,
                      &segments);

  std::vector<TrajectorySegment> segments_split;
  SplitAtIntersections(poses, segments, 0.0, &segments_split);

  OINFO("Split at intersections");

  SplitMaxLength(poses, segments_split, detector_params.max_segment_length, &segments);

  OINFO("Split into {} segments.", segments.size());

  // Convert localization map to cv::Mat and get the min cell coordinate of the image
  // (top left corner). This is needed for converting localization map coordinates
  // into the image frame.
  cv::Mat1b map_image;
  Vector2i min_cell_coord;
  ConvertTiledMapToImage(map->GetMap(), &map_image, &min_cell_coord);

  // Preprocess the image - not used currently.
  cv::Mat1b processed;
  PreprocessMapImage(map_image, &processed);

  cv::Mat3b debug;
  cv::cvtColor(map_image, debug, cv::COLOR_GRAY2BGR);
  DrawSegments(debug, segments, trajectory, map, min_cell_coord);

  // Stores each lane that we detect.
  ExtractObservationsMultiple(
      segments, map, map_image, trajectory, min_cell_coord,
      detector_params.min_search_distance_left, detector_params.max_search_distance_left,
      detector_params.min_search_distance_right, detector_params.max_search_distance_right, lanes);

  // For each lane, fit to observations only, then refit with default width.
  for (LaneModel& lane : *lanes) {
    // DrawLane(debug, lane, map, min_cell_coord);
    FitLane(0.0, 0.0, ransac_params, &lane);
    FitLane(lane.GetLeftFit().half_width, lane.GetRightFit().half_width, ransac_params, &lane);

    if (lane.GetLeftFit().spline != nullptr) {
      DrawLaneFit(debug, lane.GetLeftFit(), map, min_cell_coord);
    }
    if (lane.GetRightFit().spline != nullptr) {
      DrawLaneFit(debug, lane.GetRightFit(), map, min_cell_coord);
    }

    // Compute final fit score.
    double leftScore = lane.GetLeftFit().score * lane.PercentLeftObservations();
    double rightScore = lane.GetRightFit().score * lane.PercentRightObservations();
    OINFO("Scores: l={} r={}", leftScore, rightScore);
  }

  cv::imshow("debug", debug);
  cv::waitKey(0);

  return;
}

<<<<<<< Updated upstream
=======
void onMouse(int evt, int x, int y, int /*flags*/, void* param)
{
  if (evt == CV_EVENT_LBUTTONDOWN) {
    // OINFO("onMouse: {} {}", x, y);
    cv::Point* ptPtr = (cv::Point*)param;
    ptPtr->x = x;
    ptPtr->y = y;
  }
}

void LaneDetector::DetectLanesInteractive(const LocalizationMapPtr& map,
                                          const std::vector<FrameUpdate>& trajectory,
                                          const DetectorParams& detector_params,
                                          const RansacParams& ransac_params,
                                          std::vector<LaneModel>* lanes)
{
  lanes->clear();
  cv::namedWindow("debug", cv::WINDOW_NORMAL);

  cv::Point clickedPoint;
  cv::setMouseCallback("debug", onMouse, (void*)&clickedPoint);

  // Convert trajectory into poses in global frame.
  std::vector<Vector3d> poses;
  ConvertTrajectoryGlobal(trajectory, &poses);

  // Subdivide the trajectory based on curvature.
  std::vector<TrajectorySegment> segments;
  SubdivideTrajectory(trajectory, detector_params.local_spline_length,
                      detector_params.pts_per_spline, detector_params.straight_max_curvature,
                      &segments);
  std::vector<TrajectorySegment> segments_split;
  SplitAtIntersections(poses, segments, 0.0, &segments_split);
  SplitMaxLength(poses, segments_split, detector_params.max_segment_length, &segments);

  OINFO("Split into {} segments.", segments.size());

  // Convert localization map to cv::Mat and get the min cell coordinate of the image
  // (top left corner). This is needed for converting localization map coordinates
  // into the image frame.
  cv::Mat1b map_image;
  Vector2i min_cell_coord;
  ConvertTiledMapToImage(map->GetMap(), &map_image, &min_cell_coord);

  // Preprocess the image - not used currently.
  cv::Mat1b processed;
  PreprocessMapImage(map_image, &processed);

  cv::Mat3b debug;
  cv::cvtColor(map_image, debug, cv::COLOR_GRAY2BGR);
  DrawSegments(debug, segments, trajectory, map, min_cell_coord);

  // Stores each lane that we detect.
  ExtractObservationsMultiple(
      segments, map, map_image, trajectory, min_cell_coord,
      detector_params.min_search_distance_left, detector_params.max_search_distance_left,
      detector_params.min_search_distance_right, detector_params.max_search_distance_right, lanes);

  // For each lane, fit to observations only, then refit with default width.
  size_t laneId = 0;
  cv::Mat3b debug_initial_fit;
  cv::Mat3b debug_refined_fit;

  while (laneId < lanes->size()) {
    LaneModel& lane = (*lanes)[laneId];

    // Make the initial lane fit suggestion.
    debug_initial_fit = debug.clone();

    FitLane(lane, 0.0, 0.0, ransac_params);
    FitLane(lane, lane.GetLeftFit().half_width, lane.GetRightFit().half_width, ransac_params);

    if (lane.GetLeftFit().spline != nullptr) {
      DrawLaneFit(debug_initial_fit, lane.GetLeftFit(), map, min_cell_coord);
    }
    if (lane.GetRightFit().spline != nullptr) {
      DrawLaneFit(debug_initial_fit, lane.GetRightFit(), map, min_cell_coord);
    }
    OINFO("Press ENTER to accept, SPACE to input point.");
    cv::imshow("debug", debug_initial_fit);

    char key = cv::waitKey(0);

    // Make the refined fit.
    debug_refined_fit = debug_initial_fit.clone();

    while (key != 13) { // ENTER=13, BACKSPACE=8.
      debug_refined_fit = debug.clone();
      OINFO("LEFTCLICK on point. Then press any key to continue.");
      cv::waitKey(0);
      cv::circle(debug_refined_fit, clickedPoint, 5, cv::viz::Color::blue());

      // TODO: get half width from clicked point
      Vector2d clickedPointGlobal = Image2GlobalCoords(clickedPoint, map, min_cell_coord);
      Vector3d closestPose = lane.GetPoseAtIndex(0);
      double bestDist = (closestPose.head<2>() - clickedPointGlobal).norm();
      int ctr = 0;
      for (const Vector3d& pose : lane.GetPoses()) {
        double dist = (pose.head<2>() - clickedPointGlobal).norm();
        if (dist < bestDist) {
          bestDist = dist;
          closestPose = pose;
        }
        ctr++;
      }
      OINFO("Clicked width: {}", bestDist);

      // Determine left/right:
      // https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
      Vector2d pointRay = clickedPointGlobal - closestPose.head<2>();
      Vector2d poseRay(std::cos(-1 * closestPose(2)), std::sin(-1 * closestPose(2)));

      // True for left, false for right.
      bool side = (pointRay(0) * poseRay(1) - pointRay(1) * poseRay(0)) >= 0;

      // Extract observations again with new distance constraints.
      double min_search_distance_left = side ? bestDist - 0.5 : detector_params.min_search_distance_left;
      double max_search_distance_left = side ? bestDist + 0.2 : detector_params.max_search_distance_left;
      double min_search_distance_right = side ? detector_params.min_search_distance_right : bestDist - 0.5;
      double max_search_distance_right = side ? detector_params.max_search_distance_right : bestDist + 0.2;

      LaneModel updateLane(lane.GetType());
      ExtractObservations(segments[laneId], map, map_image, trajectory, min_cell_coord,
          min_search_distance_left, max_search_distance_left, min_search_distance_right, max_search_distance_right,
          &updateLane);

      lane = updateLane; // TODO: add a reset method to the lane model.

      // Refit the lane with a new default distance.
      double ldefault = side ? bestDist : lane.GetLeftFit().half_width;
      double rdefault = side ? lane.GetRightFit().half_width : bestDist;
      FitLane(lane, ldefault, rdefault, ransac_params);

      if (lane.GetLeftFit().spline != nullptr) {
        DrawLaneFit(debug_refined_fit, lane.GetLeftFit(), map, min_cell_coord);
      }
      if (lane.GetRightFit().spline != nullptr) {
        DrawLaneFit(debug_refined_fit, lane.GetRightFit(), map, min_cell_coord);
      }
      OINFO("Press ENTER to accept, SPACE to input point.");
      cv::imshow("debug", debug_refined_fit);
      key = cv::waitKey(0);
    }

    // Compute final fit score.
    double leftScore = lane.GetLeftFit().score * lane.PercentLeftObservations();
    double rightScore = lane.GetRightFit().score * lane.PercentRightObservations();
    OINFO("Scores: l={} r={}", leftScore, rightScore);

    // Accept the refined fit.
    debug = debug_refined_fit.clone();

    laneId++;
  }
  return;
}

>>>>>>> Stashed changes
void LaneDetector::ConvertTiledMapToImage(const perception::TiledUniformGridMapUint8* map,
                                          cv::Mat1b* image, Vector2i* min_cell_coord)
{
  const Vector2i& tiled_map_min_corner = map->GetMinCornerCellUnits();
  const Vector2i& cells_per_tile = map->GetNumCellsPerTile();
  assert(cells_per_tile.x() == cells_per_tile.y());

  // BBox for initialized tiles.
  const AlignedBox2i& tiles_bbox = map->GetTileInitializedBBox();
  const Vector2i& tiles_bbox_mincell = tiles_bbox.min() * cells_per_tile.x();
  const Vector2i& cells_dim = (tiles_bbox.max() - tiles_bbox.min()) * cells_per_tile.x();

  *min_cell_coord = tiles_bbox_mincell;
  *image = cv::Mat::zeros(cells_dim.y(), cells_dim.x(), CV_8UC1);

  for (int tx = tiles_bbox.min().x(); tx < tiles_bbox.max().x(); ++tx) {
    for (int ty = tiles_bbox.min().y(); ty < tiles_bbox.max().y(); ++ty) {
      if (!map->TileInitializedByTileIndex(tx, ty)) {
        continue;
      }
      const perception::UniformGridMap<uint8_t>& tile = map->GetTileByTileIndex(tx, ty);
      const Vector2i& tile_min_corner = tile.GetMinCornerCellunits();

      for (int cx = 0; cx < cells_per_tile.x(); ++cx) {
        for (int cy = 0; cy < cells_per_tile.y(); ++cy) {
          int x_offset = tile_min_corner.x() - tiled_map_min_corner.x() - tiles_bbox_mincell.x();
          int y_offset = tile_min_corner.y() - tiled_map_min_corner.y() - tiles_bbox_mincell.y();
          image->at<uchar>(cy + y_offset, cx + x_offset) =
              static_cast<uint8_t>(tile.GetCell(cx, cy));
        }
      }
    }
  }
  return;
}

}  // namespace mapping
}  // namespace ori
