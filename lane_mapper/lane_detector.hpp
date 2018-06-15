#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Eigenvalues>

#include "mc/common/ocommon/olog.hpp"

#include "mc/common/oimage/color_conversion.hpp"
#include "mc/common/oimage/data_buffer_allocator.hpp"
#include "mc/common/oimage/oframe.hpp"
#include "mc/common/oimage/oimage.hpp"
#include "mc/common/oimage/opencv_adapters.hpp"
#include "mc/common/oimage/stereo_rectify.hpp"

#include "mc/common/curves2d/cubic_hermite_spline.hpp"
#include "mc/common/omath/plane.hpp"

#include "mc/common/osettings/settings.hpp"
#include "mc/common/vehicle_settings/vehicle_settings_loader.hpp"

#include "mc/perception/cuda_utils/cuda_buffer_allocator.hpp"
#include "mc/perception/cuda_utils/cuda_utils.hpp"

namespace ori {
namespace mapping {

class LaneDetector {
 public:
  struct Params
  {
    Params() = default;

    bool verbose = true;     // Print debug statements.
    bool show_debug = true;  // Show debug images.
    float lane_width = 0.1;  // meters.
  };

  explicit LaneDetector(const Params& params);

  /**
   * @brief Extract lane features in the form of Cubic Hermite splines.
   * @param[in] input_oimages A front facing stereo image pair (rectified and undistorted).
   * @param[in] ground_plane The current ground plane estimate in the camera frame.
   * @param[in] stereo_calib A vector of calibration for the front left and right cameras.
   * @param[in] vehicle_settings Settings used for static transforms (i.e camera to body).
   * @param[out] lane_features A vector of Cubic Hermite Splines in the camera frame.
   */
  void ExtractLaneFeatures(const oimage::OImagePtrVector& input_oimages,
                           const math::Plane& ground_plane,
                           const oimage::CameraCalibrationPtrVectorPtr& stereo_calib,
                           std::vector<curves2d::CubicHermiteSpline>* lane_features);

  // void ProcessImages(const oimage::OImagePtrVector& input_oimages,
  //                    oimage::OImagePtr detection_oimage);

  void ApplyColorFilter(const cv::Mat3b& input_mat, cv::Mat1b* output_mat);

  void ApplyEdgeFilter(const cv::Mat1b& input_mat, cv::Mat1b* output_mat);

  void BirdsEyeTransform(const cv::Mat1b& input_mat, cv::Mat1b* output_mat);

  void MaskRoadRoi(const cv::Mat1b& input_mat, cv::Mat1b* output_mat);

  /**
   * @brief Applies a matched filter to detect horizontal and vertical lane features.
   * @param[in] input_mat A mono (rectified, undistorted) image from left camera.
   * @param[in] ground_plane The current ground plane estimate in the camera frame.
   * @param[in] intrinsics The intrinsics of the left camera.
   * @param[out] filtered_x Horizontal filter response.
   * @param[out] filtered_y Vertical filter response.
   * Note: the output images will be initialized inside of this method.
   */
  void ApplyMatchedFilter(const cv::Mat& input_mat, const math::Plane& ground_plane,
                          const oimage::CameraIntrinsics& intrinsics, cv::Mat* filtered_x,
                          cv::Mat* filtered_y);

  void GetInterestPoints(const cv::Mat& input_mat, std::vector<cv::Point>* ipoints,
                         const int axis = 0, const float threshold = 32);

  void GetiPointOrientations(const std::vector<cv::Point>& ipoints,
                            const cv::Mat& F_xx, const cv::Mat& F_yy,
                            const cv::Mat& F_xy, const cv::Mat& F_yx,
                            std::vector<Vector2d>* orientations);

 private:
  const Params params_;

  // OpenCV Mat Allocation.
  cv::Mat1b left_mat_hue, left_mat_light, left_mat_sat;
  cv::Mat3b left_mat_bgr;
  cv::Mat3b left_mat_hls;
  cv::Mat1b left_mat_yellow_mask;
  cv::Mat1b left_mat_white_mask;
  cv::Mat1b left_mat_mono;
  std::vector<cv::Mat1b> channels{3};
};

}  // namespace mapping
}  // namespace ori