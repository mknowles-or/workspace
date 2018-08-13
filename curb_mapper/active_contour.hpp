// Copyright [2018] Optimus Ride Inc.

#include <vector>

#include <opencv2/core.hpp>

#include "mc/common/omath/eigen_types.hpp"

namespace ori {
namespace mapping {

struct ActiveContourOptions
{
  float alpha = 1.0;     // Length shape parameter. Higher values cause more elastic behavior.
  float beta = 1.5;       // Smoothness shape parameter. Higher values make the snake smoother.

  float w_intensity = 100.0;  // Attraction to bright intensity (negative to dark).
  float w_edge = 1000.0;       // Attraction to regions of high gradient.
  float w_corner = 0.0;     // Attraction to corners and other high curvature regions.

  float w_external = 1.0;
  float w_normal = 0.0;

  float gamma = 0.1;      // Time stepping parameter.
  bool periodic = false;  // If periodic, the contour is closed. If not, ends are fixed in place.
  float max_px_move = 0.5;  // Maximum pixel movement per iteration.
  int max_iterations = 2500;   // Maximum iterations to refine shape.
  float convergence = 0.01;     // Convergence criteria.
  int convergence_order = 10;  // Sliding window size for checking convergence deltas.
  int sobel_kernel_size = 3;  // Kernel size for computing derivatives with Sobel operator.
  int gaussian_kernel_size = 3; // Kernel size for gaussian blurring.

  int normal_direction = 1;
};

/**
 * @brief "Rolls" a matrix by a number of rows or columns.
 * Based on https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.roll.html.
 */
MatrixXd RollMatrix(const MatrixXd& mat, const int shift, const int axis = 0);

/**
 * @brief Get the value of a floating point image at a point with subpixel precision.
 */
float SubpixelValue(const cv::Mat1f& img, const cv::Point2f& pt);

/**
 * @brief Optimized a contour from a set of initial control points.
 * Implementation:
 * https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/active_contour_model.py
 * Equations: https://www.inf.u-szeged.hu/~kato/teaching/emm/02-Snake.pdf
 */
VecVector2d ActiveContour(const cv::Mat1f image, const VecVector2d& initial_points,
                          const ActiveContourOptions& options);
}
}
