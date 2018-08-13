// Copyright [2018] Optimus Ride Inc.

#include <iostream>
#include <cmath>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "workstation/mapping/curb_mapper/active_contour.hpp"
#include "workstation/mapping/curb_mapper/curb_drawing_utils.hpp"
#include "mc/common/ocommon/olog.hpp"

namespace ori {
namespace mapping {

#define OLOG_GROUP "active_contour"

static constexpr int kSobelScale = 1;
static constexpr int kSobelDelta = 0;

/**
 * @brief "Rolls" a matrix by a number of rows or columns.
 * Based on https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.roll.html.
 */
MatrixXd RollMatrix(const MatrixXd& mat, const int shift, const int axis)
{
  // Only support row or column axis.
  assert(axis == 0 || axis == 1);

  // No shift means that the matrix is unchanged.
  if (shift == 0) {
    return mat;
  }

  // Make the output matrix.
  MatrixXd rolled(mat.rows(), mat.cols());

  // Roll by shift rows.
  if (axis == 0) {
    const int zero_row_loc =
        (shift >= 0) ? (shift % mat.rows()) : (mat.rows() - (std::abs(shift) % mat.rows()));
    rolled.bottomRows(mat.rows() - zero_row_loc) = mat.topRows(mat.rows() - zero_row_loc);
    rolled.topRows(zero_row_loc) = mat.bottomRows(zero_row_loc);
  } else {
    // Roll by shift cols.
    const int zero_col_loc =
        (shift >= 0) ? (shift % mat.cols()) : (mat.cols() - (std::abs(shift) % mat.cols()));
    rolled.rightCols(mat.cols() - zero_col_loc) = mat.leftCols(mat.cols() - zero_col_loc);
    rolled.leftCols(zero_col_loc) = mat.rightCols(zero_col_loc);
  }

  return rolled;
}

/**
 * @brief Get the value of a floating point image at a point with subpixel precision.
 */
float SubpixelValue(const cv::Mat1f& img, const cv::Point2f& pt)
{
  // cv::Mat patch;
  // cv::getRectSubPix(img, cv::Size(1, 1), pt, patch);
  // return patch.at<float>(0, 0);
  const int x = static_cast<int>(pt.x);
  const int y = static_cast<int>(pt.y);
  const int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
  const int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
  const int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
  const int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

  const float dx = pt.x - static_cast<float>(x);
  const float dy = pt.y - static_cast<float>(y);
  const float interp = (img.at<float>(y0, x0) * (1.0 - dx) +
                       img.at<float>(y0, x1) * dx) * (1.0 - dy) +
                       (img.at<float>(y1, x0) * (1.0 - dx) + img.at<float>(y1, x1) * dx) * dy;
  return interp;
}

void ComptuteGradients(const cv::Mat1f image, cv::Mat1f gx, cv::Mat1f gy, cv::Mat1f mag)
{
  cv::Sobel(image, gx, CV_32F, 1, 0, 3, 0.25, kSobelDelta,
          cv::BORDER_DEFAULT);
  cv::Sobel(image, gy, CV_32F, 0, 1, 3, 0.25, kSobelDelta,
          cv::BORDER_DEFAULT);
  cv::magnitude(gx, gy, mag);
  cv::divide(mag, M_SQRT2, mag);
}

/**
 * @brief Compute the magnitude of curvature each pixel in the image as a measure of corner-ness.
 * https://lmb.informatik.uni-freiburg.de/Publications/2014/FB14/gcpr2014_curvature.pdf
 */
cv::Mat1f ComputeCurvature(const cv::Mat1f gradient_x, const cv::Mat1f gradient_y, const cv::Mat1f gradient_mag,
  const ActiveContourOptions& options)
{
  assert(!gradient_x.empty() && !gradient_y.empty());

  cv::Mat1f gradient_xx, gradient_yy, gradient_xy;

  // Normalize gradients first, such that [gx, gy] is a unit vector.
  // cv::GaussianBlur(gradient_x, gradient_x, cv::Size(3, 3), 0, 0);
  // cv::GaussianBlur(gradient_y, gradient_y, cv::Size(3, 3), 0, 0);
  cv::divide(gradient_x, gradient_mag, gradient_x);
  cv::divide(gradient_y, gradient_mag, gradient_y);

  cv::Sobel(gradient_x, gradient_xx, CV_32F, 1, 0, options.sobel_kernel_size, kSobelScale,
            kSobelDelta, cv::BORDER_DEFAULT);
  cv::Sobel(gradient_y, gradient_yy, CV_32F, 0, 1, options.sobel_kernel_size, kSobelScale,
            kSobelDelta, cv::BORDER_DEFAULT);
  cv::Sobel(gradient_y, gradient_xy, CV_32F, 1, 0, options.sobel_kernel_size, kSobelScale,
            kSobelDelta, cv::BORDER_DEFAULT);

  cv::Mat1f g_x2, g_y2, g_xy, den;
  cv::multiply(gradient_x, gradient_x, g_x2);
  cv::multiply(gradient_y, gradient_y, g_y2);
  cv::multiply(gradient_x, gradient_y, g_xy);
  cv::multiply(g_xy, cv::Scalar(-2.0), g_xy);

  cv::Mat1f num = cv::Mat::zeros(gradient_x.size(), CV_32FC1);
  cv::accumulateProduct(gradient_yy, g_x2, num);
  cv::accumulateProduct(gradient_xx, g_y2, num);
  cv::accumulateProduct(gradient_xy, g_xy, num);

  cv::add(g_x2, g_y2, den);
  cv::pow(den, 1.5, den);

  cv::Mat1f corners;
  cv::divide(num, den, corners);  // Handles divide by zero by replacing with zero in result.
  return corners;
}

cv::Mat1f PreprocessImage(const cv::Mat1f input)
{
  cv::Mat1f output;
  cv::GaussianBlur(input, output, cv::Size(3, 3), 0, 0);
  cv::normalize(input, output, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);
  return output;
}

/**
 * @brief Optimized a contour from a set of initial control points.
 * Implementation:
 * https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/active_contour_model.py
 * Equations: https://www.inf.u-szeged.hu/~kato/teaching/emm/02-Snake.pdf
 */
VecVector2d ActiveContour(const cv::Mat1f input_image, const VecVector2d& initial_points,
                          const ActiveContourOptions& options)
{
  assert(options.max_iterations > 0);          // Nonzero iterations required.
  assert(options.convergence_order > 0);       // Nonzero window size required.
  assert(options.sobel_kernel_size % 2 != 0);  // Sobel kernel size must be odd.

  // Normalize the image.
  cv::Mat1f image = PreprocessImage(input_image);

  // Make the control point x and y vectors.
  const int n = initial_points.size();
  VectorXd px(n);
  VectorXd py(n);
  for (int pi = 0; pi < n; ++pi) {
    px(pi) = initial_points.at(pi).x();
    py(pi) = initial_points.at(pi).y();
  }

  // Allocate matrices. Only some will actually be computed, based on params.
  cv::Mat1f gradient_x = cv::Mat::zeros(image.size(), CV_32FC1);
  cv::Mat1f gradient_y = cv::Mat::zeros(image.size(), CV_32FC1);
  cv::Mat1f gradient = cv::Mat::zeros(image.size(), CV_32FC1);
  cv::Mat1f corners = cv::Mat::zeros(image.size(), CV_32FC1);

  // Both edges and corners require first derivatives to be computed.
  if (options.w_edge != 0 || options.w_corner != 0) {
    ComptuteGradients(image, gradient_x, gradient_y, gradient);
  }

  if (options.w_corner != 0) {
    corners = ComputeCurvature(gradient_x, gradient_y, gradient, options);
  }

  // Make an image using weighted sum of components.
  // Normalize all of the weights such that they add up to 1.0.
  // E = w_intensity * I + w_edge * Gradient(I) + w_corner * Corner(I)
  cv::Mat1f image_energy = cv::Mat::zeros(image.size(), CV_32FC1);
  cv::scaleAdd(image, options.w_intensity, image_energy, image_energy);
  cv::scaleAdd(gradient, options.w_edge, image_energy, image_energy);
  cv::scaleAdd(corners, options.w_corner, image_energy, image_energy);
  // cv::GaussianBlur(image_energy, image_energy, cv::Size(9, 9), 0, 0);

  cv::namedWindow("energy", cv::WINDOW_NORMAL);

  // Compute derivatives of the image energy.
  cv::Mat1f image_energy_gradient_x, image_energy_gradient_y;
  cv::Sobel(image_energy, image_energy_gradient_x, CV_32F, 1, 0, options.sobel_kernel_size,
            kSobelScale, kSobelDelta, cv::BORDER_DEFAULT);
  cv::Sobel(image_energy, image_energy_gradient_y, CV_32F, 0, 1, options.sobel_kernel_size,
            kSobelScale, kSobelDelta, cv::BORDER_DEFAULT);

  // Build snake shape matrix for euler equation.
  const MatrixXd& a = RollMatrix(MatrixXd::Identity(n, n), -1, 0)
                      + RollMatrix(MatrixXd::Identity(n, n), -1, 1) - 2 * MatrixXd::Identity(n, n);
  const MatrixXd& b =
      RollMatrix(MatrixXd::Identity(n, n), -2, 0) + RollMatrix(MatrixXd::Identity(n, n), -2, 1)
      - 4 * RollMatrix(MatrixXd::Identity(n, n), -1, 0)
      - 4 * RollMatrix(MatrixXd::Identity(n, n), -1, 1) + 6 * MatrixXd::Identity(n, n);

  // The shape matrix.
  MatrixXd A = -options.alpha * a + options.beta * b;

  // Impose boundary conditions if not periodic (fixed endpoints).
  if (!options.periodic) {
    Eigen::VectorXd replace_boundary(3);
    replace_boundary << 1, -2, 1;
    A.topRows(2) *= 0;
    A.bottomRows(2) *= 0;
    A.row(1).head(3) = replace_boundary;
    A.row(A.rows() - 2).tail(3) = replace_boundary;
  }

  const MatrixXd& euler = (A + options.gamma * MatrixXd::Identity(n, n)).inverse();
  // std::cout << euler.topLeftCorner(6, 6);
  // cv::waitKey(0);

  // Save a sliding window of previous x and y values to check for convergence.
  MatrixXd xsave(options.convergence_order, n);
  MatrixXd ysave(options.convergence_order, n);

  // Minimize energy with explicit time stepping.
  for (int i = 0; i < options.max_iterations; ++i) {
    VecVector2d points;
    for (int j = 0; j < n; ++j) {
      points.emplace_back(Vector2d(px(j), py(j)));
    }
    cv::Mat3b state_img = DrawPolyline(image_energy, points);
    cv::imshow("energy", state_img);
    cv::waitKey(0);

    // Add a "normal force" to push the contour against boundaries.
    VectorXd dxs = RollMatrix(px, 1, 0) - RollMatrix(px, -1, 0);
    VectorXd dys = RollMatrix(py, 1, 0) - RollMatrix(py, -1, 0);

    // Calculate external image forces.
    VectorXd forces_x(n);
    VectorXd forces_y(n);

    for (int pi = 0; pi < n; ++pi) {
      const cv::Point2f xy_subpixel(px(pi), py(pi));
      const Vector2d f_normal = options.normal_direction * Vector2d(dys(pi), -dxs(pi)).normalized();
      forces_x(pi) = options.w_external * SubpixelValue(image_energy_gradient_x, xy_subpixel) + options.w_normal * f_normal.x();
      forces_y(pi) = options.w_external * SubpixelValue(image_energy_gradient_y, xy_subpixel) + options.w_normal * f_normal.y();
    }

    // No external image forces on endpoints if they are fixed.
    if (!options.periodic) {
      forces_x(0) = 0;
      forces_x(n - 1) = 0;
      forces_y(0) = 0;
      forces_y(n - 1) = 0;
    }

    // Compute the updated x and y coordinates.
    const VectorXd& px_updated = euler * (options.gamma * px + forces_x);
    const VectorXd& py_updated = euler * (options.gamma * py + forces_y);

    // Limit update movements to max_px_move at each iteration.
    const VectorXd& dx = options.max_px_move * (px_updated - px).array().tanh();
    const VectorXd& dy = options.max_px_move * (py_updated - py).array().tanh();

    // Update the x and y coordinates.
    px += dx;
    py += dy;

    // Check for convergence criteria.
    const int j = i % (options.convergence_order + 1);

    if (j < options.convergence_order) {
      xsave.row(j) = px;
      ysave.row(j) = py;
    } else {
      // Change in x and y at each timestep in window.
      const MatrixXd& delta = (xsave.rowwise() - px.transpose()).array().abs()
                              + (ysave.rowwise() - py.transpose()).array().abs();

      // Max change among all points at each timestep.
      const MatrixXd max_delta_all_points = delta.rowwise().maxCoeff();

      // Smallest shift at any of timesteps.
      float min_delta = max_delta_all_points.minCoeff();

      if (min_delta < options.convergence) {
        break;
      }
    }
  }

  // Return the optimized points.
  VecVector2d output_points;
  for (int pi = 0; pi < n; ++pi) {
    output_points.emplace_back(Vector2d(px(pi), py(pi)));
  }
  return output_points;
}
}
}
