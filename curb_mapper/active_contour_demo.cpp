#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "mc/common/ocommon/olog.hpp"

#include "workstation/mapping/curb_mapper/active_contour.hpp"
#include "workstation/mapping/curb_mapper/curb_drawing_utils.hpp"

using namespace ori;
using namespace mapping;

#define OLOG_GROUP "active_contour_test"

int main(int /*argc*/, char**/*argv*/)
{
  const std::string fname = "/home/mknowles/Documents/masked_hm2.png";
  cv::Mat1f input = cv::imread(fname.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

  VecVector2d points = {
    Vector2d(101, 358),
    Vector2d(208, 331),
    Vector2d(294, 290),
    Vector2d(443, 336),
    Vector2d(527, 450),
    Vector2d(545, 534),
    Vector2d(456, 520),
    Vector2d(456, 734),
    Vector2d(536, 738),
    Vector2d(547, 825),
    Vector2d(463, 825),
    Vector2d(468, 927)
  };

  VecVector2d tp;
  for (const Vector2d& pt : points) {
    tp.emplace_back(pt.y(), pt.x());
  }

  ActiveContourOptions snake_options;
  snake_options.alpha = 10.0;
  snake_options.beta = 1.5;
  snake_options.gamma = 0.01;
  snake_options.w_edge = 1000.0;
  snake_options.w_intensity = 100.0;
  snake_options.w_corner = 0.0;
  snake_options.w_normal = 100.0;
  snake_options.max_px_move = 1.0;
  snake_options.convergence = 0.01;
  snake_options.normal_direction = -1;

  const VecVector2d& output_points = ActiveContour(input, tp, snake_options);

  cv::Mat3b result = DrawPolyline(input, output_points);
  cv::namedWindow("result", cv::WINDOW_NORMAL);
  cv::imshow("result", result);
  cv::waitKey(0);

  return 0;
}
