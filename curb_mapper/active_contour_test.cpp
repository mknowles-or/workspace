// Copyright [2018] Optimus Ride Inc.

#include <gtest/gtest.h>

#include "workstation/mapping/curb_mapper/active_contour.hpp"

namespace ori {
namespace mapping {

TEST(ActiveContour, SubpixelValue)
{
  cv::Mat1f img = cv::Mat::zeros(10, 10, CV_32FC1);
  img.at<float>(cv::Point(3, 4)) = 1.0;

  float value1 = SubpixelValue(img, cv::Point2f(3, 4));
  float value2 = SubpixelValue(img, cv::Point2f(9, 9));
  float value3 = SubpixelValue(img, cv::Point2f(3.5, 4));

  // On top of the white pixel.
  EXPECT_EQ(1.0, value1);

  // On top of black pixel, no neighbors.
  EXPECT_EQ(0.0, value2);

  // Within one pixel of a 1.0 pixel.
  EXPECT_EQ(0.5, value3);
}

}
}
