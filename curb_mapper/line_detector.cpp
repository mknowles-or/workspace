  // Detect lines from gradient image.
  cv::Mat1b edges;
  cv::blur(MakeViewableImage(masked_heightmap_cv.Image()), edges, cv::Size(3, 3));
  cv::Canny(edges, edges, 20, 70, 3);
  cv::namedWindow("edges", cv::WINDOW_NORMAL);
  cv::imshow("edges", edges);

  std::vector<cv::Vec4i> lines;
  const int num_votes_required = 20;
  const int min_line_length = 10;
  const int max_line_gap = 5;
  cv::HoughLinesP(edges, lines, 2, M_PI/180, num_votes_required, min_line_length, max_line_gap);

  OINFO("got lines: {}", lines.size());

  cv::Mat3b line_image;
  cv::cvtColor(MakeViewableImage(masked_heightmap_cv.Image()), line_image, cv::COLOR_GRAY2BGR);
  for (const cv::Vec4i& l : lines) {
    cv::line(line_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 1);
  }
  cv::namedWindow("lines", cv::WINDOW_NORMAL);
  cv::imshow("lines", line_image);

