Compute gradient of height map.
perception::UniformGridMap<float> gradient_map = ComputeGridmapGradient(height_map);

Theshold to remove points without significant gradient magnitude.
const float gradient_threshold_value = 0.5; // TODO: make a param.
ThresholdGridMap(&gradient_map, gradient_threshold_value, 0.0f, 1.0f);
const cv::Mat1f& gradient_image = ConvertGridMapToImage(gradient_map).Image();

const cv::Mat1b& gradient_viewable = MakeViewableImage(gradient_image);
cv::namedWindow("gradient", cv::WINDOW_NORMAL);
cv::imshow("gradient", gradient_viewable);
