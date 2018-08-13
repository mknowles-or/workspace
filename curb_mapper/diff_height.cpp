
const float nominal_curb_height = 0.17;
cv::Mat1b absdiff;
cv::absdiff(gradient_image, cv::Scalar(4 * nominal_curb_height), absdiff);

// cv::Mat1f squared_diff = delta_nominal.mul(delta_nominal);
cv::Mat1b sd_viewable = MakeViewableImage(absdiff);
cv::namedWindow("squared", cv::WINDOW_NORMAL);
cv::imshow("squared", sd_viewable);
