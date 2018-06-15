#include <csignal>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <boost/program_options.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mc/common/ocommon/file_utils.hpp"
#include "mc/common/ocommon/olog.hpp"
#include "mc/common/ocommon/variant.hpp"

#include "mc/common/oimage/calibration_settings.hpp"
#include "mc/common/oimage/color_conversion.hpp"
#include "mc/common/oimage/data_buffer_allocator.hpp"
#include "mc/common/oimage/oframe.hpp"
#include "mc/common/oimage/oimage.hpp"
#include "mc/common/oimage/opencv_adapters.hpp"

#include "mc/common/curves2d/cubic_hermite_spline.hpp"
#include "mc/common/omath/plane.hpp"

#include "mc/common/opath/opath.hpp"
#include "mc/common/osettings/settings.hpp"
#include "mc/common/vehicle_model/vehicle_model.hpp"
#include "mc/common/vehicle_settings/vehicle_settings_loader.hpp"

#include "mc/infrastructure/dataset/dataset_info.hpp"
#include "mc/perception/camera_logger/examples/frame_write_mode.hpp"
#include "mc/perception/camera_logger/reader/camera_log_reader.hpp"
#include "mc/perception/camera_logger/reader/camera_log_reader_loader.hpp"
#include "mc/perception/cuda_utils/cuda_buffer_allocator.hpp"
#include "mc/perception/cuda_utils/cuda_utils.hpp"
#include "workstation/mapping/lane_mapper/lane_detector.hpp"

#define OLOG_GROUP "lane_detection_example"

static volatile std::sig_atomic_t g_run = 1;

void SignalHandler(int /*signum*/)
{
  g_run = 0;
}

namespace ori {
namespace mapping {

struct LaneDetectionExampleSettings
{
  std::string dataset_id = "beacon1_180603_170559";
  bool show_images = true;
  std::vector<std::string> sensor_ids = {"camera_front_left", "camera_front_right"};
  bool use_active_vehicle_settings = false;
  std::int64_t skip_frames = 0;
  perception::FrameWriteMode write_mode = perception::FrameWriteMode::NW;
  perception::FrameWriteFormat write_format = perception::FrameWriteFormat::PNG;
};

class LaneDetectionExample {
 public:
  explicit LaneDetectionExample(const LaneDetectionExampleSettings& settings);

  void Run();

 private:
  std::vector<cv::Mat> SplitAndConvertFrameImages(const oimage::OFramePtrVector& oframe_vector,
                                                  const bool annotate);

  void ShowImages(const oimage::OFramePtrVector& oframe_vector);

  void WriteOutputImages(const oimage::OFramePtr& oframe);

 private:
  const LaneDetectionExampleSettings settings_;
  oimage::DataBufferAllocatorPtr allocator_;

  // Directory for writing output images.
  std::string output_dir_;
  perception::CameraLogReaderPtr camera_log_reader_;
  SettingsPtr vehicle_settings_;
};

LaneDetectionExample::LaneDetectionExample(const LaneDetectionExampleSettings& settings)
    : settings_(settings)
{
  perception::SetCudaUnifiedMemoryDevice();
  allocator_ = oimage::AllocatorCudaManaged();

  // Set directory for output files.
  const std::string analysis_root = opath::OPath::GetAnalysisRoot();
  output_dir_ = fmt::format("{}/camera_log_reader_example/{}", analysis_root, settings_.dataset_id);

  // Load vehicle settings from the dataset.
  VehicleModelPtr dataset_vehicle_model;
  std::string dataset_vehicle_id;
  dataset::DatasetInfo dataset_info(settings_.dataset_id);
  if (dataset_info.HaveSettings()) {
    OWARN("Dataset settings found!");
    vehicle_settings_ = dataset_info.GetDatasetSettings();
    dataset_vehicle_model = VehicleModel::CreateFromSettings(vehicle_settings_);
    dataset_vehicle_id = dataset_vehicle_model->GetVehicleId();
  } else {
    OWARN("Dataset doesn't have settings.");
  }

  if (settings_.use_active_vehicle_settings) {
    vehicle_settings_ = vsettings::VehicleSettingsLoader::Create()->LoadActiveVehicle();
    OINFO("Using active vehicle settings.");
  } else {
    if (!vehicle_settings_) {
      throw std::runtime_error(
          "Dataset doesn't have vehicle settings, consider using the active vehicle.");
    }
    OINFO("Using dataset vehicle settings.");
    dataset::DatasetInfo dataset_info{settings_.dataset_id};
  }

  // Get vehicle model and id.
  auto vehicle_model(VehicleModel::CreateFromSettings(vehicle_settings_));
  const std::string vehicle_id(vehicle_model->GetVehicleId());
  if (vehicle_id != dataset_vehicle_id) {
    OWARN("Active vehicle id '{}' doesn't match dataset vehicle id '{}'", vehicle_id,
          dataset_vehicle_id);
  }
  OINFO("Vehicle id: {}", vehicle_id);

  // Create a reader for camera data.
  camera_log_reader_ = perception::CameraLogReaderLoader::GetCameraLogReader(
      vehicle_settings_, settings_.dataset_id, settings_.sensor_ids, allocator_);
}

std::vector<cv::Mat> LaneDetectionExample::SplitAndConvertFrameImages(
    const oimage::OFramePtrVector& oframe_vector, const bool annotate)
{
  std::vector<cv::Mat> images;
  for (const auto& frame : oframe_vector) {
    oimage::OImagePtr image{frame->image};

    OINFO_RATE(1, "image: {}x{} ({})", image->Width(), image->Height(),
               FourCCToString(image->ImageFormat()));

    oimage::OImagePtr image_bgr =
        oimage::OImage::Create(image->Width(), image->Height(), oimage::FourCC::BGR, allocator_);

    oimage::ConvertImage(frame->image, image_bgr);

    cv::Mat mat_bgr{oimage::CVWrapOImage(image_bgr)};

    if (annotate) {
      cv::putText(mat_bgr, fmt::format("{} {} ({})", frame->info.sensor_id,
                                       frame->info.timestamp.ToISOString(),
                                       frame->info.timestamp.Microseconds()),
                  cv::Point(0, 50), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 0, 0), 2.0);
    }
    cv::Mat mat;
    mat_bgr.copyTo(mat);
    images.emplace_back(mat);
  }
  return images;
}

void LaneDetectionExample::ShowImages(const oimage::OFramePtrVector& oframe_vector)
{
  std::vector<cv::Mat> mat_bgr(SplitAndConvertFrameImages(oframe_vector, true));

  // Display images side-by-side.
  cv::Mat output_bgr;
  cv::hconcat(mat_bgr, output_bgr);

  cv::namedWindow(settings_.dataset_id, cv::WINDOW_NORMAL);
  cv::imshow(settings_.dataset_id, output_bgr);
  cv::waitKey(1);
  return;
}

void LaneDetectionExample::Run()
{
  // Options are NW (no-write), WCV (write-concat-vert), WCH (write-concat-horiz), WS (write-split).
  if (settings_.write_mode != perception::FrameWriteMode::NW) {
    OINFO("Writing images to {}", output_dir_);
    FileMakeDirs(output_dir_);
  }

  if (settings_.skip_frames > 0) {
    OINFO("Skipping {} frames", settings_.skip_frames);
    camera_log_reader_->SkipOFrames(settings_.skip_frames);
  }

  std::int64_t total_frame_count = 0;

  // Create the lane detector.
  LaneDetector::Params lane_detector_params;
  LaneDetector lane_detector(lane_detector_params);

  oimage::CameraCalibrationPtrVectorPtr camera_intrinsics =
      oimage::LoadCameraCalibrationPtrVectorPtrFromSettings(vehicle_settings_,
                                                            settings_.sensor_ids);

  perception::SetCudaUnifiedMemoryDevice();
  oimage::DataBufferAllocatorPtr allocator_ = oimage::AllocatorCudaManaged();

  // This will use the default cropping policy (kCropToAllValidPixels).
  stereo::StereoRectifyUtilsPtr rectifier_ = stereo::StereoRectifyUtils::Create(camera_intrinsics);

  // Main loop.
  while (g_run) {
    OINFO_RATE(1, "Read {} frames", total_frame_count);

    oimage::OFramePtrVector oframes(camera_log_reader_->ReadOFrames());

    // check if any camera is done, if so exit loop.
    if (oframes.size() != camera_log_reader_->GetCameraIds().size()) {
      break;
    }

    // Get a stereo pair.
    oimage::OImagePtrVector input_oimages;
    for (size_t ii = 0; ii < oframes.size(); ii++) {
      input_oimages.push_back(oframes[ii]->image);
    }

    // Convert oimages to BGR format.
    oimage::OImagePtrVector input_oimages_bgr =
        oimage::ConvertImages(input_oimages, oimage::FourCC::BGR, allocator_);

    // Rectify the images (note: size will change).
    oimage::OImagePtrVector rectified_oimages_bgr;
    rectifier_->RectifyImages(input_oimages_bgr, &rectified_oimages_bgr, allocator_);

    // TODO: get the actual plane, for now just use y = 2.0 (camera y is down).
    float theta = 10 * 3.14195 / 180;
    math::Plane ground_plane(0, std::cos(theta), std::sin(theta), -1.5);
    std::vector<curves2d::CubicHermiteSpline> lane_features;
    lane_detector.ExtractLaneFeatures(rectified_oimages_bgr, ground_plane, camera_intrinsics,
                                      &lane_features);

    // Show the input stereo pair.
    if (settings_.show_images) {
      ShowImages(oframes);
    }
    ++total_frame_count;
  }
  OINFO("Read {} frames in total", total_frame_count);
  return;
}

}  // namespace mapping
}  // namespace ori

int main(int argc, char** /*argv[]*/)
{
  using namespace ori;
  using namespace mapping;

  // Install signal handlers.
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  // Need this because compiler warnings treater as errors.
  // Second arg should be the video mode flag.
  bool videoMode = (argc >= 2);

  if (videoMode) {
    LaneDetectionExampleSettings settings;
    settings.skip_frames = 100;
    LaneDetectionExample lane_detection_example(settings);
    lane_detection_example.Run();
  } else {
    std::string test_img_path = "/home/mknowles/Desktop/img2.png";
    // TODO: test on a static image
  }

  return 0;
}
