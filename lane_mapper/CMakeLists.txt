ori_module(LaneMapping)

# TODO: which link is supplying the OpenCV include path?

ori_cc_library(lane_detector
  SOURCES
  lane_detector.hpp
  lane_detector.cpp

  LIBS
  oimage
  oimage_utils
  ocommon
  oimage_cuda # Need to link against this to get include dirs for cuda drivers.
  stereo_rectify
  ${OpenCV_LIBS}
)

ori_cc_library(lane_mapper
  SOURCES
  lane_mapper.hpp
  lane_mapper.cpp

  LIBS
  ocommon
)

ori_cc_binary(lane_detection_main
  SOURCES
  lane_detection_main.cpp

  LIBS
  oimage
  ocommon
  camera_logger
  logging_coordinator_collection
  vehicle_model
  Boost::program_options
  vehicle_settings_loader
  ${OpenCV_LIBS}
  lane_detector
  stereo_rectify
)