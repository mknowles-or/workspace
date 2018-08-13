for (size_t i = 0; i < polyline_points.size(); ++i) {
  const Vector2d& pt = polyline_points.at(i);
  polyline_cloud->push_back(pcl::PointXYZ(pt.x(), pt.y(), 0.0f));
}
// Draw points underneath the curve as a hack to make the polygon look like a polyline.
for (size_t i = polyline_points.size(); i > 0; --i) {
  const Vector2d& pt = polyline_points.at(i - 1);
  polyline_cloud->push_back(pcl::PointXYZ(pt.x(), pt.y(), 0.1f));
}

viewer->addPolygon<pcl::PointXYZ>(polyline_cloud, 1.0f, 0.0f, 0.0f, "spline", 0);
viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "spline");
