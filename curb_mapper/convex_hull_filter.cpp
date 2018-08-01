const double scale = 100.0;
bbox3d->push_back(pcl::PointXYZ(0, 0, 0));
bbox3d->push_back(pcl::PointXYZ(scale, 0, 0));
bbox3d->push_back(pcl::PointXYZ(scale, 0, scale));
bbox3d->push_back(pcl::PointXYZ(scale, scale, scale));
bbox3d->push_back(pcl::PointXYZ(0, scale, scale));
bbox3d->push_back(pcl::PointXYZ(0, scale, 0));
bbox3d->push_back(pcl::PointXYZ(0, 0, scale));
bbox3d->push_back(pcl::PointXYZ(scale, scale, 0));

// Get the surface points and vertices of the bounding box.
PointCloudXYZPtr hull_surface_points(new pcl::PointCloud<pcl::PointXYZ>());
std::vector<pcl::Vertices> hull_vertices;
pcl::ConvexHull<pcl::PointXYZ> convex_hull;
convex_hull.setDimension(3);
convex_hull.setComputeAreaVolume(true);
convex_hull.setInputCloud(bbox3d);
convex_hull.reconstruct(*hull_surface_points, hull_vertices);
OINFO("Constructed convex hull: {} points volume={}", hull_surface_points->size(), convex_hull.getTotalVolume());

// Create a crop hull filter using the computed convex hull.
pcl::CropHull<pcl::PointXYZ> crop_hull_filter;
crop_hull_filter.setDim(3);
crop_hull_filter.setInputCloud(full_point_cloud);
crop_hull_filter.setHullIndices(hull_vertices);
crop_hull_filter.setHullCloud(hull_surface_points);

// Create the output point cloud by cropping around convex hull.
PointCloudXYZPtr output_cloud(new pcl::PointCloud<pcl::PointXYZ>());
crop_hull_filter.filter(*output_cloud);
