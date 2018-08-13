
Show inlier points from ground plane.
pcl::ExtractIndices<pcl::PointXYZ> extract;
PointCloudXYZPtr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>());
extract.setInputCloud(point_cloud);
extract.setIndices(inliers);
extract.filter(*inlier_cloud);
viewer->addPointCloud(inlier_cloud, "inliers");
viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0,
                                       "inliers");

VisualizeGroundPlane(coeff, viewer);

Compute normals and display.
pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = CalculateNormals(point_cloud_ground,
point_cloud, 0.4);
viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(point_cloud_ground, cloud_normals, 10,
0.5, "normals");

Build an octree using the remaining points.
const float resolution = 0.6;
pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
octree.setInputCloud(point_cloud_ground);
octree.addPointsFromInputCloud();

// Get all occupied voxels.
std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> voxel_centers;
octree.getOccupiedVoxelCenters(voxel_centers);

// Compute the height range of points within each voxel.
std::vector<float> voxel_z_ranges;
voxel_z_ranges.resize(voxel_centers.size());

for (size_t i = 0; i < voxel_centers.size(); ++i) {
std::vector<int> point_indices_in_voxel;
float z_min = std::numeric_limits<float>::max();
float z_max = std::numeric_limits<float>::min();
octree.voxelSearch(voxel_centers.at(i), point_indices_in_voxel);

// Find the min and max z coord within this voxel.
for (size_t j = 0; j < point_indices_in_voxel.size(); ++j) {
  z_min = std::min(z_min, point_cloud_ground->at(j).z);
  z_max = std::max(z_max, point_cloud_ground->at(j).z);
}
voxel_z_ranges.at(i) = (z_max - z_min);
}

float max_range = 0;
float min_range = 0;
float total = 0;
for (const float range : voxel_z_ranges) {
min_range = std::min(min_range, range);
max_range = std::max(max_range, range);
total += range;
}
const float mean_range = (total / static_cast<double>(voxel_z_ranges.size()));
OINFO("ranges: min={} max={} mean={}", min_range, max_range, mean_range);
