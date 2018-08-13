/**
 * @brief Calculate surface normal at each point in a point cloud, using a (denser) cloud.
 */
pcl::PointCloud<pcl::Normal>::Ptr CalculateNormals(const PointCloudXYZPtr poi_cloud,
                                                   const PointCloudXYZPtr full_cloud,
                                                   const float normal_search_radius)
{
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
  normal_estimation.setInputCloud(poi_cloud);
  normal_estimation.setSearchSurface(full_cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZ>());
  normal_estimation.setSearchMethod(kd_tree);
  normal_estimation.setRadiusSearch(normal_search_radius);

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  normal_estimation.compute(*normals);

  return normals;
}
