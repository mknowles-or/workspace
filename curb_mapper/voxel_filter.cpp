// Make a downsampled point cloud.
pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ds(new pcl::PointCloud<pcl::PointXYZ>());
pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
voxel_filter.setInputCloud(point_cloud_raw);
voxel_filter.setLeafSize(options.voxel_leaf_dim, options.voxel_leaf_dim, options.voxel_leaf_dim);
voxel_filter.filter(*point_cloud_ds);
OINFO("Downsampled point cloud size: {}", point_cloud_ds->size());
