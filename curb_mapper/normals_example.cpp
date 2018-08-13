Get the surface normals of ground points.
pcl::PointCloud<pcl::Normal>::Ptr normals = CalculateNormals(point_cloud_ground, point_cloud,
1.0);
viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(point_cloud_ground, normals, 1, 0.2,
"normals");
