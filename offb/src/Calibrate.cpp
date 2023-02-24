#include <sstream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include "geometry_msgs/PointStamped.h"
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>


#include "include/run_yolo.h"
#include <string>
#include "offb/obj.h"

using namespace Eigen;
using namespace std;


Vector3d rgb_2_lidar = {-0.0369998, 0.0321837, 0.0480197};
// Vector3d rgb_2_lidar;
cv::Mat P_rect_00(3, 4, cv::DataType<double>::type);
cv::Mat R_rect_00(4, 4, cv::DataType<double>::type);
cv::Mat RT(4, 4, cv::DataType<double>::type); 
pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
// cv::Mat visImg;

void Calibrate()
{
    Vector3d left_2_lidar = {-0.0367, 0.0175, 0.04790};
    // cout<<"left_2_lidar: "<<left_2_lidar<<endl;
    Vector3d RGB_2_left = {-0.0146837, -0.000119724, -0.000299758};
    // cout<<"RGB_2_left: "<<RGB_2_left<<endl;
    Matrix<double,3,3> cam_2_lidar;
    cam_2_lidar << 0,0,1,
    -1,0,0,
    0,-1,0;
    // cout<<"cam_2_lidar: \n"<<cam_2_lidar<<endl;
    rgb_2_lidar = left_2_lidar + cam_2_lidar * RGB_2_left;
    cout<<"extrinsic: " <<rgb_2_lidar<<endl;
    cout<<"Extrinsic done!"<<endl;
}

void camera_info_cb(const sensor_msgs::CameraInfoPtr& msg )
{
    RT.at<double>(0,0) = 0; RT.at<double>(0,1) = -1; RT.at<double>(0,2) = 0; RT.at<double>(0,3) = rgb_2_lidar[0];
    RT.at<double>(1,0) = 0; RT.at<double>(1,1) = 0; RT.at<double>(1,2) = -1; RT.at<double>(1,3) = rgb_2_lidar[1];
    RT.at<double>(2,0) = 1; RT.at<double>(2,1) = 0; RT.at<double>(2,2) = 0; RT.at<double>(2,3) = rgb_2_lidar[2];
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 1; R_rect_00.at<double>(0,1) = 0; R_rect_00.at<double>(0,2) = 0; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = 0; R_rect_00.at<double>(1,1) = 1; R_rect_00.at<double>(1,2) = 0; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 0; R_rect_00.at<double>(2,1) = 0; R_rect_00.at<double>(2,2) = 1; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = msg->K[0]; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = msg->K[2]; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = msg->K[4]; P_rect_00.at<double>(1,2) = msg->K[5]; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;
    // cout<<"camera_info_cb done!"<<endl;

}

// void lidaronimg()
// {
// 	// store calibration data in OpenCV matrices
// 	// cv::Mat P_rect_00(3, 4, cv::DataType<double>::type);
// 	// cv::Mat R_rect_00(4, 4, cv::DataType<double>::type);
// 	// cv::Mat RT(4, 4, cv::DataType<double>::type);

// 	// project lidar points
// 	cv::Mat overlay = visImg.clone();
// 	cv::Mat X(4, 1, cv::DataType<double>::type);
// 	cv::Mat Y(3, 1, cv::DataType<double>::type);

// 	for(int i=0; i<=pc->points.size(); i++)
// 	{

// 		X.at<double>(0,0) = pc->points[i].x;
// 		X.at<double>(1,0) = pc->points[i].y;
// 		X.at<double>(2,0) = pc->points[i].z;
// 		X.at<double>(3,0) = 1;
// 		Y = P_rect_00 * R_rect_00 * RT *X;
// 		cv::Point pt;
// 		pt.x = Y.at<double>(0, 0)/ Y.at<double>(2,0);
// 		pt.y = Y.at<double>(1, 0)/ Y.at<double>(2,0);
//         cout<<"x: "<<pt.x<<endl;
//         cout<<"y: "<<pt.y<<endl;

//         if(pt.x >= 1260 || pt.y >= 700 || pt.x <= 10 || pt.y <= 10)
//         {
//             continue;
//         }

// 		// float val = pc->points[i].x;
// 		// float maxVal = 20.0;
// 		// int red = min(255, (int)(255 * abs((val - maxVal) / maxVal)));
// 		// int green = min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal ))));
// 		cv::circle(overlay, pt, 1, cv::Scalar(0, 255, 0), cv::FILLED);

// 	}

// 	// float opacity = 0.6;
// 	// cv::addWeighted(overlay, opacity, visImg, 1-opacity, 0, visImg);

// 	string windowName = "LiDAR data on image overlay";

// 	cv::namedWindow( windowName, cv::WINDOW_AUTOSIZE);
// 	cv::imshow(windowName, overlay);
// 	cv::waitKey(20);

// }

void callback(const sensor_msgs::CompressedImage::ConstPtr &img ,const sensor_msgs::PointCloud2::ConstPtr &pcl)
{
    cv::Mat visImg;    
    visImg = cv::imdecode(cv::Mat(img->data),1);
    cv::Mat overlay = visImg.clone();
	cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);

    pcl::fromROSMsg(*pcl, *pc);

    for(int i=0; i<=pc->points.size(); i++)
	{

		X.at<double>(0,0) = pc->points[i].x;
		X.at<double>(1,0) = pc->points[i].y;
		X.at<double>(2,0) = pc->points[i].z;
		X.at<double>(3,0) = 1;
		Y = P_rect_00 * R_rect_00 * RT *X;
		cv::Point pt;
		pt.x = Y.at<double>(0, 0)/ Y.at<double>(2,0);
		pt.y = Y.at<double>(1, 0)/ Y.at<double>(2,0);
        cout<<"x: "<<pt.x<<endl;
        cout<<"y: "<<pt.y<<endl;

        if(pt.x >= 640 || pt.y >= 480 || pt.x <= 5 || pt.y <= 5)
        {
            continue;
        }

		// float val = pc->points[i].x;
		// float maxVal = 20.0;
		// int red = min(255, (int)(255 * abs((val - maxVal) / maxVal)));
		// int green = min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal ))));
		cv::circle(overlay, pt, 1, cv::Scalar(0, 255, 0), cv::FILLED);

	}

    string windowName = "LiDAR data on image overlay";

	cv::namedWindow( windowName, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowName, overlay);
	cv::waitKey(10);


}


// void cb_pointcloud(const sensor_msgs::PointCloud2::ConstPtr &pcl)
// {
//     pcl::fromROSMsg(*pcl, *pc);
//     // ROS_INFO("pcl_cb");
//     // cout<<"size of pc: "<<pc->points.size()<<endl;
//     lidaronimg();
//     // cout<<"first point: "<<pc->points[0]<<endl;
// }

// void cb_img(const sensor_msgs::Image::ConstPtr &img)
// {
//     cv_bridge::CvImageConstPtr rgb_ptr;
//     rgb_ptr = cv_bridge::toCvCopy(img,img->encoding);
//     visImg = rgb_ptr->image.clone();
//     // ROS_INFO("img_cb");
//     // cout<<visImg.cols<<endl;
//     // cout<<visImg.rows<<endl;
// }




int main(int argc, char **argv)
{
    ros::init(argc, argv, "Align_RGB_2_Lidar");
    ros::NodeHandle nh;
    
    message_filters::Subscriber<sensor_msgs::CompressedImage> subimage(nh, "/camera/color/image_raw/compressed", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> subpcl(nh, "/livox/lidar", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), subimage, subpcl);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    ros::Subscriber camera_info_sub = nh.subscribe("/camera/aligned_depth_to_color/camera_info", 1, camera_info_cb);
    // Calibrate();

    // ros::Subscriber pc_sub =nh.subscribe<sensor_msgs::PointCloud2>("/livox/lidar",1,cb_pointcloud);
    // ros::Subscriber img_sub =nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw",1,cb_img);
    
    ros::spin();
    
    return 0;
}