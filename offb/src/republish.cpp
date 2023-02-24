#include <sstream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>



#include "include/run_yolo.h"
#include <string>
#include "offb/obj.h"

using namespace std;
sensor_msgs::PointCloud2 pcl_msg;
sensor_msgs::PointCloud2 pcl_msg_last;
sensor_msgs::Image img_msg;
sensor_msgs::Image img_msg_last;

void pcl_cb(const sensor_msgs::PointCloud2::ConstPtr pcl)
{
    pcl_msg_last = pcl_msg;
    pcl_msg = *pcl;
}

void img_cb(const sensor_msgs::Image::ConstPtr img)
{
    img_msg_last = img_msg;
    img_msg = *img;
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "Synchronizer");
    ros::NodeHandle node;

    ros::Subscriber pcl_sub = node.subscribe<sensor_msgs::PointCloud2>("/livox/lidar",1,pcl_cb);
    ros::Subscriber img_sub = node.subscribe<sensor_msgs::Image>("/camera/color/image_raw",1,img_cb);
    ros::Publisher pcl_pub = node.advertise<sensor_msgs::PointCloud2>("/pcl_syn",1);
    ros::Publisher img_pub = node.advertise<sensor_msgs::Image>("/img_syn",1);

    while(ros::ok())
    {
        if(pcl_msg_last.header.stamp != pcl_msg.header.stamp)
        {
            pcl_pub.publish(pcl_msg);
        }
        if(img_msg_last.header.stamp != img_msg.header.stamp)
        {
            img_pub.publish(img_msg);
        }
        ros::spinOnce();
    }
    ros::spin();
    return 0;
}
