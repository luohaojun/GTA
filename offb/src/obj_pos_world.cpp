#include <iostream>
#include <ros/ros.h>
#include "include/movement.h"
#include "geometry_msgs/PointStamped.h"
#include "geometry_msgs/PoseStamped.h"
#include <std_msgs/Bool.h>

using namespace std;


static UAVpose caminfo;// camera pose ground truth
static double obj_x_c, obj_y_c, obj_z_c;
static double x_prev = 0, y_prev = 0, z_prev = 0; // previous obj_camera;

Eigen::Matrix<double, 4, 1> obj_world;



void obj_info_cb(const geometry_msgs::PointStampedConstPtr& msg)
{
    obj_x_c = msg->point.x;
    obj_y_c = msg->point.y;
    obj_z_c = msg->point.z;
    if(obj_x_c==0)
        obj_x_c = x_prev;
    if(obj_y_c==0)
        obj_y_c = y_prev;
    if(obj_z_c==0)
        obj_z_c = z_prev;
    if(obj_x_c!=0)
        x_prev = obj_x_c ;
    if(obj_y_c!=0)
        y_prev = obj_y_c;
    if(obj_z_c!=0)
        z_prev = obj_z_c;

}


static bool meaornot;

void mea_cb(const std_msgs::Bool::ConstPtr& msg)
{
    meaornot = msg->data;
}


geometry_msgs::PoseStamped obj_pose_w;
void obj_pose_world_cb(const geometry_msgs::PoseStamped::ConstPtr &cam_pos_w, const geometry_msgs::PoseStamped::ConstPtr &obj_pos_c)
{
    

    caminfo.x = cam_pos_w->pose.position.x;
    caminfo.y = cam_pos_w->pose.position.y;
    caminfo.z = cam_pos_w->pose.position.z;
    caminfo.ow = cam_pos_w->pose.orientation.w;
    caminfo.ox = cam_pos_w->pose.orientation.x;
    caminfo.oy = cam_pos_w->pose.orientation.y;
    caminfo.oz = cam_pos_w->pose.orientation.z;

    obj_x_c = obj_pos_c->pose.position.x;
    obj_y_c = obj_pos_c->pose.position.y;
    obj_z_c = obj_pos_c->pose.position.z;
    obj_pose_w.header.stamp = obj_pos_c->header.stamp;

    if (obj_z_c == 0)
    {
        obj_x_c = x_prev;
        obj_y_c = y_prev;
        obj_z_c = z_prev;
    }
        
    if (obj_z_c != 0)
    {
        x_prev = obj_x_c;       
        y_prev = obj_y_c;
        z_prev = obj_z_c;
    }

    Eigen::Matrix<double, 4, 1> cam (obj_x_c,obj_y_c,obj_z_c,1), offset(0.0, 0.0, 0.0, 0);
    Eigen::Matrix<double, 4, 4> cam_to_body;
    cam_to_body << 0,0,1,0,
        -1,0,0,0,
        0,-1,0,0,
        0,0,0,1;

    Eigen::Matrix<double, 3, 3> matrix_for_q;
    Eigen::Quaterniond q2r_matrix(caminfo.ow, caminfo.ox, caminfo.oy, caminfo.oz);
    matrix_for_q = q2r_matrix.toRotationMatrix();

    Eigen::Matrix<double, 4, 4> body_to_world;
    body_to_world << matrix_for_q(0, 0), matrix_for_q(0, 1), matrix_for_q(0, 2), caminfo.x,
        matrix_for_q(1, 0), matrix_for_q(1, 1), matrix_for_q(1, 2), caminfo.y,
        matrix_for_q(2, 0), matrix_for_q(2, 1), matrix_for_q(2, 2), caminfo.z,
        0, 0, 0, 1;

    obj_world = body_to_world * cam_to_body * cam;


    // Eigen::Matrix<double, 3, 1> obj_cam(obj_x_c,obj_y_c,obj_z_c), cam_wolrd(caminfo.x,caminfo.y,caminfo.z);
    // obj_world = obj_cam + cam_wolrd;

}




int main(int argc, char **argv)
{
    cout<<"Transforming obj pose to world frame..."<<endl;
    ros::init(argc, argv, "Cal_obj_pose_wolrd");
    ros::NodeHandle nh;

    // ros::Subscriber obj_info_sub = nh.subscribe<geometry_msgs::PoseStamped>
    //                                ("/vrpn_client_node/gh034_luo/pose", 1, obj_gt_cb);
    // ros::Subscriber mav_pos_sub = nh.subscribe<geometry_msgs::PoseStamped>
    //                                ("/mavros/local_position/pose", 1, position_callback);
    // ros::Subscriber mav_vel_sub = nh.subscribe<geometry_msgs::TwistStamped>
    //                                ("/mavros/local_position/velocity_local", 1, velocity_callback);
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_cam_w(nh, "/vrpn_client_node/gh034_sensor_lhj/pose", 1);
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_obj_c(nh, "/obj_pose_cam", 1);  
    typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::PoseStamped, geometry_msgs::PoseStamped> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_cam_w, sub_obj_c);
    sync.registerCallback(boost::bind(&obj_pose_world_cb, _1, _2));

    ros::Subscriber sub_mea= nh.subscribe<std_msgs::Bool>
                              ("/obj_found", 1, mea_cb);

    // message_filters::Subscriber<sensor_msgs::CompressedImage> subimage(nh, "/camera/color/image_raw/compressed", 1);
    // message_filters::Subscriber<sensor_msgs::Image> subdepth(nh, "/camera/aligned_depth_to_color/image_raw", 1);
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, sensor_msgs::Image> MySyncPolicy;
    // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), subimage, subdepth);
    // sync.registerCallback(boost::bind(&callback, _1, _2));

    // ros::Subscriber camera_info_sub = nh.subscribe("/camera/aligned_depth_to_color/camera_info",1,camera_info_cb);

    ros::Publisher obj_pos_w = nh.advertise<geometry_msgs::PoseStamped> ("/scout_wp/pose",20);



    ros::Rate rate(20.0);
    while(ros::ok())
    {
        // geometry_msgs::PoseStamped obj_pose_w;
        obj_pose_w.pose.position.x = obj_world[0];
        obj_pose_w.pose.position.y = obj_world[1];
        obj_pose_w.pose.position.z = obj_world[2];
        // obj_pose_w.header.stamp = ros::Time::now();
        cout<<"time stamp of /scout_wp/pose: "<<obj_pose_w.header.stamp<<endl;
        obj_pos_w.publish(obj_pose_w);
        ros::spinOnce();
        rate.sleep();
    }
    ros::spin();
    return 0;




}



