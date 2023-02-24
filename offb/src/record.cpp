#include <iostream>
#include <ros/ros.h>
#include "include/movement.h"
#include "geometry_msgs/PointStamped.h"
#include "geometry_msgs/PoseStamped.h"
#include <std_msgs/Bool.h>



static double err_x, err_y, err_z;


// void position_callback(const geometry_msgs::PoseStamped::ConstPtr& pose)
// {
//     ofstream save("/home/luo/scout_ws/src/scout_ros/lala/src/log/mav_pos.txt",ios::app);
//     save<<pose->pose.position.x<<endl;
//     save<<pose->pose.position.y<<endl;
//     save<<pose->pose.position.z<<endl;
//     save.close();
// }

// void velocity_callback(const geometry_msgs::TwistStamped::ConstPtr& velocity)
// {
//     ofstream save("/home/luo/scout_ws/src/scout_ros/lala/src/log/mav_vel.txt",ios::app);
//     save<<velocity->twist.linear.x<<endl;
//     save<<velocity->twist.linear.y<<endl;
//     save<<velocity->twist.linear.z<<endl;
//     save<<velocity->twist.angular.x<<endl;
//     save<<velocity->twist.angular.y<<endl;
//     save<<velocity->twist.angular.z<<endl;
//     save.close();
// }


static bool meaornot;

void mea_cb(const std_msgs::Bool::ConstPtr& msg)
{
    meaornot = msg->data;
}


void err_cal(const geometry_msgs::PoseStamped::ConstPtr obj_pose_w, const geometry_msgs::PoseStamped::ConstPtr obj_pose_gt)
{
    cout<<"lala"<<endl;

    err_x = obj_pose_gt->pose.position.x-obj_pose_w->pose.position.x;
    err_y = obj_pose_gt->pose.position.y-obj_pose_w->pose.position.y;
    err_z = obj_pose_gt->pose.position.z-obj_pose_w->pose.position.z;

    ofstream save("/home/luo/gta/src/AUTO/offb/src/log/ball_yolotest1_gt.txt",ios::app);
    save<<obj_pose_gt->pose.position.x<<","<<obj_pose_gt->pose.position.y<<","<<obj_pose_gt->pose.position.z<<","<<endl;
    // save<<obj_pose_gt->pose.position.x<<endl;
    // save<<obj_pose_gt->pose.position.y<<endl;
    // save<<obj_pose_gt->pose.position.z<<endl;
    // save<<obj_pose_gt->pose.orientation.w<<endl;
    // save<<obj_pose_gt->pose.orientation.x<<endl;
    // save<<obj_pose_gt->pose.orientation.y<<endl;
    // save<<obj_pose_gt->pose.orientation.z<<endl;
    save.close();

    ofstream save_3("/home/luo/gta/src/AUTO/offb/src/log/ball_yolotest1.txt",ios::app);
    save_3<<obj_pose_w->pose.position.x<<","<<obj_pose_w->pose.position.y<<","<<obj_pose_w->pose.position.z<<","<<endl;

    // save_3<<obj_pose_w->pose.position.x<<endl;
    // save_3<<obj_pose_w->pose.position.y<<endl;
    // save_3<<obj_pose_w->pose.position.z<<endl;
    // save_3<<obj_pose_w->pose.orientation.w<<endl;
    // save_3<<obj_pose_w->pose.orientation.x<<endl;
    // save_3<<obj_pose_w->pose.orientation.y<<endl;
    // save_3<<obj_pose_w->pose.orientation.z<<endl;
    save_3.close();

    ofstream save_2("/home/luo/gta/src/AUTO/offb/src/log/err_yolotest1_xyz.txt",ios::app);
    save_2<<err_x<<","<<err_y<<","<<err_z<<","<<endl;
    save_2.close();
}

void record_cam(const geometry_msgs::PoseStamped::ConstPtr obj_pose_gt, const geometry_msgs::PoseStamped::ConstPtr cam_pose_gt)
{
    ofstream save("/home/luo/gta/src/AUTO/offb/src/log/cam_yolotest1_gt.txt",ios::app);
    save<<cam_pose_gt->pose.position.x<<","<<cam_pose_gt->pose.position.y<<","<<cam_pose_gt->pose.position.z<<","<<endl;
    // save<<cam_pose_gt->pose.position.y<<endl;
    // save<<cam_pose_gt->pose.position.z<<endl;
    // save<<cam_pose_gt->pose.orientation.w<<endl;
    // save<<cam_pose_gt->pose.orientation.x<<endl;
    // save<<cam_pose_gt->pose.orientation.y<<endl;
    // save<<cam_pose_gt->pose.orientation.z<<endl;
    save.close();
}


int main(int argc, char **argv)
{
    cout<<"Recording data..."<<endl;
    ros::init(argc, argv, "inforecorder");
    ros::NodeHandle nh;

    message_filters::Subscriber<geometry_msgs::PoseStamped> obj_info_sub(nh, "/vrpn_client_node/gh034_ball_lhj/pose", 1);
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_obj_w(nh, "/scout_wp/pose", 1);  
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_cam_w(nh, "/vrpn_client_node/gh034_sensor_lhj/pose", 1);
    typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::PoseStamped, geometry_msgs::PoseStamped> MySyncPolicy_r;
    message_filters::Synchronizer<MySyncPolicy_r> syncon(MySyncPolicy_r(10), obj_info_sub, sub_obj_w);
    syncon.registerCallback(boost::bind(&err_cal, _1, _2));

    message_filters::Synchronizer<MySyncPolicy_r> syncon_2(MySyncPolicy_r(10), obj_info_sub, sub_cam_w);
    syncon_2.registerCallback(boost::bind(&record_cam, _1, _2));

    // ros::Subscriber mav_pos_sub = nh.subscribe<geometry_msgs::PoseStamped>
    //                                ("/mavros/local_position/pose", 1, position_callback);
    // ros::Subscriber mav_vel_sub = nh.subscribe<geometry_msgs::TwistStamped>
    //                                ("/mavros/local_position/velocity_local", 1, velocity_callback);

    ros::Subscriber sub_mea= nh.subscribe<std_msgs::Bool>
                              ("/obj_found", 1, mea_cb);




    ros::Rate rate(20.0);
    while(ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }
    ros::spin();
    return 0;




}



