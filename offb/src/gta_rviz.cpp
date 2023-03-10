#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>


#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>

using namespace std;

nav_msgs::Path obj_gt_path, obj_cal_path; //rviz Path msg
geometry_msgs::PoseStamped obj_gt_pose,obj_cal_pose; //object PoseStamped msg
ros::Publisher pub_obj_gt, pub_obj_cal; // rviz path pubilisher


// void callback(const geometry_msgs::PoseStamped::ConstPtr &obj_gt_sub, const geometry_msgs::PoseStamped::ConstPtr &obj_cal_sub)
// {
//     obj_gt_pose.pose = obj_gt_sub->pose;
//     obj_gt_path.poses.push_back(obj_gt_pose);
//     // cout<<"obj_gt_path size: "<<obj_gt_path.poses.size()<<endl;

//     obj_cal_pose.pose = obj_cal_sub->pose;
//     obj_cal_path.poses.push_back(obj_cal_pose);
//     // cout<<"obj_cal_path size: "<<obj_cal_path.poses.size()<<endl;

//     pub_obj_gt.publish(obj_gt_path);
//     pub_obj_cal.publish(obj_cal_path);
// }

void gt_cb(const geometry_msgs::PoseStamped::ConstPtr gt)
{
    // cout<<"gt timestamp: "<<gt->header.stamp<<endl;
    obj_gt_pose.pose = gt->pose;
    obj_gt_path.poses.push_back(obj_gt_pose);
    pub_obj_gt.publish(obj_gt_path);
}  

void cal_cb(const geometry_msgs::PoseStamped::ConstPtr cal)
{
    // cout<<"cal timestamp: "<<cal->header.stamp<<endl;
    obj_cal_pose.pose = cal->pose;
    obj_cal_path.poses.push_back(obj_cal_pose);
    pub_obj_cal.publish(obj_cal_path);
}


int main(int argc, char** argv)
{
    cout<<"Object path visulization..."<<endl;

    ros::init(argc, argv, "rviz_node");
    ros::NodeHandle nh;

    obj_gt_path.header.frame_id = "world";
    obj_cal_path.header.frame_id = "world";

    // message_filters::Subscriber<geometry_msgs::PoseStamped> obj_gt_sub(nh, "/vrpn_client_node/gh034_ball_lhj/pose", 1);
    // message_filters::Subscriber<geometry_msgs::PoseStamped> obj_cal_sub(nh, "/scout_wp/pose", 1);
    // typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::PoseStamped, geometry_msgs::PoseStamped> MySyncPolicy;
    // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), obj_gt_sub, obj_cal_sub);
    // sync.registerCallback(boost::bind(&callback, _1, _2));


    ros::Subscriber gt = nh.subscribe("/vrpn_client_node/gh034_ball_lhj/pose", 1, gt_cb);
    ros::Subscriber cal = nh.subscribe<geometry_msgs::PoseStamped>("/scout_wp/pose", 1, cal_cb);

    pub_obj_gt =nh.advertise<nav_msgs::Path>("/obj_gt_path",1);
    pub_obj_cal =nh.advertise<nav_msgs::Path>("/obj_cal_path",1);

    // while(ros::ok())
    // {
    //     // pub_obj_gt.publish(obj_gt_path);
    //     // pub_obj_cal.publish(obj_cal_path);
    //     ros::spinOnce();
    // }
    ros::spin();

    return 0;
}