// #include "include/yolo.h"
#include "run_ncnn.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/filter.h>


#include "offb/obj.h"

using namespace Eigen;
using namespace std;

static cv::Mat frame, res, gt;
static geometry_msgs::PoseStamped send;
Vector3d rgb_2_lidar = {-0.0369998, 0.0321837, 0.0480197}; //Extrinsic parameter between lidar and RGB camera

static char* parampath = "/home/luo/lala/src/haha/yolov4-tiny-opt.param";
static char* binpath = "/home/luo/lala/src/haha/yolov4-tiny-opt.bin";

pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);

cv::Mat P_rect_00(3, 4, cv::DataType<double>::type);
cv::Mat R_rect_00(4, 4, cv::DataType<double>::type);
cv::Mat RT(4, 4, cv::DataType<double>::type); 

int target_size = 416;




static double fx, fy, cx, cy; //focal length and principal point
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
    fx = msg->K[0];
    fy = msg->K[4];
    cx = msg->K[2];
    cy = msg->K[5];

}


geometry_msgs::PoseStamped obj_c(int x_pixel, int y_pixel, double z_depth)
{
    double z = z_depth; // the distance between center of the object is surface depth + object size
    double x = z * (x_pixel - cx)/fx;  //pixel coordinate u,v -> camera coordinate x,y
    double y = z * (y_pixel - cy)/fy;
    
    geometry_msgs::PoseStamped ps;
    ps.pose.position.x = x;
    ps.pose.position.y = y;
    ps.pose.position.z = z;
    ps.header.stamp = send.header.stamp;
    return ps;
}



void callback(const sensor_msgs::CompressedImageConstPtr & rgbimage, const sensor_msgs::PointCloud2::ConstPtr &pcl)
{
    // cv_bridge::CvImageConstPtr depth_ptr;
    // try
    // {
    //     depth_ptr  = cv_bridge::toCvCopy(depth, depth->encoding);
    // }
    // catch (cv_bridge::Exception& e)
    // {
    //     ROS_ERROR("cv_bridge exception: %s", e.what());
    //     return;
    // }
    // cv::Mat image_dep = depth_ptr->image;
    
    // try
    // {
    //     frame = cv::imdecode(cv::Mat(rgbimage->data),1);
    //     res   = cv::imdecode(cv::Mat(rgbimage->data),1);
    //     gt    = cv::imdecode(cv::Mat(rgbimage->data),1);
    // }
    // catch (cv_bridge::Exception& e)
    // {
    //     ROS_ERROR("cv_bridge exception: %s", e.what());
    // }
    // cout<<frame.size<<endl;

    send.header.stamp = rgbimage->header.stamp;

    try
    {
        pcl::fromROSMsg(*pcl, *pc);
        frame = cv::imdecode(cv::Mat(rgbimage->data),1);
        res   = cv::imdecode(cv::Mat(rgbimage->data),1);
        gt    = cv::imdecode(cv::Mat(rgbimage->data),1);
        
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    cout<<frame.size<<endl;
}



// remove closed points
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                            pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}





int main(int argc, char** argv)
{
    static run_ncnn yolonet(parampath, binpath, target_size);
    int counter = 0;

    int stateSize = 8;
    int measSize = 5;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type);  // [x,y,z,v_x,v_y,v_z,w,h] need z as well
    cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]


    cv::setIdentity(kf.transitionMatrix);

    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(9) = 1.0f;
    kf.measurementMatrix.at<float>(18) = 1.0f;

    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(9) = 1e-2;
    kf.processNoiseCov.at<float>(18) = 1e-2;
    kf.processNoiseCov.at<float>(27) = 5.0f;
    kf.processNoiseCov.at<float>(36) = 5.0f;
    kf.processNoiseCov.at<float>(45) = 5.0f;
    kf.processNoiseCov.at<float>(54) = 1e-2;
    kf.processNoiseCov.at<float>(63) = 1e-2;

    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

    int idx = 0;

    cout<<"Object detection..."<<endl;

    ros::init(argc, argv, "ncnn");
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::CompressedImage> subimage(nh, "/camera/color/image_raw/compressed", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> subpcl(nh, "/livox/lidar", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), subimage, subpcl);
    sync.registerCallback(boost::bind(&callback, _1, _2));


    ros::Subscriber camera_info_sub = nh.subscribe("/camera/aligned_depth_to_color/camera_info", 1, camera_info_cb);

    ros::Publisher publish_obj_c = nh.advertise<geometry_msgs::PoseStamped>("/obj_pose_cam", 1);
    ros::Publisher publish_found = nh.advertise<std_msgs::Bool>("/obj_found", 1);
    ros::Publisher publish_obj_vel = nh.advertise<offb::obj>("/obj_v", 1);

    // static double t1;
    // static double t2;
    bool found = false;
    bool measured =false;
    int notFoundCount = 0;
    int Failmeasured = 0;
    double dT;
    double tpf=0;
    int w = 200,h = 200;
    double ticks = 0;
    int i=0;

    cv::Point center_true;
    cv::Point center_pred;
    double depth = 0, depth_ = 0;
    double camera_z;
    double fps;
    double fps_average = 0;
    vector<double> fpss;
    double time_start, time_end;
    cv::Rect temprect;

    double prob;
    cv::Rect predRect;

    // ros::Rate rate_manager(40);

    while(ros::ok())
    {
        time_start = ros::Time::now().toSec();
        if(!frame.empty())
        {
            double precTick = ticks;
            ticks = (double) cv::getTickCount();

            dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
            if (found)
            {
                // >>>> Matrix A
                kf.transitionMatrix.at<float>(3) = dT;
                kf.transitionMatrix.at<float>(12) = dT;
                kf.transitionMatrix.at<float>(21) = dT;

                // <<<< Matrix A
                //            cout << "dT:" << endl << dT << endl;
                //            cout << "State post:" << endl << state << endl;

                state = kf.predict();

                cv::Point center;
                center.x = state.at<float>(0);
                center.y = state.at<float>(1);
                double z_c_temp = state.at<float>(2);

                predRect.width = temprect.width;
                predRect.height = temprect.height;
                predRect.x = state.at<float>(0) - predRect.width / 2;
                predRect.y = state.at<float>(1) - predRect.height / 2;


                // ofstream save(gt_txt, ios::app);
                // save<<counter<<endl;
                // save<<predRect.x <<endl;
                // save<<predRect.y <<endl;
                // save<<predRect.x + predRect.width <<endl;
                // save<<predRect.y + predRect.height<<endl<<endl;;
                // save.close();


                //                cv::Scalar color(rand()&255, rand()&255, rand()&255);
                //cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);
                center_pred = cv::Point(predRect.x+predRect.width/2, predRect.y+predRect.height/2);

                char temp_depth[40];
                sprintf(temp_depth, "%.2f", z_c_temp);
                string d = "Predicted z_C: ";
                string textoutputonframe =d+temp_depth + " m";
                cv::Point placetext = cv::Point((predRect.x-10),(predRect.y+predRect.height+24));
                //cv::putText(res, textoutputonframe, placetext,cv::FONT_HERSHEY_COMPLEX_SMALL,1,CV_RGB(255,0,0));
                depth_ = z_c_temp;
            }

            double starttime = ros::Time::now().toSec();

            yolonet.detect_yolo(frame);

            // Yolonet.rundarknet(frame);
            double endtime = ros::Time::now().toSec();
            double deltatime = endtime - starttime;
            //cout<<deltatime<<endl;
            fps = 1/deltatime;
            cout<<"fps: "<<fps<<endl;
            if(fpss.size()>2000)
                fpss.clear();
            fpss.push_back(fps);
            fps_average = accumulate(fpss.begin(), fpss.end(),0.0)/fpss.size();
            cout<<"fps_avg: "<<fps_average<<endl;

            vector<Object> temp = yolonet.objects;
            cout<<"temp size: "<<temp.size()<<endl;

            cv::Rect interested;
            vector<Object> potential;
            vector<float> potential_c;
            vector<float> potential_c_area;
            bool got=false;

            cv::Mat tempp;
            if(temp.size()!=0)
            {
                for(auto stuff:temp)
                {
                    cout<<stuff.classnameofdetection<<endl;
                    if(stuff.classnameofdetection=="ball")
                    {
                        potential.push_back(stuff);
                        potential_c.push_back(stuff.prob); //confidence
                        potential_c_area.push_back(stuff.rect.area());
                    }
                }
                cout<<"potential size: "<<potential.size()<<endl;

                if(potential.size()!=0)
                {
                    //                int maxElementIndex = max_element(potential_c.begin(),potential_c.end()) - potential_c.begin();
                    int maxElementIndex = max_element(potential_c_area.begin(),potential_c_area.end()) - potential_c_area.begin();
                    interested = potential[maxElementIndex].rect;
                    temprect = potential[maxElementIndex].rect;
                    // cout<<"potential[maxElementIndex].rect"<<potential[maxElementIndex].rect<<endl;

                    got = true;


                    tpf = yolonet.appro_fps;
                    w=interested.width;
                    h=interested.height;
                    tempp = potential[maxElementIndex].frame;

                    //get depth from lidar
                    /*******************fuse lidar and RGB camera************************/
                    cv::Mat overlay = potential[maxElementIndex].frame.clone();
                    cv::Mat X(4, 1, cv::DataType<double>::type);
                    cv::Mat Y(3, 1, cv::DataType<double>::type);
                    int depthbox_w = potential[maxElementIndex].rect.width * 0.25;
                    int depthbox_h = potential[maxElementIndex].rect.height * 0.25;
                    cv::Point depthbox_vertice1 = cv::Point(potential[maxElementIndex].center_bdbox.x - depthbox_w / 2, potential[maxElementIndex].center_bdbox.y - depthbox_w / 2);
                    cv::Point depthbox_vertice2 = cv::Point(potential[maxElementIndex].center_bdbox.x + depthbox_w / 2, potential[maxElementIndex].center_bdbox.y + depthbox_h / 2);
                    // cout<<"depthbox_vertice1: "<<depthbox_vertice1<<", depthbox_vertice2: "<<depthbox_vertice2<<endl;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr depth_pcl(new pcl::PointCloud<pcl::PointXYZ>);

                    // // remove NAN points
                    // std::vector<int> NAN_indices;
                    // pcl::removeNaNFromPointCloud(*pc, *pc, NAN_indices);

                    for (int i = 0; i <= pc->points.size(); i++)
                    {

                        X.at<double>(0, 0) = pc->points[i].x;
                        X.at<double>(1, 0) = pc->points[i].y;
                        X.at<double>(2, 0) = pc->points[i].z;
                        X.at<double>(3, 0) = 1;
                        Y = P_rect_00 * R_rect_00 * RT * X;
                        cv::Point pt;
                        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
                        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
                        // cout<<"lidar point on RGB image: "<<pt.x<<" "<<pt.y<<endl;

                        if (pt.x >= depthbox_vertice2.x || pt.y >= depthbox_vertice2.y || pt.x <= depthbox_vertice1.x || pt.y <= depthbox_vertice1.y)
                        {
                            continue;
                        }
                        else
                        {
                            depth_pcl->points.push_back(pc->points[i]);
                        }

                        cv::circle(overlay, pt, 1, cv::Scalar(0, 255, 0), cv::FILLED);
                    }
                    cout<<"depth_pcl->points.size(): "<<depth_pcl->points.size()<<endl;
                    if (depth_pcl->points.size() > 2)
                    {
                        // remove NAN points
                        std::vector<int> NAN_indices;
                        pcl::removeNaNFromPointCloud(*depth_pcl, *depth_pcl, NAN_indices);

                        // remove closed points
                        removeClosedPointCloud(*depth_pcl, *depth_pcl, 0.1);
                          

                        // Creating the KdTree object for the search method of the extraction
                        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
                        tree->setInputCloud(depth_pcl);

                        // Cluster
                        std::vector<pcl::PointIndices> cluster_indices;
                        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
                        ec.setClusterTolerance(0.1); // 2cm
                        ec.setMinClusterSize(1);
                        ec.setMaxClusterSize(25000);
                        ec.setSearchMethod(tree);
                        ec.setInputCloud(depth_pcl);
                        ec.extract(cluster_indices);

                        int j = 0;
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
                        for (const auto &cluster : cluster_indices)
                        {
                            for (const auto &idx : cluster.indices)
                            {
                                cloud_cluster->push_back((*depth_pcl)[idx]);
                            } //*
                            cloud_cluster->width = cloud_cluster->size();
                            cloud_cluster->height = 1;
                            cloud_cluster->is_dense = true;
                            j++;
                        }

                        Vector4f centroid_depth;
                        pcl::compute3DCentroid(*depth_pcl, centroid_depth);
                        X.at<double>(0, 0) = centroid_depth[0];
                        X.at<double>(1, 0) = centroid_depth[1];
                        X.at<double>(2, 0) = centroid_depth[2];
                        X.at<double>(3, 0) = 1;
                        Y = P_rect_00 * R_rect_00 * RT * X;
                        potential[maxElementIndex].depth = Y.at<double>(2, 0);
                        if(abs(potential[maxElementIndex].depth-depth_) > 1.5 && depth_ > 0.1)
                        {
                            measured = false;
                            notFoundCount++;
                            if (notFoundCount > 10)
                            {
                                found = false;
                            }
                        }
                        else
                        {
                            measured = true;
                            camera_z = potential[maxElementIndex].depth;
                            notFoundCount = 0;
                        }
                        
                    }
                    else
                    {
                        measured = false;
                        notFoundCount++;
                        if (notFoundCount > 10)
                        {
                            found = false;
                        }
                        cout<<"No point cloud found!"<<endl;
                    }

                    string windowName = "LiDAR data on image overlay";
                    // cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                    // cv::imshow(windowName, overlay);
                    // cv::waitKey(20);
                    // videolidar.write(overlay);
                    /******************************fuse lidar and RGB camera**********************/

                    prob = potential[maxElementIndex].prob;

                    char temp_depth[40];
                    sprintf(temp_depth, "%.2f", camera_z);
                    string d = "z_C: ";
                    string textoutputonframe =d+temp_depth + " m";
                    cv::Point placetext = cv::Point((interested.x-10),(interested.y-24));


                    if(prob > 30)
                    {
                        center_true=cv::Point(interested.x+interested.width/2, interested.y+interested.height/2);
                        cv::rectangle(res, interested, CV_RGB(255,255,0), 1);
                        //cv::putText(res, textoutputonframe, placetext,cv::FONT_HERSHEY_COMPLEX_SMALL,1,CV_RGB(255,255,0));
                    }
                }
            }
            time_end=ros::Time::now().toSec();
            cout<<"ms: "<<time_end-time_start<<endl<<endl;
            if(!got)
            {
                notFoundCount++;
                measured = false;
                cout << "notFoundCount:" << notFoundCount << endl;
                if(notFoundCount>10)
                {
                    found = false;
                }
            }
            else
            {
                //            cout<<"hey"<<endl;
                // measured = true;
                // notFoundCount = 0;
                if(measured)
                {
                    notFoundCount = 0;
                    meas.at<float>(0) = interested.x + interested.width / 2;
                    meas.at<float>(1) = interested.y + interested.height / 2;
                    meas.at<float>(2) = camera_z;
                    meas.at<float>(3) = interested.width;
                    meas.at<float>(4) = interested.height;
                }

                
                if (!found) // First detection!
                {
                    // >>>> Initialization
                    kf.errorCovPre.at<float>(0) = 1; // px
                    kf.errorCovPre.at<float>(9) = 1; // px
                    kf.errorCovPre.at<float>(18) = 1;
                    kf.errorCovPre.at<float>(27) = 1;
                    kf.errorCovPre.at<float>(36) = 1; // px
                    kf.errorCovPre.at<float>(45) = 1; // px
                    kf.errorCovPre.at<float>(54) = 1; // px
                    kf.errorCovPre.at<float>(63) = 1; // px

                    state.at<float>(0) = meas.at<float>(0);
                    state.at<float>(1) = meas.at<float>(1);
                    state.at<float>(2) = meas.at<float>(2);
                    state.at<float>(3) = 0;
                    state.at<float>(4) = 0;
                    state.at<float>(5) = 0;
                    state.at<float>(6) = meas.at<float>(3);
                    state.at<float>(7) = meas.at<float>(4);
                    // <<<< Initialization

                    kf.statePost = state;
                    found = true;
                }
                else if(measured)
                {
                    kf.correct(meas); // Kalman Correction                    

                    cv::Point center;
                    center.x = state.at<float>(0);
                    center.y = state.at<float>(1);
                    depth = state.at<float>(2);

                    cv::Rect Rect;
                    Rect.width = temprect.width;
                    Rect.height = temprect.height;
                    Rect.x = state.at<float>(0) - Rect.width / 2;
                    Rect.y = state.at<float>(1) - Rect.height / 2;
                    center_true=cv::Point(Rect.x+Rect.width/2, Rect.y+Rect.height/2);
                    if(prob <= 30)
                        cv::rectangle(res, Rect, CV_RGB(0,255,0), 1);
                }
            }
            cv::Mat display;

            if(measured)
            {
                cout<<"show measure: "<<endl;
                send = obj_c(center_true.x, center_true.y,depth);
                // send.header.stamp = ros::Time::now();
                cout<<"object depth: "<<depth<<endl;
            }
            else
            {
                cout<<"show predict"<<endl;
                cv::rectangle(res, predRect, CV_RGB(255,0,0), 1);
                send = obj_c(center_pred.x, center_pred.y,depth_);
                // send.header.stamp = ros::Time::now();
                 cout<<"object depth_: "<<depth_<<endl<<endl;
            }
            publish_obj_c.publish(send);
            // publish_obj_w.publish(obj_pos_w);

            std_msgs::Bool obj_found;
            if(found)
                obj_found.data = true;
            else
                obj_found.data = false;
            publish_found.publish(obj_found);

            offb::obj obj_depth_v;
            obj_depth_v.Z_c = state.at<float>(5);
            // cout<<endl<<obj_depth_v<<endl<<endl;;
            publish_obj_vel.publish(obj_depth_v);

            //cv::hconcat(tempp, res, display);
            //cv::line( res, cv::Point(320,240), center_pred, CV_RGB(100,0,255), 1, cv::LINE_AA);
            cv::Point text = cv::Point((320+center_pred.x)/2,(240+center_pred.y)/2);
            char temp_depth[40];
            sprintf(temp_depth, "%.2f", depth);
            string d = "distance: ";
            string textoutputonframe =d+temp_depth + " m";
            cv::putText(res, textoutputonframe, text,cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(180,140,120));
            // cv::imshow("Yolo", frame);
            // cv::imshow("Tracking...", res);
            // cv::waitKey(20);
            // video.write(res);
            // videoyolo.write(frame);


            cv::putText(gt, to_string(counter),cv::Point(20,20),cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0));
            // videogt.write(gt);
            counter++;
        }
        
        
        // if(!frame.empty())
        //     yolonet.detect_yolo(frame);


        ros::spinOnce();
        // rate_manager.sleep();
        
    }
    ros::spin();
    return 0;
}
