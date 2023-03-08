// #include "/home/patty/ncnn/src/benchmark.h"
// #include "essential.h"
#include "/home/luo/lala/src/haha/include/ncnn/net.h"

#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>

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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include "run_yolo_only.h"

using namespace std;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob = 0; // confidence
    string classnameofdetection;
    cv::Mat frame;
    cv::Point center_bdbox;
    double depth;
};

static const char *class_names[] = {"null", "ball", "uav"};

class run_ncnn
{
    // cv::Mat depthdata;
    chrono::time_point <chrono::steady_clock> total_start, total_end, dnn_start, dnn_end;
    float total_fps;
    // Object obj;
    char* parampath = "";
    char* binpath   = "";
    int target_size = 0;
    
    ncnn::Net* cnn_local = new ncnn::Net();    

public:
    run_ncnn(char* param_input, char* bin_input, int target_size_input);
    ~run_ncnn();

    void detect_yolo(cv::Mat& bgr);
    void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects); //void display(cv::Mat frame);
    void getdepthdata(cv::Mat depthdata);
    float appro_fps;
    std::vector<Object> objects;
    cv::Mat depthdata;
};




