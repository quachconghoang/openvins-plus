//
// Created by hoangqc on 18/03/2019.
//

#ifndef TEST_LIBTORCH_SUPERPOINT_H
#define TEST_LIBTORCH_SUPERPOINT_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

class Superpoint_tracker{

};

class Superpoint {

public:
//    int raw_W = 752;
//    int raw_H = 480;
    int W = 752; // 188
    int H = 480; // 120
    float scale = 1.f;
    const int cell = 8; //8
    const int border_remove = 4; //4
    float thres = 0.03f;
    int dist_thresh = 16; //8

    bool m_debug = false;
    bool m_use_cuda = true;
    torch::DeviceType m_device_type;

    Superpoint();
    ~Superpoint();

    void init(const cv::String & model_path, bool debug=false, bool use_cuda = true);
    void compute_NN(cv::Mat & img_gray);
    void getKeyPoints(std::vector<cv::KeyPoint> & kps, cv::Mat & desc);


private:
    std::shared_ptr<torch::jit::script::Module> module;
    std::vector<torch::jit::IValue> inputs;
    c10::intrusive_ptr<torch::ivalue::Tuple> outputs;

//    std::vector<cv::Point> m_pts_nms;
    at::Tensor m_desc;
    at::Tensor semi, coarse_desc;


};



#endif //TEST_LIBTORCH_SUPERPOINT_H
