//
// Created by hoangqc on 10/08/2021.
//

#ifndef OPENVINS_PLUS_TRACKTORCH_H
#define OPENVINS_PLUS_TRACKTORCH_H

#include "TrackBase.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "Superpoint.h"

namespace ov_core{
    class TrackTorch : public TrackBase  {

    public:
        explicit TrackTorch(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool binocular,
                HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist)
        : TrackBase(cameras, numfeats, numaruco, binocular, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist) {
            printf("TrackTorch INIT!!! \n");
            cv::String modelPath = "/home/hoangqc/Datasets/Weights/superpoint_v1_752x480.pt";
            engine.init(modelPath, true, true);
            torch::NoGradGuard no_grad;
        }


        void feed_new_camera(const CameraData &message);

    protected:
        void feed_monocular(const CameraData &message, size_t msg_id);


        void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);


        void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                         std::vector<size_t> &ids0);

        void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                      const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                      std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);


        void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                              std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);


        // Timing variables
//        boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

        // Our descriptor matcher
        int threshold;
        int grid_x;
        int grid_y;
        int min_px_dist;

        // Descriptor matrices
        // How many pyramid levels to track
        int pyr_levels = 3;
        cv::Size win_size = cv::Size(20, 20);
        // Last set of image pyramids
        std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;

        Superpoint engine;
    };
}

#endif //OPENVINS_PLUS_TRACKTORCH_H
