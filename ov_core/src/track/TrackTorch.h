//
// Created by hoangqc on 10/08/2021.
//

#ifndef OPENVINS_PLUS_TRACKTORCH_H
#define OPENVINS_PLUS_TRACKTORCH_H

#include "TrackBase.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <cudnn.h>
#include <iostream>
#include <opencv2/features2d.hpp>

namespace ov_core{
    class TrackTorch : public TrackBase  {
    public:
        explicit TrackTorch(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras,
                            int numfeats, int numaruco, bool binocular,
                HistogramMethod histmethod, int fast_threshold,
                int gridx, int gridy, int minpxdist, double knnratio)
        : TrackBase(cameras, numfeats, numaruco, binocular, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist), knn_ratio(knnratio) {}
        void feed_new_camera(const CameraData &message);

    protected:
//        void feed_monocular();
        void feed_stereo();
//        void perform_detection_monocular();
        void perform_detection_stereo();

        // Timing variables
//        boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;
        // Our orb extractor
//        cv::Ptr<cv::ORB> orb0 = cv::ORB::create();
//        cv::Ptr<cv::ORB> orb1 = cv::ORB::create();

        // Our descriptor matcher
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        int threshold;
        int grid_x;
        int grid_y;
        int min_px_dist;
        double knn_ratio;

        // Descriptor matrices
        std::unordered_map<size_t, cv::Mat> desc_last;
    };
}

#endif //OPENVINS_PLUS_TRACKTORCH_H
