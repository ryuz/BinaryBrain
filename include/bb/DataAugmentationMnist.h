// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>
#include <random>
#include <opencv2/opencv.hpp>


namespace bb {

template <typename T = float>
void DataAugmentationMnist(std::vector< std::vector<T> >& vec_x, std::vector< std::vector<T> >& vec_y, int size=-1, std::uint64_t seed = 1)
{
    if (size < 0) {
        size = (int)vec_x.size();
    }

    std::mt19937_64 mt(seed);

    int w = 28;
    int h = 28;

    // マスク生成
    cv::Mat maskEdge(h, w, cv::DataType<T>::type);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            maskEdge.at<T>(i, j) = (i == 0 || i == (h - 1) || j == 0 || j == (w - 1)) ? (T)1 : (T)0;
        }
    }
    cv::Mat maskInner = maskEdge * -1.0 + 1.0;
    

    std::uniform_real_distribution<T> rand_scale((T)0.6, (T)1.0);
    std::uniform_real_distribution<T> rand_rot((T)(-3.1415926535/4), (T)(+3.1415926535 / 4));
    std::normal_distribution<T> rand_move_x(0, (T)(w / 10.0));
    std::normal_distribution<T> rand_move_y(0, (T)(h / 10.0));

    for (int i = 0; i < size; ++i) {
        int src_index = i % (int)vec_x.size();
        auto x = vec_x[src_index];
        auto y = vec_y[src_index];

        cv::Mat imgSrc(h, w, cv::DataType<T>::type);
        memcpy(imgSrc.data, &x[0], w * h * sizeof(T));

        cv::Mat matShift = cv::Mat::eye(3, 3, CV_32F);
        matShift.at<T>(0, 2) = (T)(w / 2.0);
        matShift.at<T>(1, 2) = (T)(h / 2.0);

        cv::Mat matScale = cv::Mat::eye(3, 3, CV_32F);
        matScale.at<T>(0, 0) = rand_scale(mt);
        matScale.at<T>(1, 1) = rand_scale(mt);

        cv::Mat matRotation = cv::Mat::eye(3, 3, CV_32F);
        T theta = rand_rot(mt);
        matRotation.at<T>(0, 0) = cos(theta);
        matRotation.at<T>(0, 1) = -sin(theta);
        matRotation.at<T>(1, 0) = -matScale.at<T>(0, 1);    // sin(theta);
        matRotation.at<T>(1, 1) = matScale.at<T>(0, 0);     // cos(theta);

        cv::Mat matMove = cv::Mat::eye(3, 3, CV_32F);
        matMove.at<T>(0, 2) = rand_move_x(mt);
        matMove.at<T>(1, 2) = rand_move_x(mt);

        cv::Mat mat = cv::Mat::eye(3, 3, CV_32F);
        mat = matMove * matShift * matScale * matRotation * matShift.inv() * mat;

        cv::Mat imgDst = cv::Mat::zeros(h, w, cv::DataType<T>::type);
        cv::warpAffine(imgSrc, imgDst, mat(cv::Rect(0, 0, 3, 2)), imgSrc.size(), CV_INTER_LINEAR, cv::BORDER_TRANSPARENT);


        double maxEdge, maxInner;
        cv::minMaxLoc(imgDst.mul(maskEdge), nullptr, &maxEdge);
        cv::minMaxLoc(imgDst.mul(maskInner), nullptr, &maxInner);
        if (maxEdge >= 0.5 || maxInner < 0.5) {
            std::fill(y.begin(), y.end(), 0.0f);
        }

        // ネガポジ
        if ( mt() % 2 == 1) {
            imgDst = imgDst * -1.0 + 1.0;
        }


        // 追加
        vec_x.push_back(x);
        vec_y.push_back(y);

//      cv::imshow("imgSrc", imgSrc);
//      cv::imshow("imgDst", imgDst);
//      cv::waitKey();
    }
}

}


