


#pragma once


#include "bb/DataType.h"


template<typename T=float>
bb::TrainData<T> LoadDiabetes(int num_train=400)
{
    const int n = 442;

    std::ifstream ifs_x("diabetes_data.txt");
    std::ifstream ifs_t("diabetes_target.txt");

    bb::TrainData<T> td;
    td.x_shape = bb::indices_t({ 10 });
    td.t_shape = bb::indices_t({ 1 });

    for (int i = 0; i < num_train; ++i) {
        std::vector<T> train(10);
        std::vector<T> target(1);
        for (int j = 0; j < 10; ++j) {
            ifs_x >> train[j];
        }
        ifs_t >> target[0];

        td.x_train.push_back(train);
        td.t_train.push_back(target);
    }

    for (int i = 0; i < n - num_train; ++i) {
        std::vector<T> train(10);
        std::vector<T> target(1);
        for (int j = 0; j < 10; ++j) {
            ifs_x >> train[j];
        }
        ifs_t >> target[0];

        td.x_test.push_back(train);
        td.t_test.push_back(target);
    }

    return td;
}


