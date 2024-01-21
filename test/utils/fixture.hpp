#pragma once

#include <cstdlib>
#include <random>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

class TensorTest : public ::testing::Test {
protected:
    std::mt19937_64 rng;

    std::uniform_int_distribution<size_t> dim_dist;
    std::uniform_int_distribution<size_t> shape_dist;

    std::uniform_int_distribution<int> int_data_dist;
    std::uniform_real_distribution<float> float_data_dist;
    std::bernoulli_distribution bool_data_dist;

    TensorTest() : rng(std::random_device()()),
                   dim_dist(1, 4),
                   shape_dist(1, 8),
                   float_data_dist(-100, 100),
                   int_data_dist(-100, 100),
                   bool_data_dist() {}

    void SetUp() override {
        if (std::getenv("LOG_DEBUG")) {
            spdlog::set_level(spdlog::level::debug);
        }
    }
};
