#include <algorithm>
#include <numeric>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>

#include "benchmark/api.hpp"
#include "../utils/fixture.hpp"
#include "../utils/helper.hpp"
#include "../utils/xt_adaptor.hpp"

TEST_F(TensorTest, CreateWithInitData) {
    RANDOM_SHAPE(shape)

    const auto data_size = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<>());
    const auto data = new float[data_size];
    std::generate(data, data + data_size, [&]() { return float_data_dist(rng); });

    EXPECT_NO_THROW(
            {
                bm::create_with_data(shape, data);
            }
            bm::create_with_data(shape, data);
            bm::create_with_data(shape, data);
    );
    EXPECT_EQUIV(xtada::to_xarray(shape, data), bm::create_with_data(shape, data));

    delete[] data;
}

TEST_F(TensorTest, CreateRandomTensor) {
    RANDOM_SHAPE(shape)

    EXPECT_NO_THROW(bm::rand<int>(shape));
    EXPECT_NO_THROW(bm::rand<float>(shape));
    EXPECT_NO_THROW(bm::rand<bool>(shape));
}

TEST_F(TensorTest, CreateFilledTensor) {
    RANDOM_SHAPE(shape)

    EXPECT_EQUIV(xt::zeros<float>(shape), bm::zeros<float>(shape));
    EXPECT_EQUIV(xt::zeros<bool>(shape), bm::zeros<bool>(shape));
    EXPECT_EQUIV(xt::ones<bool>(shape), bm::ones<bool>(shape));
    EXPECT_EQUIV(xt::xarray<int>(shape, 3), bm::full<int>(shape, 3));
}

TEST_F(TensorTest, CreatePatternedTensor) {
    EXPECT_EQUIV(xt::eye<bool>({3, 3}), bm::eye<bool>(3, 3));
    EXPECT_EQUIV(xt::eye<float>({100, 100}), bm::eye<float>(100, 100));
    EXPECT_EQUIV(xt::eye<float>({3, 4}), bm::eye<float>(3, 4));
}
