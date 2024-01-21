#include <vector>
#include <tuple>

#include <gtest/gtest.h>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>

#include "benchmark/api.hpp"
#include "../utils/fixture.hpp"
#include "../utils/helper.hpp"
#include "../utils/xt_adaptor.hpp"
#include <iostream>
TEST_F(TensorTest, Reduction) {
    for (int run = 0; run < 5; run++) {
        RANDOM_SHAPE(shape)
        auto axis = std::uniform_int_distribution<size_t>(0, shape_dim - 1)(rng);

        auto xarr = xt::eval(xt::random::randn<float>(shape));
        auto tensor = xtada::to_tensor(xarr);
        // std::cout<< xarr << std::endl;
        // std::cout<< tensor << std::endl;
        // std::cout<< xt::sum(xarr, axis) << std::endl;
        // std::cout<< bm::reduce_sum(tensor, axis) << std::endl;
        // std::cout<< bm::reduce_sum(tensor, axis).size() << std::endl;
        // std::cout<< bm::reduce_sum(tensor, axis).stride() << std::endl;
        // std::cout<< bm::reduce_sum(tensor, axis).data_ptr() << std::endl;
        // std::cout<< bm::reduce_sum(tensor, axis).total_size() << std::endl;
        // xtada::equiv(xt::eval(xt::sum(xarr, axis)), bm::reduce_sum(tensor, axis));
        EXPECT_EQUIV(xt::sum(xarr, axis), bm::reduce_sum(tensor, axis));
        EXPECT_EQUIV((xt::xarray<float>) xt::mean(xarr, axis), bm::reduce_mean(tensor, axis));
        EXPECT_EQUIV(xt::amax(xarr, axis), bm::reduce_max(tensor, axis));
        EXPECT_EQUIV(xt::amin(xarr, axis), bm::reduce_min(tensor, axis));
    }
}

#define TEST_BIN_OP(name, op, xtop) \
    TEST_F(TensorTest, BinOp_##name) { \
        for (int run = 0; run < 5; run++) { \
            RANDOM_SHAPE(shape) \
            auto xarr1 = xt::eval(xt::random::randn<float>(shape)); \
            auto xarr2 = xt::eval(xt::random::randn<float>(shape)); \
            auto tensor1 = xtada::to_tensor(xarr1); \
            auto tensor2 = xtada::to_tensor(xarr2); \
            EXPECT_EQUIV(xtop, bm::op(tensor1, tensor2)); \
        } \
    };

namespace bm {
    template<typename T>
    ts::Tensor<T> pointwise_log(const ts::Tensor<T> &tensor, const ts::Tensor<T> &_) {
        return bm::pointwise_log(tensor);
    }
}

TEST_BIN_OP(Add, pointwise_add, xarr1 + xarr2)
TEST_BIN_OP(Sub, pointwise_sub, xarr1 - xarr2)
TEST_BIN_OP(Mul, pointwise_mul, xarr1 * xarr2)
TEST_BIN_OP(Div, pointwise_div, xarr1 / xarr2)
TEST_BIN_OP(Log, pointwise_log, xt::log(xarr1))

TEST_BIN_OP(ComparisionEq, eq, xt::equal(xarr1, xarr2))
TEST_BIN_OP(ComparisionNe, ne, xt::not_equal(xarr1, xarr2))
TEST_BIN_OP(ComparisionGt, gt, xt::greater(xarr1, xarr2))
TEST_BIN_OP(ComparisionGe, ge, xt::greater_equal(xarr1, xarr2))
TEST_BIN_OP(ComparisionLt, lt, xt::less(xarr1, xarr2))
TEST_BIN_OP(ComparisionLe, le, xt::less_equal(xarr1, xarr2))
