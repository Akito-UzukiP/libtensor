#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

#include "benchmark/api.hpp"

namespace xtada {

    template<typename T>
    xt::xarray<T> to_xarray(const std::vector<size_t> &shape, const T *data) {
        size_t data_size = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<>());
        return xt::adapt(data, data_size, xt::no_ownership(), shape);
    }

    template<typename T>
    ts::Tensor<T> to_tensor(const xt::xarray<T> &xarray) {
        std::vector<size_t> shape(xarray.shape().begin(), xarray.shape().end());
        return bm::create_with_data(shape, xarray.data());
    }

    template<typename T, typename V>
    bool equals(const T &a, const V &b) {
        return a == b;
    }
    template<>
    bool equals(const double &a, const double &b);

    template<>
    bool equals(const double &a, const float &b);

    template<>
    bool equals(const float &a, const double &b);

    template<>
    bool equals(const float &a, const float &b);
    template<typename T>
    bool equiv(const xt::xarray<T> &expected, const ts::Tensor<T> &actual) {
        auto shape = expected.shape();
        auto size = expected.size();
        for (size_t i = 0; i < size; i++) {
            std::vector<size_t> index;
            size_t tmp = i, step = size;
            for (int j = 0; j < shape.size(); j++) {
                step /= shape[j];
                index.push_back(tmp / step);
                tmp %= step;
            }
            // for test
            if (!equals(expected[index], bm::at(actual, index))) {
                spdlog::warn("#{} index: [{}], expected: {}, actual: {}",
                             i,
                             fmt::join(index, ", "),
                             expected[index],
                             bm::at(actual, index));
                return false;
            }
        }
        return true;
    }
}
