#include "api.hpp"
// include your implementation's header file here, e.g.
#include "../../include/tensor.h"

namespace bm {

    inline std::vector<int> size_t2int(std::vector<size_t> shape) {
        std::vector<int> list;
        list.reserve(shape.size()); // 预分配所需内存以提高性能
        for (size_t val : shape) {
            list.push_back(static_cast<int>(val)); // 将 size_t 转换为 int 并添加到列表
        }
        return list;
    }

    template<typename T>
    ts::Tensor<T> create_with_data(const std::vector<size_t> &shape, const T *data) {
        // do necessary conversions and call your implementation, e.g.

        // int dim_ = static_cast<int>(shape.size());
        // int shape_[dim_];
        // for (int i = 0; i < dim_; i++) {
        //     shape_[i] = static_cast<int>(shape[i]);
        // }
        // return ts::Tensor<T>(dim_, shape_, data);

        // TODO
        std::vector<int> list = size_t2int(shape);
        return ts::Tensor<T>(data, list);
    }

    template<typename T>
    ts::Tensor<T> rand(const std::vector<size_t> &shape) {
        // TODO
        std::vector<int> list = size_t2int(shape);
        return ts::rand<T>(list);
    }

    template<typename T>
    ts::Tensor<T> zeros(const std::vector<size_t> &shape) {
        // TODO
        std::vector<int> list = size_t2int(shape);
        return ts::zeros<T>(list);
    }

    template<typename T>
    ts::Tensor<T> ones(const std::vector<size_t> &shape) {
        // TODO
        std::vector<int> list = size_t2int(shape);
        return ts::ones<T>(list);

    }

    template<typename T>
    ts::Tensor<T> full(const std::vector<size_t> &shape, const T &value) {
        // TODO
        std::vector<int> list = size_t2int(shape);
        return ts::full<T>(list, value);
    }

    template<typename T>
    ts::Tensor<T> eye(size_t rows, size_t cols) {
        // TODO
        return ts::eye<T>(rows, cols);

    }

    template<typename T>
    ts::Tensor<T> slice(const ts::Tensor<T> &tensor, const std::vector<std::pair<size_t, size_t>> &slices) {
        // TODO
        std::vector<std::pair<int,int>> slices3;
        std::pair<int,int> slices2;
        for (int i = 0; i < slices.size(); i++) {
            slices2.first = static_cast<int>(slices[i].first);
            slices2.second = static_cast<int>(slices[i].second);
            slices3.push_back(slices2);
        }
        return ts::slice<T>(tensor, slices3);
    }

    template<typename T>
    ts::Tensor<T> concat(const std::vector<ts::Tensor<T>> &tensors, size_t axis) {
        // TODO
        int axis2 = static_cast<int>(axis);
        ts::Tensor<T> tensor = ts::concat<T>(tensors, axis2);
        return tensor;

    }

    template<typename T>
    ts::Tensor<T> tile(const ts::Tensor<T> &tensor, const std::vector<size_t> &shape) {
        // TODO
        // friend Tensor<U> tile(const Tensor<U>& org, const std::vector<int>& counts);
        std::vector<int> list = size_t2int(shape);
        
        return ts::tile<T>(tensor, list);
    }

    template<typename T>
    ts::Tensor<T> transpose(const ts::Tensor<T> &tensor, size_t dim1, size_t dim2) {
        // TODO
        int dim1_1 = static_cast<int>(dim1);
        int dim2_2 = static_cast<int>(dim2);
        return ts::transpose<T>(tensor, dim1_1, dim2_2);
    }

    template<typename T>
    ts::Tensor<T> permute(const ts::Tensor<T> &tensor, const std::vector<size_t> &permutation) {
        // TODO
        std::vector<int> list = size_t2int(permutation);
        return ts::permute<T>(tensor, list);
    }

    template<typename T>
    T at(const ts::Tensor<T> &tensor, const std::vector<size_t> &indices) {
        // TODO
        return tensor(size_t2int(indices));
    }

    template<typename T>
    void set_at(ts::Tensor<T> &tensor, const std::vector<size_t> &indices, const T &value) {
        // TODO
        tensor(size_t2int(indices)) = value;
    }

    template<typename T>
    ts::Tensor<T> pointwise_add(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        // TODO
        return a + b;
    }

    template<typename T>
    ts::Tensor<T> pointwise_sub(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        // TODO
        return a - b;
    }

    template<typename T>
    ts::Tensor<T> pointwise_mul(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        // TODO
        return a * b;
    }

    template<typename T>
    ts::Tensor<T> pointwise_div(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        // TODO
        return a / b;
    }

    template<typename T>
    ts::Tensor<T> pointwise_log(const ts::Tensor<T> &tensor) {
        // TODO
        return ts::log<T>(tensor);
    }

    template<typename T>
    ts::Tensor<T> reduce_sum(const ts::Tensor<T> &tensor, size_t axis) {
        // TODO
        int int_axis = static_cast<int>(axis);
        return ts::sum<T>(tensor, int_axis);
    }

    template<typename T>
    ts::Tensor<T> reduce_mean(const ts::Tensor<T> &tensor, size_t axis) {
        // TODO
        int int_axis = static_cast<int>(axis);
        return ts::mean<T>(tensor, int_axis);

    }

    template<typename T>
    ts::Tensor<T> reduce_max(const ts::Tensor<T> &tensor, size_t axis) {
        // TODO
        int int_axis = static_cast<int>(axis);
        return ts::max<T>(tensor, int_axis);
    }

    template<typename T>
    ts::Tensor<T> reduce_min(const ts::Tensor<T> &tensor, size_t axis) {
        // TODO
        int int_axis = static_cast<int>(axis);
        return ts::min<T>(tensor, int_axis);
    }

    // you may modify the following functions' implementation if necessary

    template<typename T>
    ts::Tensor<bool> eq(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a == b;
    }

    template<typename T>
    ts::Tensor<bool> ne(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a != b;
    }

    template<typename T>
    ts::Tensor<bool> gt(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a > b;
    }

    template<typename T>
    ts::Tensor<bool> ge(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a >= b;
    }

    template<typename T>
    ts::Tensor<bool> lt(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a < b;
    }

    template<typename T>
    ts::Tensor<bool> le(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
        return a <= b;
    }
}
