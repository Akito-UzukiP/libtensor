#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace ts {
    template<typename T>
    class Tensor;

    // If your API does not match the above, you can use the following snippet to adapt your API to the above.
    // Specifically, you need to expose a class template named {@code Tensor} accepting one type parameter {@code T}.

    /*
    template<typename T>
    class TensorImpl;  // change this forward declaration to your implementation

    template<typename T>
    using Tensor = TensorImpl<T>;
    */
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-redundant-declaration"

namespace bm {

    /**
     * @brief Creates a tensor from a given array by copying data to its own memory (the tensor has no ownership of {@code data}).
     * @param shape shape of the tensor (size of each dimension)
     * @param data flatten data of the tensor (row major)
     * @note Please delegate a proper implemented {@code ts::Tensor} constructor to create a tensor
     */
    template<typename T>
    ts::Tensor<T> create_with_data(const std::vector<size_t> &shape, const T *data);

    template<typename T>
    ts::Tensor<T> rand(const std::vector<size_t> &shape);

    template<typename T>
    ts::Tensor<T> zeros(const std::vector<size_t> &shape);

    template<typename T>
    ts::Tensor<T> ones(const std::vector<size_t> &shape);

    template<typename T>
    ts::Tensor<T> full(const std::vector<size_t> &shape, const T &value);

    template<typename T>
    ts::Tensor<T> eye(size_t rows, size_t cols);

    template<typename T>
    ts::Tensor<T> concat(const std::vector<ts::Tensor<T>> &tensors, size_t axis);

    template<typename T>
    ts::Tensor<T> tile(const ts::Tensor<T> &tensor, const std::vector<size_t> &shape);

    template<typename T>
    ts::Tensor<T> transpose(const ts::Tensor<T> &tensor, size_t dim1, size_t dim2);

    template<typename T>
    ts::Tensor<T> permute(const ts::Tensor<T> &tensor, const std::vector<size_t> &permutation);

    /**
     * @brief Returns the element at the given position.
     * @param tensor tensor to access
     * @param indices indices of the element, guaranteed to be a vector of size {@code tensor.dim()}
     * @return reference to the element at the given position
     */
    template<typename T>
    T at(const ts::Tensor<T> &tensor, const std::vector<size_t> &indices);

    template<typename T>
    void set_at(ts::Tensor<T> &tensor, const std::vector<size_t> &indices, const T &value);

    template<typename T>
    ts::Tensor<T> pointwise_add(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<T> pointwise_sub(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<T> pointwise_mul(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<T> pointwise_div(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<T> pointwise_log(const ts::Tensor<T> &tensor);

    template<typename T>
    ts::Tensor<T> reduce_sum(const ts::Tensor<T> &tensor, size_t axis);

    template<typename T>
    ts::Tensor<T> reduce_mean(const ts::Tensor<T> &tensor, size_t axis);

    template<typename T>
    ts::Tensor<T> reduce_max(const ts::Tensor<T> &tensor, size_t axis);

    template<typename T>
    ts::Tensor<T> reduce_min(const ts::Tensor<T> &tensor, size_t axis);

    template<typename T>
    ts::Tensor<bool> eq(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<bool> ne(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<bool> gt(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<bool> ge(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<bool> lt(const ts::Tensor<T> &a, const ts::Tensor<T> &b);

    template<typename T>
    ts::Tensor<bool> le(const ts::Tensor<T> &a, const ts::Tensor<T> &b);
}

#pragma clang diagnostic pop

#include "api.tpp"
