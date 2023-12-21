#ifndef TS_TENSOR_CALCULATION_H
#define TS_TENSOR_CALCULATION_H
#include "tensor.h"
namespace ts{

    template<typename U>
    Tensor<U> operator+(const Tensor<U>& lhs, const Tensor<U>& rhs){
        Tensor<U> res = lhs;
        std::vector<int> indices(res.m_nDim, 0);  // 用于存储当前索引的向量
        std::vector<bool> dimensionEntered(res.m_nDim, false); // 用于跟踪是否进入了一个新的维度
        bool done = false;

        while (!done) {
            // 遍历维度
            for (int dim = 0; dim < res.m_nDim; dim++) {
                if (!dimensionEntered[dim]) {
                    dimensionEntered[dim] = true;
                }
            }

            // 计算当前索引下的值
            int index = lhs.m_start_index;
            for (int i = 0; i < lhs.m_nDim; ++i) {
                index += indices[i] * res.m_strides[i];
            }
            
            res.m_pData.get()[index] = lhs.m_pData.get()[index] + rhs.m_pData.get()[index];

            // 更新索引并检查是否完成
            for (int dim = res.m_nDim - 1; dim >= 0; dim--) { // 从最内层往外更新，如果最内层到头了就更新上一层 ，break保证不会碰到未满的层的外层
                if (indices[dim] < res.m_dims[dim] - 1) { 
                    indices[dim]++;
                    std::fill(dimensionEntered.begin() + dim + 1, dimensionEntered.end(), false);
                    break;
                } else {
                    if (dim == 0) done = true; // 如果最外层都到头了就结束
                    indices[dim] = 0; // 如果没到头就把当前层的index置0，然后继续更新上一层
                }
            }
        }

        return res;

    }
    template<typename U>
    Tensor<U> operator-(const Tensor<U>& lhs, const Tensor<U>& rhs){

    }
    template<typename U>
    Tensor<U> operator*(const Tensor<U>& lhs, const Tensor<U>& rhs){

    }
    template<typename U>
    Tensor<U> operator/(const Tensor<U>& lhs, const Tensor<U>& rhs){

    }

}
#endif