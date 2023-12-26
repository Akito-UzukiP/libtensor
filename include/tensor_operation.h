#ifndef TENSOR_OPERATION_H
#define TENSOR_OPERATION_H
#include "tensor_basic.h"

namespace ts{
        // View操作群
    template <typename T>
    Tensor<T> Tensor<T>::view(const std::initializer_list<int>& dims){
        int total_size = 1;
        for(int i:dims){
            total_size *= i;
        }
        if(total_size != this->m_total_size){
            throw "View操作的目标维度大小与原张量不匹配";
        }
        Tensor<T> t = *this;
        t.m_nDim = dims.size();
        t.m_dims = new int[t.m_nDim];
        t.m_strides = new int[t.m_nDim];
        for(int i = 0;i<dims.size();i++){
            t.m_dims[i] = *(dims.begin()+i);
        }
        for(int i = dims.size()-1;i>=0;i--){
            if(i == dims.size()-1){
                t.m_strides[i] = 1;
            }else{
                t.m_strides[i] = t.m_strides[i+1]*t.m_dims[i+1];
            }
        }
        return t;
    }
    template <typename U>
    Tensor<U> view(const Tensor<U> org, const std::initializer_list<int>& dims){
        int total_size = 1;
        for(int i : dims){
            total_size *= i;
        }
        if(total_size != org.m_total_size){
            throw std::invalid_argument("View操作的目标维度大小与原张量不匹配");
        }
        Tensor<U> t = org;
        t.m_nDim = dims.size();
        t.m_dims = new int[t.m_nDim];
        t.m_strides = new int[t.m_nDim];
        for(int i = 0;i<dims.size();i++){
            t.m_dims[i] = *(dims.begin()+i);
        }
        for(int i = dims.size()-1;i>=0;i--){
            if(i == dims.size()-1){
                t.m_strides[i] = 1;
            }else{
                t.m_strides[i] = t.m_strides[i+1]*t.m_dims[i+1];
            }
        }
        return t;
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose(const int dim1, const int dim2) {
        if (dim1 >= m_nDim || dim2 >= m_nDim || dim1 < 0 || dim2 < 0) {
            throw std::invalid_argument("Transpose操作的维度超出张量维度");
        }

        Tensor<T> transposed_view = *this; // 创建原始张量的副本
        std::swap(transposed_view.m_strides[dim1], transposed_view.m_strides[dim2]);
        std::swap(transposed_view.m_dims[dim1], transposed_view.m_dims[dim2]);
        return transposed_view;
    }

    template <typename U>
    Tensor<U> transpose(const Tensor<U> org, const int dim1, const int dim2) {
        if (dim1 >= org.m_nDim || dim2 >= org.m_nDim || dim1 < 0 || dim2 < 0) {
            throw std::invalid_argument("Transpose操作的维度超出张量维度");
        }

        Tensor<U> transposed_view = org; // 创建原始张量的副本， shallow copy
        std::swap(transposed_view.m_strides[dim1], transposed_view.m_strides[dim2]);
        std::swap(transposed_view.m_dims[dim1], transposed_view.m_dims[dim2]);
        return transposed_view;
    }
    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::initializer_list<int> &dims) {
        if (dims.size() != m_nDim) {
            throw std::invalid_argument("Permute操作的维度数量与张量维度数量不匹配");
        }

        // 检查dims中的值是否合法且不重复
        std::vector<bool> dim_used(m_nDim, false);
        for (int dim : dims) {
            if (dim < 0 || dim >= m_nDim || dim_used[dim]) {
                throw std::invalid_argument("Permute操作中存在非法或重复维度");
            }
            dim_used[dim] = true;
        }

        Tensor<T> permuted_view = *this; // 创建原始张量的副本
        std::vector<int> new_dims(m_nDim);
        std::vector<int> new_strides(m_nDim);
        int i = 0;
        for (int dim : dims) {
            new_dims[i] = m_dims[dim];
            new_strides[i] = m_strides[dim];
            ++i;
        }
        for(int i = 0;i<m_nDim;i++){
            permuted_view.m_dims[i] = new_dims[i];
            permuted_view.m_strides[i] = new_strides[i];
        }
        return permuted_view;
    }

    template <typename U>
    Tensor<U> permute(const Tensor<U> org,const std::initializer_list<int> &dims){
        if (dims.size() != org.m_nDim) {
            throw std::invalid_argument("Permute操作的维度数量与张量维度数量不匹配");
        }

        // 检查dims中的值是否合法且不重复
        std::vector<bool> dim_used(org.m_nDim, false);
        for (int dim : dims) {
            if (dim < 0 || dim >= org.m_nDim || dim_used[dim]) {
                throw std::invalid_argument("Permute操作中存在非法或重复维度");
            }
            dim_used[dim] = true;
        }

        Tensor<U> permuted_view = org; // 创建原始张量的副本
        std::vector<int> new_dims(org.m_nDim);
        std::vector<int> new_strides(org.m_nDim);
        int i = 0;
        for (int dim : dims) {
            new_dims[i] = org.m_dims[dim];
            new_strides[i] = org.m_strides[dim];
            ++i;
        }
        for(int i = 0;i<org.m_nDim;i++){
            permuted_view.m_dims[i] = new_dims[i];
            permuted_view.m_strides[i] = new_strides[i];
        }
        return permuted_view;
    }


    // Mutate操作
    template <typename T>
    Tensor<T>& Tensor<T>::operator=(std::initializer_list<T> l){
        if(l.size() != m_total_size){
            throw std::invalid_argument("Mutate操作的目标维度大小与原张量不匹配");
        }

        std::vector<int> indices(m_nDim, 0); // 初始化索引向量
        auto it = l.begin(); // 初始化列表的迭代器
        bool done = false;

        while (!done) {
            // 计算当前索引下的一维索引
            int index = m_start_index;
            for (int i = 0; i < m_nDim; ++i) {
                index += indices[i] * m_strides[i];
            }

            // 使用迭代器值更新张量元素
            m_pData.get()[index] = *it;
            ++it;

            // 更新索引并检查是否完成
            for (int dim = m_nDim - 1; dim >= 0; dim--) {
                if (indices[dim] < m_dims[dim] - 1) {
                    indices[dim]++;
                    break;
                } else {
                    if (dim == 0) done = true;
                    indices[dim] = 0;
                }
            }
        }

        return *this;
    }



    // Slice操作
    template <typename T>
    Tensor<T> Tensor<T>::operator()(int dim_index) {
        if(dim_index >= m_nDim || dim_index < 0){
            throw std::invalid_argument("Slice操作的维度超出张量维度");
        }
        Tensor<T> sliced_view = *this;
        sliced_view.m_start_index += sliced_view.m_strides[0] * dim_index;
        sliced_view.m_nDim -= 1;
        sliced_view.m_dims = new int[sliced_view.m_nDim];
        sliced_view.m_strides = new int[sliced_view.m_nDim];
        sliced_view.m_total_size = 1;
        for(int i = 0; i < sliced_view.m_nDim; i++){
            sliced_view.m_dims[i] = m_dims[i + 1];
            sliced_view.m_strides[i] = m_strides[i + 1];
            sliced_view.m_total_size *= sliced_view.m_dims[i];
        }
        return sliced_view;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(int dim_index, std::initializer_list<int> indices){
        if(dim_index >= m_nDim || dim_index < 0){
            throw std::invalid_argument("Slice操作的维度超出张量维度");
        }
        Tensor<T> sliced_view = *this;
        sliced_view.m_start_index += sliced_view.m_strides[0] * dim_index + *(indices.begin()) * sliced_view.m_strides[1];
        sliced_view.m_nDim -= 1;
        sliced_view.m_total_size /= m_dims[0];
        sliced_view.m_dims = new int[sliced_view.m_nDim];
        sliced_view.m_strides = new int[sliced_view.m_nDim];
        for(int i = 0; i < sliced_view.m_nDim; i++){
            sliced_view.m_dims[i] = m_dims[i + 1];
            sliced_view.m_strides[i] = m_strides[i + 1];
        }
        sliced_view.m_total_size = sliced_view.m_total_size / m_dims[1] * (indices.end()-1 - indices.begin());
        sliced_view.m_dims[0] = indices.end()-1 - indices.begin();
        return sliced_view;
    }

    template <typename U>
    Tensor<U> concat(const Tensor<U> t1, const Tensor<U> t2, const int axis){
        /*
            思路：先做完合法性检查
                构建一个新的张量，然后把t1和t2的数据按照axis的位置拼接到新的张量上
        
        */
        if(t1.m_nDim != t2.m_nDim){
            throw std::invalid_argument("Concat操作的两个张量维度不匹配");
        }
        if(axis >= t1.m_nDim || axis < 0){
            throw std::invalid_argument("Concat操作的维度超出张量维度");
        }
        for(int i = 0;i<t1.m_nDim;i++){
            if(i != axis && t1.m_dims[i] != t2.m_dims[i]){
                throw std::invalid_argument("Concat操作的两个张量在非拼接维度上的维度不匹配");
            }
        }
        int* dim = new int[t1.m_nDim];
        for(int i = 0;i<t1.m_nDim;i++){
            dim[i] = t1.m_dims[i];
        }
        dim[axis] = t1.m_dims[axis] + t2.m_dims[axis];
        std::vector<int> dims(t1.m_nDim);
        for(int i = 0;i<t1.m_nDim;i++){
            dims[i] = dim[i];
        }
        Tensor<U> t = Tensor<U>(t1.m_pData.get(),dim,t1.m_nDim);


        // 把t1的数据赋值到t上
        std::vector<int> indices(t1.m_nDim, 0);  // 用于存储当前索引的向量
        std::vector<bool> dimensionEntered(t1.m_nDim, false); // 用于跟踪是否进入了一个新的维度
        bool done = false;
        while (!done) {
            // 遍历维度
            for (int dim = 0; dim < t1.m_nDim; dim++) {
                if (!dimensionEntered[dim]) {
                    dimensionEntered[dim] = true;
                }
            }

            // 计算当前索引下的值
            int index = t1.m_start_index;
            for (int i = 0; i < t1.m_nDim; ++i) {
                index += indices[i] * t1.m_strides[i];
            }
            int target_index = t.m_start_index;
            for (int i = 0; i < t.m_nDim; ++i) {
                target_index += indices[i] * t.m_strides[i];
            }
            t.m_pData.get()[target_index] =  t1.m_pData.get()[index];

            // 更新索引并检查是否完成
            for (int dim = t1.m_nDim - 1; dim >= 0; dim--) { // 从最内层往外更新，如果最内层到头了就更新上一层 ，break保证不会碰到未满的层的外层
                if (indices[dim] < t1.m_dims[dim] - 1) { 
                    indices[dim]++;
                    std::fill(dimensionEntered.begin() + dim + 1, dimensionEntered.end(), false);
                    break;
                } else {
                    if (dim == 0) done = true; // 如果最外层都到头了就结束
                    indices[dim] = 0; // 如果没到头就把当前层的index置0，然后继续更新上一层
                }
            }
        }
        // 把t1的数据赋值到t上
        std::fill(indices.begin(),indices.end(),0);
        std::fill(dimensionEntered.begin(),dimensionEntered.end(),false);
        done = false;
        while (!done) {
            // 遍历维度
            for (int dim = 0; dim < t2.m_nDim; dim++) {
                if (!dimensionEntered[dim]) {
                    dimensionEntered[dim] = true;
                }
            }

            // 计算当前索引下的值
            int index = t2.m_start_index;
            for (int i = 0; i < t2.m_nDim; ++i) {
                index += indices[i] * t2.m_strides[i];
            }
            int target_index = t.m_start_index;
            for (int i = 0; i < t.m_nDim; ++i) {
                if(i == axis){
                    target_index += (indices[i] + t1.m_dims[i]) * t.m_strides[i];
                }else{
                    target_index += indices[i] * t.m_strides[i];
                }
            }
            t.m_pData.get()[target_index] =  t2.m_pData.get()[index];

            // 更新索引并检查是否完成
            for (int dim = t2.m_nDim - 1; dim >= 0; dim--) { // 从最内层往外更新，如果最内层到头了就更新上一层 ，break保证不会碰到未满的层的外层
                if (indices[dim] < t2.m_dims[dim] - 1) { 
                    indices[dim]++;
                    std::fill(dimensionEntered.begin() + dim + 1, dimensionEntered.end(), false);
                    break;
                } else {
                    if (dim == 0) done = true; // 如果最外层都到头了就结束
                    indices[dim] = 0; // 如果没到头就把当前层的index置0，然后继续更新上一层
                }
            }
        }
        delete[] dim;

        return t;

    }
}

#endif