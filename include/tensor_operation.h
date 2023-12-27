#ifndef TENSOR_OPERATION_H
#define TENSOR_OPERATION_H
#include "tensor_basic.h"
#include "iterator.h"

namespace ts{
        // View操作群
    template <typename T>
    Tensor<T> Tensor<T>::view(const std::initializer_list<int>& dims) const{
        int total_size = 1;
        for(int i:dims){
            total_size *= i;
        }
        if(total_size != this->m_total_size){
            throw std::invalid_argument("View操作的目标维度大小与原张量不匹配");
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

    template <typename T>
    Tensor<T> Tensor<T>::view(const int* dims, const int nDim) const{
        int total_size = 1;
        for(int i = 0;i<nDim;i++){
            total_size *= dims[i];
        }
        if(total_size != this->m_total_size){
            throw std::invalid_argument("View操作的目标维度大小与原张量不匹配");
        }
        Tensor<T> t = *this;
        t.m_nDim = nDim;
        t.m_dims = new int[t.m_nDim];
        t.m_strides = new int[t.m_nDim];
        for(int i = 0;i<nDim;i++){
            t.m_dims[i] = dims[i];
        }
        for(int i = nDim-1;i>=0;i--){
            if(i == nDim-1){
                t.m_strides[i] = 1;
            }else{
                t.m_strides[i] = t.m_strides[i+1]*t.m_dims[i+1];
            }
        }
        return t;
    }



    template <typename U>
    Tensor<U> view(const Tensor<U>& org, const std::initializer_list<int>& dims){
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
    Tensor<U> transpose(const Tensor<U>& org, const int dim1, const int dim2) {
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
    Tensor<U> permute(const Tensor<U>& org,const std::initializer_list<int> &dims){
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
    Tensor<U> concat(const Tensor<U>& t1, const Tensor<U>& t2, const int axis){
        /*
            思路：先做完合法性检查
                构建一个新的张量，然后把t1和t2的数据按照axis的位置拼接到新的张量上
        
        */
        if(t1.m_nDim != t2.m_nDim){
            throw std::invalid_argument("Concat操作的两个张量维度数量不匹配");
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
            if(i == axis){
                dim[i] += t2.m_dims[i];
            }
        }
        Tensor<U> t = Tensor<U>(dim,t1.m_nDim);

        int* dim_order = new int[t1.m_nDim];
        int i = 0;
        for(int j = 0;j<t1.m_nDim;j++){
            if(j != axis){
                dim_order[i] = j;
                i++;
            }
        }
        dim_order[i] = axis;

        typename Tensor<U>::_Const_Iterator it1(&t1, dim_order, t1.m_nDim);
        typename Tensor<U>::_Const_Iterator it2(&t2, dim_order, t1.m_nDim);
        typename Tensor<U>::_Iterator it(&t, dim_order, t1.m_nDim);
        int index_of_target_axis = 0;
        while(it.hasNext()){
            if(index_of_target_axis < t1.m_dims[axis]){
                *it = *it1;
                ++it1;
            }else{
                *it = *it2;
                ++it2;
            }
            ++it;
            ++index_of_target_axis;
            if(index_of_target_axis == t.m_dims[axis]){
                index_of_target_axis = 0;
            }
        }
        return t;
    }


    template <typename T>
    Tensor<T> Tensor<T>::squeeze(const int dim) const{

        if(dim >= m_nDim || dim < 0){
            throw std::invalid_argument("Squeeze操作的维度超出张量维度");
        }
        if(m_dims[dim] != 1){
            throw std::invalid_argument("Squeeze操作的维度不为1");
        }
        Tensor<T> squeezed_view(*this);
        // 变化：m_dims少一个，m_nDim减一，m_strides少一个（其余strides不变）
        squeezed_view.m_nDim -= 1;
        int* temp_m_Dims = new int[squeezed_view.m_nDim];
        int* temp_m_Strides = new int[squeezed_view.m_nDim];
        int j = 0;
        for(int i = 0;i<m_nDim;i++){
            if(i != dim){
                temp_m_Dims[j] = m_dims[i];
                temp_m_Strides[j] = m_strides[i];
                j++;
            }
        }
        delete[] squeezed_view.m_dims;
        delete[] squeezed_view.m_strides;
        squeezed_view.m_dims = temp_m_Dims;
        squeezed_view.m_strides = temp_m_Strides;
        return squeezed_view;
    }
    template <typename U>
    Tensor<U> squeeze(const Tensor<U>& org, const int dim){

        if(dim >= org.m_nDim || dim < 0){
            throw std::invalid_argument("Squeeze操作的维度超出张量维度");
        }
        if(org.m_dims[dim] != 1){
            throw std::invalid_argument("Squeeze操作的维度不为1");
        }
        Tensor<U> squeezed_view(org,true);
        // 变化：m_dims少一个，m_nDim减一，m_strides少一个（其余strides不变）
        squeezed_view.m_nDim -= 1;
        int* temp_m_Dims = new int[squeezed_view.m_nDim];
        int* temp_m_Strides = new int[squeezed_view.m_nDim];
        int j = 0;
        for(int i = 0;i<org.m_nDim;i++){
            if(i != dim){
                temp_m_Dims[j] = org.m_dims[i];
                temp_m_Strides[j] = org.m_strides[i];
                j++;
            }
        }
        delete[] squeezed_view.m_dims;
        delete[] squeezed_view.m_strides;
        squeezed_view.m_dims = temp_m_Dims;
        squeezed_view.m_strides = temp_m_Strides;
        return squeezed_view;
    }

    // Unsqueeze操作
    template <typename T>
    Tensor<T> Tensor<T>::unsqueeze(const int dim) const{

        if(dim > m_nDim || dim < 0){
            throw std::invalid_argument("Unsqueeze操作的维度超出张量维度");
        }
        Tensor<T> unsqueezed_view(*this);
        // 变化：m_dims多一个，m_nDim加一，m_strides多一个（其余strides不变）
        unsqueezed_view.m_nDim += 1;
        int* temp_m_Dims = new int[unsqueezed_view.m_nDim];
        int* temp_m_Strides = new int[unsqueezed_view.m_nDim];
        int j = 0;
        for(int i = 0;i<unsqueezed_view.m_nDim;i++){
            if(i != dim){
                temp_m_Dims[i] = m_dims[j];
                temp_m_Strides[i] = m_strides[j];
                j++;
            }else{
                temp_m_Dims[i] = 1;
                temp_m_Strides[i] = 0;
            }
        }
        delete[] unsqueezed_view.m_dims;
        delete[] unsqueezed_view.m_strides;
        unsqueezed_view.m_dims = temp_m_Dims;
        unsqueezed_view.m_strides = temp_m_Strides;
        return unsqueezed_view;
    }
    template <typename U>
    Tensor<U> unsqueeze(const Tensor<U>& org, const int dim){
        if(dim > org.m_nDim || dim < 0){
            throw std::invalid_argument("Unsqueeze操作的维度超出张量维度");
        }
        Tensor<U> unsqueezed_view(org,true);
        // 变化：m_dims多一个，m_nDim加一，m_strides多一个（其余strides不变）
        unsqueezed_view.m_nDim += 1;
        int* temp_m_Dims = new int[unsqueezed_view.m_nDim];
        int* temp_m_Strides = new int[unsqueezed_view.m_nDim];
        int j = 0;
        for(int i = 0;i<unsqueezed_view.m_nDim;i++){
            if(i != dim){
                temp_m_Dims[i] = org.m_dims[j];
                temp_m_Strides[i] = org.m_strides[j];
                j++;
            }else{
                temp_m_Dims[i] = 1;
                temp_m_Strides[i] = 0;
            }
        }
        delete[] unsqueezed_view.m_dims;
        delete[] unsqueezed_view.m_strides;
        unsqueezed_view.m_dims = temp_m_Dims;
        unsqueezed_view.m_strides = temp_m_Strides;
        return unsqueezed_view;
    }

    template <typename U>
    Tensor<U> repeat_along_axis(const Tensor<U>& org, const int axis, const int count){
        if(axis >= org.m_nDim || axis < 0){
            throw std::invalid_argument("Repeat操作的维度超出张量维度");
        }
        if(count <= 0){
            throw std::invalid_argument("Repeat操作的重复次数小于等于0");
        }

        int *dim = new int[org.m_nDim];
        for(int i = 0;i<org.m_nDim;i++){
            dim[i] = org.m_dims[i];
            if(i == axis){
                dim[i] *= count;
            }
        }

        Tensor<U> t = Tensor<U>(dim,org.m_nDim);

        int* dim_order = new int[org.m_nDim];
        int i = 1;
        for(int j = 0;j<org.m_nDim;j++){
            if(j != axis){
                dim_order[i] = j;
                i++;
            }
        }
        dim_order[0] = axis;
        typename Tensor<U>::_Const_Iterator it1(&org, dim_order, org.m_nDim);
        typename Tensor<U>::_Iterator it(&t, dim_order, org.m_nDim);
        for(int i = 0;i<count;i++){
            while(it1.hasNext()){
                *it = *it1;
                ++it;
                ++it1;
            }
            it1.reset();
        }
        return t;
    }

    template <typename U>
    Tensor<U> tile(const Tensor<U>& org, const std::initializer_list<int>& counts){
        if(counts.size() > org.m_nDim+1 || counts.size() < org.m_nDim){
            throw std::invalid_argument("Tile操作的维度数量与张量维度数量不匹配");
        }
        Tensor<U> temp;
        if(counts.size() == org.m_nDim + 1){
            temp = org.unsqueeze(0);
        }else{
            temp = org;
        }

        for(int i = 0;i<counts.size();i++){
            if(*(counts.begin()+i) <= 0){
                throw std::invalid_argument("Tile操作的重复次数小于等于0");
            }
            temp = repeat_along_axis(temp,i,*(counts.begin()+i));
        }
        return temp;

    }
    template <typename U>
    Tensor<U> tile(const Tensor<U>& org, const int* counts, const int nCount){
        if(nCount > org.m_nDim+1 || nCount < org.m_nDim){
            throw std::invalid_argument("Tile操作的维度数量与张量维度数量不匹配");
        }
        Tensor<U> temp;
        if(nCount == org.m_nDim + 1){
            temp = org.unsqueeze(0);
        }else{
            temp = org;
        }

        for(int i = 0;i<nCount;i++){
            if(counts[i] <= 0){
                throw std::invalid_argument("Tile操作的重复次数小于等于0");
            }
            temp = repeat_along_axis(temp,i,counts[i]);
        }
        return temp;
    }
    
}

#endif