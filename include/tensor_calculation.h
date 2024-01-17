#ifndef TS_TENSOR_CALCULATION_H
#define TS_TENSOR_CALCULATION_H
#include "tensor_basic.h"
#include "tensor_operation.h"
#include <chrono>
namespace ts{


    template<typename U>
    Tensor<U> operator+(const Tensor<U>& lhs, const Tensor<U>& rhs){
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs,rhs});
        Tensor<U> lhs_ = broadcast(lhs, broadcast_dims);
        Tensor<U> rhs_ = broadcast(rhs, broadcast_dims);
        Tensor<U> res(lhs.shape());
        binary_elementwise_operation<U>(lhs, rhs, res, [](U a, U b){return a+b;});
        return res;
    }
    template<typename U>
    Tensor<U> operator-(const Tensor<U>& lhs, const Tensor<U>& rhs){
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs,rhs});
        Tensor<U> lhs_ = broadcast(lhs, broadcast_dims);
        Tensor<U> rhs_ = broadcast(rhs, broadcast_dims);
        Tensor<U> res(lhs.shape());
        binary_elementwise_operation<U>(lhs, rhs, res, [](U a, U b){return a-b;});
        return res;
    }

    template<typename U>
    Tensor<U> operator*(const Tensor<U>& lhs, const Tensor<U>& rhs){
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs,rhs});
        Tensor<U> lhs_ = broadcast(lhs, broadcast_dims);
        Tensor<U> rhs_ = broadcast(rhs, broadcast_dims);
        Tensor<U> res(lhs.shape());
        binary_elementwise_operation<U>(lhs, rhs, res, [](U a, U b){return a*b;});
        return res;
    }

    template<typename U>
    Tensor<U> operator/(const Tensor<U>& lhs, const Tensor<U>& rhs){
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs,rhs});
        Tensor<U> lhs_ = broadcast(lhs, broadcast_dims);
        Tensor<U> rhs_ = broadcast(rhs, broadcast_dims);
        Tensor<U> res(lhs.shape());
        binary_elementwise_operation<U>(lhs, rhs, res, [](U a, U b){return a/b;});
        return res;
    }

    template<typename U, typename V>
    Tensor<U> operator+(const Tensor<U>& lhs, const V& rhs){
        Tensor<U> res(lhs.shape());
        unary_elementwise_operation(lhs, res, [rhs](U a){return a+rhs;});
        return res;
    }

    template<typename U, typename V>
    Tensor<U> operator-(const Tensor<U>& lhs, const V& rhs){
        Tensor<U> res(lhs.shape());
        unary_elementwise_operation(lhs, res, [rhs](U a){return a-rhs;});
        return res;
    }

    template<typename U, typename V>
    Tensor<U> operator*(const Tensor<U>& lhs, const V& rhs){
        Tensor<U> res(lhs.shape());
        unary_elementwise_operation(lhs, res, [rhs](U a){return a*rhs;});
        return res;
    }

    template<typename U, typename V>
    Tensor<U> operator/(const Tensor<U>& lhs, const V& rhs){
        Tensor<U> res(lhs.shape());
        unary_elementwise_operation(lhs, res, [rhs](U a){return a/rhs;});
        return res;
    }




    template<typename U>
    Tensor<U> log(const Tensor<U>& t){
        Tensor<U> res(t,false);
        long t_index = 0;
        long res_index = 0;
        long* _indices = new long[t.m_nDim];
        bool done = false;
        for(int i = 0;i<t.m_nDim;i++){
            _indices[i] = 0;
        }
        while(!done){
            res.data_ptr()[res_index] = log(t.data_ptr()[t_index]);
            for(long dim = t.m_nDim-1;dim>=0;dim--){
                if(_indices[dim] < t.m_dims[dim]-1){ 
                    _indices[dim]++;
                    t_index += t.m_strides[dim];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        done = true;
                    }
                    t_index -= t.m_strides[dim]*(_indices[dim]);
                    res_index -= res.m_strides[dim]*(_indices[dim]);
                    _indices[dim] = 0;
                }
            }
        }
        delete[] _indices;
        return res;

    }


    template<typename T>
    Tensor<T> Tensor<T>::add(const Tensor<T>& t) const{
        return *this + t;
    }

    template <typename U>
    Tensor<U> add(const Tensor<U>& t1, const Tensor<U>& t2){
        return t1 + t2;
    }

    template<typename T>
    Tensor<T> Tensor<T>::sub(const Tensor<T>& t) const{
        return *this - t;
    }


    template <typename U>
    Tensor<U> sub(const Tensor<U>& t1, const Tensor<U>& t2){
        return t1 - t2;
    }

    template<typename T>
    Tensor<T> Tensor<T>::mul (const Tensor<T>& t) const{
        return *this * t;
    }

    template <typename U>
    Tensor<U> mul(const Tensor<U>& t1, const Tensor<U>& t2){
        return t1 * t2;
    }


    template<typename T>
    Tensor<T> Tensor<T>::div (const Tensor<T>& t) const{
        return *this / t;
    }

    template <typename U>
    Tensor<U> div(const Tensor<U>& t1, const Tensor<U>& t2){
        return t1 / t2;
    }

    template<typename T>
    template<typename V>
    Tensor<T> Tensor<T>::add(const V& v) const{
        return *this + v;
    }

    template<typename T, typename V>
    Tensor<T> add(const Tensor<T>& t, const V& v){
        return t + v;
    }

    template<typename T>
    template<typename V>
    Tensor<T> Tensor<T>::sub(const V& v) const{
        return *this - v;
    }

    template<typename T, typename V>
    Tensor<T> sub(const Tensor<T>& t, const V& v){
        return t - v;
    }

    template<typename T>
    template<typename V>
    Tensor<T> Tensor<T>::mul(const V& v) const{
        return *this * v;
    }

    template<typename T, typename V>
    Tensor<T> mul(const Tensor<T>& t, const V& v){
        return t * v;
    }

    template<typename T>
    template<typename V>
    Tensor<T> Tensor<T>::div(const V& v) const{
        return *this / v;
    }

    template<typename T, typename V>
    Tensor<T> div(const Tensor<T>& t, const V& v){
        return t / v;
    }





    template<typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor<T>& t) const{
        if(m_nDim < 2 || t.m_nDim < 2){
            throw std::invalid_argument("Dimensions of lhs and rhs must be greater than 1.");
        }
        for(int i = 0; i < m_nDim-2; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        if(m_dims[m_nDim-1] != t.m_dims[t.m_nDim-2]){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        int m = m_dims[m_nDim-2];
        int n = t.m_dims[t.m_nDim-1];
        int k = m_dims[m_nDim-1];
        int b = 1;
        int* dims = new int[m_nDim];
        int nDim = m_nDim;
        for(int i = 0; i < m_nDim-2; i++){
            dims[i] = m_dims[i];
        }
        dims[m_nDim-2] = m;
        dims[m_nDim-1] = n;

        if(m_nDim > 2){
            for(int i = 0; i < m_nDim-2; i++){
                b *= m_dims[i];
            }
        }
        Tensor<T> lhs = this->view({b,m,k});
        Tensor<T> rhs = t.view({b,k,n});
        Tensor<T> res = zeros<T>({b,m,n});
        long lhs_index = lhs.m_start_index, rhs_index = rhs.m_start_index, res_index = res.m_start_index;


        for(int i = 0; i < b; i++){
            for(int j = 0; j < m; j++){
                for(int l = 0; l < n; l++){
                    for(int p = 0; p < k; p++){
                        res.data_ptr()[res_index] += lhs.data_ptr()[lhs_index] * rhs.data_ptr()[rhs_index];
                        lhs_index += lhs.m_strides[2];
                        rhs_index += rhs.m_strides[1];
                    }
                    lhs_index -= lhs.m_strides[2] * k;
                    rhs_index -= rhs.m_strides[1] * k;
                    res_index += res.m_strides[2];
                    rhs_index += rhs.m_strides[2];
                }
                res_index -= res.m_strides[2] * n;
                rhs_index -= rhs.m_strides[2] * n;
                res_index += res.m_strides[1];
                lhs_index += lhs.m_strides[1];
            }
            res_index -= res.m_strides[1] * m;
            lhs_index -= lhs.m_strides[1] * m;
            lhs_index += lhs.m_strides[0];
            rhs_index += rhs.m_strides[0];
            res_index += res.m_strides[0];
        }
        
        return res.view(dims,nDim);
    }

    template<typename U>
    Tensor<U> matmul(const Tensor<U>& lhs, const Tensor<U>& rhs){
        return lhs.matmul(rhs);
    }

    template<typename T>
    Tensor<T> Tensor<T>::sum( int dim) const{
        if(dim < 0 || dim >= m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims;
        Tensor<T> res;
        if( m_nDim == 1){
            dims = new int[1];
            dims[0] = 1;
            res = zeros<T>(dims,m_nDim);
        }else{
            dims = new int[m_nDim-1];
            int nDim = m_nDim-1;
            int j = 0;
            for(int i = 0; i < m_nDim; i++){ // 如果dim=3，Tensor是5维度，则将0124维度赋予dims
                if(i != dim){
                    dims[j] = m_dims[i];
                    j++;
                }
            }
            res = zeros<T>(dims,nDim);
        }
        int* dim_order = new int[m_nDim];
        int i = 0;
        dim_order[m_nDim-1] = dim;
        for(int j = 0; j < m_nDim; j++){ // 0,1,2,4,3
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        long t_index = m_start_index;
        long res_index = res.m_start_index;
        long* res_indices = new long[res.m_nDim];
        bool done = false;
        for(int i = 0;i<res.m_nDim;i++){
            res_indices[i] = 0;
        }

        while(!done){ 

            // 在第一个for循环对压缩掉的那个维度进行操作
            for(long i = 0;i<m_dims[dim_order[m_nDim-1]];i++){
                res.data_ptr()[res_index] += data_ptr()[t_index];
                t_index += m_strides[dim_order[m_nDim-1]];
            }

            // 更新坐标的同时根据stride即刻更新index
            t_index -= m_strides[dim_order[m_nDim-1]]*m_dims[dim_order[m_nDim-1]];

            for(long dim = res.m_nDim-1;dim>=0;dim--){ // dim: 3 2 1 0 
                if(res_indices[dim] < res.m_dims[dim]-1){ 
                    res_indices[dim]++;
                    t_index += this->m_strides[dim_order[dim]];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        done = true;
                    }
                    t_index -= this->m_strides[dim_order[dim]]*(res_indices[dim]);
                    res_index -= res.m_strides[dim]*(res_indices[dim]);
                    res_indices[dim] = 0;
                }
            }
        }

        delete[] dims;
        delete[] dim_order;
        delete[] res_indices;
        return res;
    }

    template<typename U>
    Tensor<U> sum(const Tensor<U>& t, int dim){
        return t.sum(dim);
    }
    template<typename T>
    Tensor<T> Tensor<T>::mean( int dim) const{
        if(dim < 0 || dim >= m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims;
        Tensor<T> res;
        if( m_nDim == 1){
            dims = new int[1];
            dims[0] = 1;
            res = zeros<T>(dims,m_nDim);
        }else{
            dims = new int[m_nDim-1];
            int nDim = m_nDim-1;
            int j = 0;
            for(int i = 0; i < m_nDim; i++){ // 如果dim=3，Tensor是5维度，则将0124维度赋予dims
                if(i != dim){
                    dims[j] = m_dims[i];
                    j++;
                }
            }
            res = zeros<T>(dims,nDim);
        }
        int* dim_order = new int[m_nDim];
        int i = 0;
        dim_order[m_nDim-1] = dim;
        for(int j = 0; j < m_nDim; j++){ // 0,1,2,4,3
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        long t_index = m_start_index;
        long res_index = res.m_start_index;
        long* res_indices = new long[res.m_nDim];
        bool done = false;
        for(int i = 0;i<res.m_nDim;i++){
            res_indices[i] = 0;
        }

        while(!done){ 

            // 在第一个for循环对压缩掉的那个维度进行操作
            for(long i = 0;i<m_dims[dim_order[m_nDim-1]];i++){
                res.data_ptr()[res_index] += data_ptr()[t_index];
                t_index += m_strides[dim_order[m_nDim-1]];
            }
            res.data_ptr()[res_index] /= m_dims[dim];
 
            // 更新坐标的同时根据stride即刻更新index
            t_index -= m_strides[dim_order[m_nDim-1]]*m_dims[dim_order[m_nDim-1]];

            for(long dim = res.m_nDim-1;dim>=0;dim--){ // dim: 3 2 1 0 
                if(res_indices[dim] < res.m_dims[dim]-1){ 
                    res_indices[dim]++;
                    t_index += this->m_strides[dim_order[dim]];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        done = true;
                    }
                    t_index -= this->m_strides[dim_order[dim]]*(res_indices[dim]);
                    res_index -= res.m_strides[dim]*(res_indices[dim]);
                    res_indices[dim] = 0;
                }
            }
        }

        delete[] dims;
        delete[] dim_order;
        delete[] res_indices;
        return res;
    }


    template<typename U>
    Tensor<U> mean(const Tensor<U>& t, int dim){
        return t.mean(dim);
    }    

    template<typename T>
    Tensor<T> Tensor<T>::max( int dim) const{
        if(dim < 0 || dim >= m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims;
        Tensor<T> res;
        if( m_nDim == 1){
            dims = new int[1];
            dims[0] = 1;
            res = zeros<T>(dims,m_nDim);
        }else{
            dims = new int[m_nDim-1];
            int nDim = m_nDim-1;
            int j = 0;
            for(int i = 0; i < m_nDim; i++){ // 如果dim=3，Tensor是5维度，则将0124维度赋予dims
                if(i != dim){
                    dims[j] = m_dims[i];
                    j++;
                }
            }
            res = zeros<T>(dims,nDim);
        }
        int* dim_order = new int[m_nDim];
        int i = 0;
        dim_order[m_nDim-1] = dim;
        for(int j = 0; j < m_nDim; j++){ // 0,1,2,4,3
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        long t_index = m_start_index;
        long res_index = res.m_start_index;
        long* res_indices = new long[res.m_nDim];
        bool done = false;
        for(int i = 0;i<res.m_nDim;i++){
            res_indices[i] = 0;
        }

        while(!done){ 

            // 在第一个for循环对压缩掉的那个维度进行操作
            for(long i = 0;i<m_dims[dim_order[m_nDim-1]];i++){
                if( i == 0){
                    res.data_ptr()[res_index] = data_ptr()[t_index];
                }else{
                    res.data_ptr()[res_index] = std::max(res.data_ptr()[res_index], data_ptr()[t_index]);
                }
                t_index += m_strides[dim_order[m_nDim-1]];
            }
 
            // 更新坐标的同时根据stride即刻更新index
            t_index -= m_strides[dim_order[m_nDim-1]]*m_dims[dim_order[m_nDim-1]];

            for(long dim = res.m_nDim-1;dim>=0;dim--){ // dim: 3 2 1 0 
                if(res_indices[dim] < res.m_dims[dim]-1){ 
                    res_indices[dim]++;
                    t_index += this->m_strides[dim_order[dim]];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        done = true;
                    }
                    t_index -= this->m_strides[dim_order[dim]]*(res_indices[dim]);
                    res_index -= res.m_strides[dim]*(res_indices[dim]);
                    res_indices[dim] = 0;
                }
            }
        }

        delete[] dims;
        delete[] dim_order;
        delete[] res_indices;
        return res;
    }

    template<typename U>
    Tensor<U> max(const Tensor<U>& t, int dim){
        return t.max(dim);
    }

    template<typename T>
    Tensor<T> Tensor<T>::min( int dim) const{
        if(dim < 0 || dim >= m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims;
        Tensor<T> res;
        if( m_nDim == 1){
            dims = new int[1];
            dims[0] = 1;
            res = zeros<T>(dims,m_nDim);
        }else{
            dims = new int[m_nDim-1];
            int nDim = m_nDim-1;
            int j = 0;
            for(int i = 0; i < m_nDim; i++){ // 如果dim=3，Tensor是5维度，则将0124维度赋予dims
                if(i != dim){
                    dims[j] = m_dims[i];
                    j++;
                }
            }
            res = zeros<T>(dims,nDim);
        }
        int* dim_order = new int[m_nDim];
        int i = 0;
        dim_order[m_nDim-1] = dim;
        for(int j = 0; j < m_nDim; j++){ // 0,1,2,4,3
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        long t_index = m_start_index;
        long res_index = res.m_start_index;
        long* res_indices = new long[res.m_nDim];
        bool done = false;
        for(int i = 0;i<res.m_nDim;i++){
            res_indices[i] = 0;
        }

        while(!done){ 

            // 在第一个for循环对压缩掉的那个维度进行操作
            for(long i = 0;i<m_dims[dim_order[m_nDim-1]];i++){
                if( i == 0){
                    res.data_ptr()[res_index] = data_ptr()[t_index];
                }else{
                    res.data_ptr()[res_index] = std::min(res.data_ptr()[res_index], data_ptr()[t_index]);
                }
                t_index += m_strides[dim_order[m_nDim-1]];
            }
 
            // 更新坐标的同时根据stride即刻更新index
            t_index -= m_strides[dim_order[m_nDim-1]]*m_dims[dim_order[m_nDim-1]];

            for(long dim = res.m_nDim-1;dim>=0;dim--){ // dim: 3 2 1 0 
                if(res_indices[dim] < res.m_dims[dim]-1){ 
                    res_indices[dim]++;
                    t_index += this->m_strides[dim_order[dim]];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        done = true;
                    }
                    t_index -= this->m_strides[dim_order[dim]]*(res_indices[dim]);
                    res_index -= res.m_strides[dim]*(res_indices[dim]);
                    res_indices[dim] = 0;
                }
            }
        }

        delete[] dims;
        delete[] dim_order;
        delete[] res_indices;
        return res;
    }

    template<typename U>
    Tensor<U> min(const Tensor<U>& t, int dim){
        return t.min(dim);
    }

}
#endif