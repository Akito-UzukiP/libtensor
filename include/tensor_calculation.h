#ifndef TS_TENSOR_CALCULATION_H
#define TS_TENSOR_CALCULATION_H
#include "tensor.h"
#include "iterator.h"
#include "tensor_operation.h"
#include <chrono>
namespace ts{

    template<typename U>
    Tensor<U> operator+(const Tensor<U>& lhs, const Tensor<U>& rhs){
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs,rhs});
        Tensor<U> lhs_ = broadcast(lhs, broadcast_dims);
        Tensor<U> rhs_ = broadcast(rhs, broadcast_dims);
        Tensor<U> res(lhs.shape());
        std::cout<<rhs<<std::endl;
        long lhs_index = 0;
        long rhs_index = 0;
        long res_index = 0;
        long* _indices = new long[lhs_.m_nDim];
        for(int i = 0;i<lhs_.m_nDim;i++){
            _indices[i] = 0;
        }
        // 按照_dim_order更新索引
        bool _done = false;
        while(!_done){
            res.data_ptr()[res_index] = lhs_.data_ptr()[lhs_index] + rhs_.data_ptr()[rhs_index];
            for(long dim = lhs_.m_nDim-1;dim>=0;dim--){
                if(_indices[dim] < lhs_.m_dims[dim]-1){ 
                    _indices[dim]++;
                    lhs_index += lhs_.m_strides[dim];
                    rhs_index += rhs_.m_strides[dim];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        _done = true;
                    }
                    lhs_index -= lhs_.m_strides[dim]*(_indices[dim]);
                    rhs_index -= rhs_.m_strides[dim]*(_indices[dim]);
                    res_index -= res.m_strides[dim]*(_indices[dim]);
                    _indices[dim] = 0;
                }
            }
        }

        return res;
    }
    template<typename U>
    Tensor<U> operator-(const Tensor<U>& lhs, const Tensor<U>& rhs){
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs,rhs});
        Tensor<U> lhs_ = broadcast(lhs, broadcast_dims);
        Tensor<U> rhs_ = broadcast(rhs, broadcast_dims);
        Tensor<U> res(lhs.shape());
        std::cout<<rhs<<std::endl;
        long lhs_index = 0;
        long rhs_index = 0;
        long res_index = 0;
        long* _indices = new long[lhs_.m_nDim];
        for(int i = 0;i<lhs_.m_nDim;i++){
            _indices[i] = 0;
        }
        // 按照_dim_order更新索引
        bool _done = false;
        while(!_done){
            res.data_ptr()[res_index] = lhs_.data_ptr()[lhs_index] - rhs_.data_ptr()[rhs_index];
            for(long dim = lhs_.m_nDim-1;dim>=0;dim--){
                if(_indices[dim] < lhs_.m_dims[dim]-1){ 
                    _indices[dim]++;
                    lhs_index += lhs_.m_strides[dim];
                    rhs_index += rhs_.m_strides[dim];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        _done = true;
                    }
                    lhs_index -= lhs_.m_strides[dim]*(_indices[dim]);
                    rhs_index -= rhs_.m_strides[dim]*(_indices[dim]);
                    res_index -= res.m_strides[dim]*(_indices[dim]);
                    _indices[dim] = 0;
                }
            }
        }

        return res;
    }

    template<typename U>
    Tensor<U> operator*(const Tensor<U>& lhs, const Tensor<U>& rhs){
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs,rhs});
        Tensor<U> lhs_ = broadcast(lhs, broadcast_dims);
        Tensor<U> rhs_ = broadcast(rhs, broadcast_dims);
        Tensor<U> res(lhs.shape());
        std::cout<<rhs<<std::endl;
        long lhs_index = 0;
        long rhs_index = 0;
        long res_index = 0;
        long* _indices = new long[lhs_.m_nDim];
        for(int i = 0;i<lhs_.m_nDim;i++){
            _indices[i] = 0;
        }
        // 按照_dim_order更新索引
        bool _done = false;
        while(!_done){
            res.data_ptr()[res_index] = lhs_.data_ptr()[lhs_index] * rhs_.data_ptr()[rhs_index];
            for(long dim = lhs_.m_nDim-1;dim>=0;dim--){
                if(_indices[dim] < lhs_.m_dims[dim]-1){ 
                    _indices[dim]++;
                    lhs_index += lhs_.m_strides[dim];
                    rhs_index += rhs_.m_strides[dim];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        _done = true;
                    }
                    lhs_index -= lhs_.m_strides[dim]*(_indices[dim]);
                    rhs_index -= rhs_.m_strides[dim]*(_indices[dim]);
                    res_index -= res.m_strides[dim]*(_indices[dim]);
                    _indices[dim] = 0;
                }
            }
        }

        return res;
    }


    template<typename U>
    Tensor<U> operator/(const Tensor<U>& lhs, const Tensor<U>& rhs){
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs,rhs});
        Tensor<U> lhs_ = broadcast(lhs, broadcast_dims);
        Tensor<U> rhs_ = broadcast(rhs, broadcast_dims);
        Tensor<U> res(lhs.shape());
        std::cout<<rhs<<std::endl;
        long lhs_index = 0;
        long rhs_index = 0;
        long res_index = 0;
        long* _indices = new long[lhs_.m_nDim];
        for(int i = 0;i<lhs_.m_nDim;i++){
            _indices[i] = 0;
        }
        // 按照_dim_order更新索引
        bool _done = false;
        while(!_done){
            res.data_ptr()[res_index] = lhs_.data_ptr()[lhs_index] / rhs_.data_ptr()[rhs_index];
            for(long dim = lhs_.m_nDim-1;dim>=0;dim--){
                if(_indices[dim] < lhs_.m_dims[dim]-1){ 
                    _indices[dim]++;
                    lhs_index += lhs_.m_strides[dim];
                    rhs_index += rhs_.m_strides[dim];
                    res_index += res.m_strides[dim];
                    break;
                }else{
                    if(dim == 0){
                        _done = true;
                    }
                    lhs_index -= lhs_.m_strides[dim]*(_indices[dim]);
                    rhs_index -= rhs_.m_strides[dim]*(_indices[dim]);
                    res_index -= res.m_strides[dim]*(_indices[dim]);
                    _indices[dim] = 0;
                }
            }
        }

        return res;
    }

    template<typename U>
    Tensor<U> log(const Tensor<U>& lhs){
        Tensor<U> res(lhs,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = std::log(*it_lhs);
            it_lhs++;
            it_res++;
        }

        return res;

    }


    template<typename T>
    Tensor<T> Tensor<T>::add(const Tensor<T>& t){
        return *this + t;
    }

    template <typename U>
    Tensor<U> add(const Tensor<U>& t1, const Tensor<U>& t2){
        return t1 + t2;
    }

    template<typename T>
    Tensor<T> Tensor<T>::sub(const Tensor<T>& t){
        return *this - t;
    }


    template <typename U>
    Tensor<U> sub(const Tensor<U>& t1, const Tensor<U>& t2){
        return t1 - t2;
    }

    template<typename T>
    Tensor<T> Tensor<T>::mul (const Tensor<T>& t){
        return *this * t;
    }

    template <typename U>
    Tensor<U> mul(const Tensor<U>& t1, const Tensor<U>& t2){
        return t1 * t2;
    }


    template<typename T>
    Tensor<T> Tensor<T>::div (const Tensor<T>& t){
        return *this / t;
    }

    template <typename U>
    Tensor<U> div(const Tensor<U>& t1, const Tensor<U>& t2){
        return t1 / t2;
    }


    template<typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor<T>& t){
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

        for(int i = 0; i < b; i++){
            for(int j = 0; j < m; j++){
                for(int l = 0; l < n; l++){
                    for(int p = 0; p < k; p++){
                        res(i,j,l) += lhs(i,j,p) * rhs(i,p,l);
                    }
                }
            }
        }

        return res.view(dims,nDim);
    }

    template<typename U>
    Tensor<U> matmul(const Tensor<U>& lhs, const Tensor<U>& rhs){
        if(lhs.m_nDim < 2 || rhs.m_nDim < 2){
            throw std::invalid_argument("Dimensions of lhs and rhs must be greater than 1.");
        }
        // for(int i = 0; i < lhs.m_nDim-2; i++){
        //     if(lhs.m_dims[i] != rhs.m_dims[i]){
        //         throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        //     }
        // }
        if(lhs.m_dims[lhs.m_nDim-1] != rhs.m_dims[rhs.m_nDim-2]){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }

        Tensor<U> lhs_ = transpose(lhs, lhs.m_nDim-2,lhs.m_nDim-1);
        Tensor<U> rhs_ = rhs;
        std::vector<int> broadcast_dims = get_broadcast_shape({lhs_,rhs_});
        lhs_ = broadcast(lhs_, broadcast_dims);
        rhs_ = broadcast(rhs_, broadcast_dims);
        lhs_ = transpose(lhs_, lhs_.m_nDim-2,lhs_.m_nDim-1);
       // rhs_ = transpose(rhs_, rhs_.m_nDim-2,rhs_.m_nDim-1);


        int m = lhs_.m_dims[lhs_.m_nDim-2];
        int n = rhs_.m_dims[rhs_.m_nDim-1];
        int k = lhs_.m_dims[lhs_.m_nDim-1];
        int b = 1;
        int* dims = new int[lhs_.m_nDim];
        int nDim = lhs_.m_nDim;
        for(int i = 0; i < lhs_.m_nDim-2; i++){
            dims[i] = lhs_.m_dims[i];
        }
        dims[lhs_.m_nDim-2] = m;
        dims[lhs_.m_nDim-1] = n;

        if(lhs_.m_nDim > 2){
            for(int i = 0; i < lhs_.m_nDim-2; i++){
                b *= lhs_.m_dims[i];
            }
        }
        // std::cout<<b<<" "<<m<<" "<<k<<" "<<n<<std::endl;
        // std::cout<< lhs_.size()<<" "<<rhs_.size()<<std::endl;
        lhs_ = lhs_.view({b,m,k});
        rhs_ = rhs_.view({b,k,n});
        Tensor<U> res = zeros<U>({b,m,n});

        for(int i = 0; i < b; i++){
            for(int j = 0; j < m; j++){
                for(int l = 0; l < n; l++){
                    for(int p = 0; p < k; p++){
                        res(i,j,l) += lhs_(i,j,p) * rhs_(i,p,l);
                    }
                }
            }
        }

        return res.view(dims,nDim);
    }

    template<typename U>
    Tensor<U> sum(const Tensor<U>& t, int dim){
        if(dim < 0 || dim >= t.m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims = new int[t.m_nDim-1];
        int nDim = t.m_nDim-1;
        int j = 0;
        for(int i = 0; i < t.m_nDim; i++){
            if(i != dim){
                dims[j] = t.m_dims[i];
                j++;
            }
        }
        Tensor<U> res(dims,nDim);
        int* dim_order = new int[t.m_nDim];
        int i = 0;
        dim_order[t.m_nDim-1] = dim;
        for(int j = 0; j < t.m_nDim; j++){
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        typename Tensor<U>::_Const_Iterator it(&t,dim_order,t.m_nDim);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it.done()){
            for(int i = 0;i<t.m_dims[dim];i++){
                *it_res += *it;
                it++;
            }
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<T> Tensor<T>::sum( int dim){
        if(dim < 0 || dim >= m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims = new int[m_nDim-1];
        int nDim = m_nDim-1;
        int j = 0;
        for(int i = 0; i < m_nDim; i++){
            if(i != dim){
                dims[j] = m_dims[i];
                j++;
            }
        }
        Tensor<T> res(dims,nDim);
        int* dim_order = new int[m_nDim];
        int i = 0;
        dim_order[m_nDim-1] = dim;
        for(int j = 0; j < m_nDim; j++){
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        typename Tensor<T>::_Const_Iterator it(this,dim_order,m_nDim);
        typename Tensor<T>::_Iterator it_res(&res);
        while(!it.done()){
            for(int i = 0;i<m_dims[dim];i++){
                *it_res += *it;
                it++;
            }
            it_res++;
        }

        return res;
    }

    template<typename U>
    Tensor<U> mean(const Tensor<U>& t, int dim){
        if(dim < 0 || dim >= t.m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims = new int[t.m_nDim-1];
        int nDim = t.m_nDim-1;
        int j = 0;
        for(int i = 0; i < t.m_nDim; i++){
            if(i != dim){
                dims[j] = t.m_dims[i];
                j++;
            }
        }
        Tensor<U> res(dims,nDim);
        int* dim_order = new int[t.m_nDim];
        int i = 0;
        dim_order[t.m_nDim-1] = dim;
        for(int j = 0; j < t.m_nDim; j++){
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        typename Tensor<U>::_Const_Iterator it(&t,dim_order,t.m_nDim);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it.done()){
            for(int i = 0;i<t.m_dims[dim];i++){
                *it_res += *it;
                it++;
            }
            *it_res /= t.m_dims[dim];
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<T> Tensor<T>::mean( int dim){
        if(dim < 0 || dim >= m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims = new int[m_nDim-1];
        int nDim = m_nDim-1;
        int j = 0;
        for(int i = 0; i < m_nDim; i++){
            if(i != dim){
                dims[j] = m_dims[i];
                j++;
            }
        }
        Tensor<T> res(dims,nDim);
        int* dim_order = new int[m_nDim];
        int i = 0;
        dim_order[m_nDim-1] = dim;
        for(int j = 0; j < m_nDim; j++){
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        typename Tensor<T>::_Const_Iterator it(this,dim_order,m_nDim);
        typename Tensor<T>::_Iterator it_res(&res);
        while(!it.done()){
            for(int i = 0;i<m_dims[dim];i++){
                *it_res += *it;
                it++;
            }
            *it_res /= m_dims[dim];
            it_res++;
        }

        return res;
    }

    template<typename U>
    Tensor<U> max(const Tensor<U>& t, int dim){
        if(dim < 0 || dim >= t.m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims = new int[t.m_nDim-1];
        int nDim = t.m_nDim-1;
        int j = 0;
        for(int i = 0; i < t.m_nDim; i++){
            if(i != dim){
                dims[j] = t.m_dims[i];
                j++;
            }
        }
        Tensor<U> res(dims,nDim);
        int* dim_order = new int[t.m_nDim];
        int i = 0;
        dim_order[t.m_nDim-1] = dim;
        for(int j = 0; j < t.m_nDim; j++){
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        typename Tensor<U>::_Const_Iterator it(&t,dim_order,t.m_nDim);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it.done()){
            for(int i = 0;i<t.m_dims[dim];i++){
                if(i == 0){
                    *it_res = *it;
                }else{
                    *it_res = std::max(*it_res,*it);
                }
                it++;
            }
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<T> Tensor<T>::max( int dim){
        if(dim < 0 || dim >= m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims = new int[m_nDim-1];
        int nDim = m_nDim-1;
        int j = 0;
        for(int i = 0; i < m_nDim; i++){
            if(i != dim){
                dims[j] = m_dims[i];
                j++;
            }
        }
        Tensor<T> res(dims,nDim);
        int* dim_order = new int[m_nDim];
        int i = 0;
        dim_order[m_nDim-1] = dim;
        for(int j = 0; j < m_nDim; j++){
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        typename Tensor<T>::_Const_Iterator it(this,dim_order,m_nDim);
        typename Tensor<T>::_Iterator it_res(&res);
        while(!it.done()){
            for(int i = 0;i<m_dims[dim];i++){
                if(i == 0){
                    *it_res = *it;
                }else{
                    *it_res = std::max(*it_res , *it);
                }
                it++;
            }
            it_res++;
        }

        return res;
    }

    template<typename U>
    Tensor<U> min(const Tensor<U>& t, int dim){
        if(dim < 0 || dim >= t.m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims = new int[t.m_nDim-1];
        int nDim = t.m_nDim-1;
        int j = 0;
        for(int i = 0; i < t.m_nDim; i++){
            if(i != dim){
                dims[j] = t.m_dims[i];
                j++;
            }
        }
        Tensor<U> res(dims,nDim);
        int* dim_order = new int[t.m_nDim];
        int i = 0;
        dim_order[t.m_nDim-1] = dim;
        for(int j = 0; j < t.m_nDim; j++){
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        typename Tensor<U>::_Const_Iterator it(&t,dim_order,t.m_nDim);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it.done()){
            for(int i = 0;i<t.m_dims[dim];i++){
                if(i == 0){
                    *it_res = *it;
                }else{
                    *it_res = std::min(*it_res,*it);
                }
                it++;
            }
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<T> Tensor<T>::min( int dim){
        if(dim < 0 || dim >= m_nDim){
            throw std::invalid_argument("Invalid dimension.");
        }
        int* dims = new int[m_nDim-1];
        int nDim = m_nDim-1;
        int j = 0;
        for(int i = 0; i < m_nDim; i++){
            if(i != dim){
                dims[j] = m_dims[i];
                j++;
            }
        }
        Tensor<T> res(dims,nDim);
        int* dim_order = new int[m_nDim];
        int i = 0;
        dim_order[m_nDim-1] = dim;
        for(int j = 0; j < m_nDim; j++){
            if(j != dim){
                dim_order[i] = j;
                i++;
            }
        }

        typename Tensor<T>::_Const_Iterator it(this,dim_order,m_nDim);
        typename Tensor<T>::_Iterator it_res(&res);
        while(!it.done()){
            for(int i = 0;i<m_dims[dim];i++){
                if(i == 0){
                    *it_res = *it;
                }else{
                    std::min(*it_res , *it);
                }
                it++;
            }
            it_res++;
        }

        return res;
    }

}
#endif