#ifndef TS_TENSOR_CALCULATION_H
#define TS_TENSOR_CALCULATION_H
#include "tensor.h"
#include "iterator.h"
#include "tensor_operation.h"
#include <cmath>
namespace ts{

    template<typename U>
    Tensor<U> operator+(const Tensor<U>& lhs, const Tensor<U>& rhs){
        Tensor<U> res(lhs,false);
        bool done = false;

        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            std::cout<<*it_lhs<<std::endl;
            *it_res = *it_lhs + *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;

    }
    template<typename U>
    Tensor<U> operator-(const Tensor<U>& lhs, const Tensor<U>& rhs){
        // 确保两个张量具有相同的维度和大小
        if (lhs.m_nDim != rhs.m_nDim || lhs.m_dims != rhs.m_dims) {
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }

        Tensor<U> res(lhs,false);

        // 使用迭代器遍历 lhs 和 rhs
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<U>::_Iterator it_res(&res);

        while (!it_res.done()) {
            // 通过迭代器直接访问并操作张量的元素
            *it_res = *it_lhs - *it_rhs;

            // 递增迭代器
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template<typename U>
    Tensor<U> operator*(const Tensor<U>& lhs, const Tensor<U>& rhs){
        // 确保两个张量具有相同的维度和大小
        if (lhs.m_nDim != rhs.m_nDim || lhs.m_dims != rhs.m_dims) {
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }

        Tensor<U> res(lhs,false);

        // 使用迭代器遍历 lhs 和 rhs
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<U>::_Iterator it_res(&res);

        while (!it_res.done()) {
            // 通过迭代器直接访问并操作张量的元素
            *it_res = *it_lhs * *it_rhs;

            // 递增迭代器
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }


    template<typename U>
    Tensor<U> operator/(const Tensor<U>& lhs, const Tensor<U>& rhs){
        // 确保两个张量具有相同的维度和大小
        if (lhs.m_nDim != rhs.m_nDim || lhs.m_dims != rhs.m_dims) {
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }

        Tensor<U> res(lhs,false);

        // 使用迭代器遍历 lhs 和 rhs
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<U>::_Iterator it_res(&res);

        while (!it_res.done()) {
            // 通过迭代器直接访问并操作张量的元素
            *it_res = *it_lhs / *it_rhs;

            // 递增迭代器
            it_lhs++;
            it_rhs++;
            it_res++;
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


    template<typename U>
    Tensor<U> Tensor<U>::add(const Tensor<U>& t){
        Tensor<U> res(t,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&t);
        typename Tensor<U>::_Const_Iterator it_rhs(this);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs + *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template <typename U>
    Tensor<U> add(const Tensor<U>& t1, const Tensor<U>& t2){
        Tensor<U> res(t1,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&t1);
        typename Tensor<U>::_Const_Iterator it_rhs(&t2);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs + *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }
        

        return res;
    }

    template<typename U>
    Tensor<U> Tensor<U>::sub(const Tensor<U>& t){
        Tensor<U> res(t,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&t);
        typename Tensor<U>::_Const_Iterator it_rhs(this);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs - *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }


    template <typename U>
    Tensor<U> sub(const Tensor<U>& t1, const Tensor<U>& t2){
        Tensor<U> res(t1,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&t1);
        typename Tensor<U>::_Const_Iterator it_rhs(&t2);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs - *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }
        

        return res;
    }

    template<typename U>
    Tensor<U> Tensor<U>::mul (const Tensor<U>& t){
        Tensor<U> res(t,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&t);
        typename Tensor<U>::_Const_Iterator it_rhs(this);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs * *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template <typename U>
    Tensor<U> mul(const Tensor<U>& t1, const Tensor<U>& t2){
        Tensor<U> res(t1,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&t1);
        typename Tensor<U>::_Const_Iterator it_rhs(&t2);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs * *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }
        

        return res;
    }


    template<typename U>
    Tensor<U> Tensor<U>::div (const Tensor<U>& t){
        Tensor<U> res(t,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&t);
        typename Tensor<U>::_Const_Iterator it_rhs(this);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs / *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template <typename U>
    Tensor<U> div(const Tensor<U>& t1, const Tensor<U>& t2){
        Tensor<U> res(t1,false);
        typename Tensor<U>::_Const_Iterator it_lhs(&t1);
        typename Tensor<U>::_Const_Iterator it_rhs(&t2);
        typename Tensor<U>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs / *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }
        

        return res;
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
        for(int i = 0; i < lhs.m_nDim-2; i++){
            if(lhs.m_dims[i] != rhs.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        if(lhs.m_dims[lhs.m_nDim-1] != rhs.m_dims[rhs.m_nDim-2]){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        int m = lhs.m_dims[lhs.m_nDim-2];
        int n = rhs.m_dims[rhs.m_nDim-1];
        int k = lhs.m_dims[lhs.m_nDim-1];
        int b = 1;
        int* dims = new int[lhs.m_nDim];
        int nDim = lhs.m_nDim;
        for(int i = 0; i < lhs.m_nDim-2; i++){
            dims[i] = lhs.m_dims[i];
        }
        dims[lhs.m_nDim-2] = m;
        dims[lhs.m_nDim-1] = n;

        if(lhs.m_nDim > 2){
            for(int i = 0; i < lhs.m_nDim-2; i++){
                b *= lhs.m_dims[i];
            }
        }
        Tensor<U> lhs_ = lhs.view({b,m,k});
        Tensor<U> rhs_ = rhs.view({b,k,n});
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

  

}
#endif