#ifndef TS_TENSOR_CALCULATION_H
#define TS_TENSOR_CALCULATION_H
#include "tensor.h"
#include "iterator.h"
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


  

}
#endif