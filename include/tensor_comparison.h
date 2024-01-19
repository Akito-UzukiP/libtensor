#ifndef TENSOR_COMPARISON_H
#define TENSOR_COMPARISON_H
#include "tensor.h"
#include "iterator.h"
#include <type_traits>
namespace ts{
    // comparison gt ge lt le eq ne

    template<typename T>
    bool equal(T a, T b){
        return a == b;
    }
    template<>
    bool equal(float a, float b){
        return std::abs(a-b) < 1e-6;
    }
    template<>
    bool equal(double a, double b){
        return std::abs(a-b) < 1e-9;
    }
    template<>
    bool equal(long double a, long double b){
        return std::abs(a-b) < 1e-12;
    }

    template<typename T, typename U> // int and float
    bool equal(T a, U b){
        return std::abs(a-b) < 1e-6;
    }


    template<typename T>
    Tensor<bool> Tensor<T>::gt(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs > *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename U>
    Tensor<bool> gt(const Tensor<U>& lhs, const Tensor<U>& rhs){
        if(lhs.m_nDim != rhs.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < lhs.m_nDim; i++){
            if(lhs.m_dims[i] != rhs.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(lhs.m_dims,lhs.m_nDim);
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs > *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<bool> Tensor<T>::operator>(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs > *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::ge(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs >= *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename U>
    Tensor<bool> ge(const Tensor<U>& lhs, const Tensor<U>& rhs){
        if(lhs.m_nDim != rhs.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < lhs.m_nDim; i++){
            if(lhs.m_dims[i] != rhs.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(lhs.m_dims,lhs.m_nDim);
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs >= *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<bool> Tensor<T>::operator>=(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs >= *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::lt(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs < *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename U>
    Tensor<bool> lt(const Tensor<U>& lhs, const Tensor<U>& rhs){
        if(lhs.m_nDim != rhs.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < lhs.m_nDim; i++){
            if(lhs.m_dims[i] != rhs.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(lhs.m_dims,lhs.m_nDim);
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs < *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<bool> Tensor<T>::operator<(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs < *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::le(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs <= *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename U>
    Tensor<bool> le(const Tensor<U>& lhs, const Tensor<U>& rhs){
        if(lhs.m_nDim != rhs.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < lhs.m_nDim; i++){
            if(lhs.m_dims[i] != rhs.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(lhs.m_dims,lhs.m_nDim);
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs <= *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<bool> Tensor<T>::operator<=(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = *it_lhs <= *it_rhs;
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::eq(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = equal(*it_lhs,*it_rhs);
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename U>
    Tensor<bool> eq(const Tensor<U>& lhs, const Tensor<U>& rhs){
        if(lhs.m_nDim != rhs.m_nDim ){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < lhs.m_nDim; i++){
            if(lhs.m_dims[i] != rhs.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(lhs.m_dims,lhs.m_nDim);
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = equal(*it_lhs,*it_rhs);
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<bool> Tensor<T>::operator==(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = equal(*it_lhs,*it_rhs);
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }

    template<typename T>
    Tensor<bool> Tensor<T>::ne(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = !equal(*it_lhs,*it_rhs);
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename U>
    Tensor<bool> ne(const Tensor<U>& lhs, const Tensor<U>& rhs){
        if(lhs.m_nDim != rhs.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < lhs.m_nDim; i++){
            if(lhs.m_dims[i] != rhs.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(lhs.m_dims,lhs.m_nDim);
        typename Tensor<U>::_Const_Iterator it_lhs(&lhs);
        typename Tensor<U>::_Const_Iterator it_rhs(&rhs);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = !equal(*it_lhs,*it_rhs);
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
    template<typename T>
    Tensor<bool> Tensor<T>::operator!=(const Tensor<T>& t) const{
        if(m_nDim != t.m_nDim){
            throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
        }
        for(int i = 0; i < m_nDim; i++){
            if(m_dims[i] != t.m_dims[i]){
                throw std::invalid_argument("Dimensions of lhs and rhs do not match.");
            }
        }
        Tensor<bool> res(m_dims,m_nDim);
        typename Tensor<T>::_Const_Iterator it_lhs(this);
        typename Tensor<T>::_Const_Iterator it_rhs(&t);
        typename Tensor<bool>::_Iterator it_res(&res);
        while(!it_lhs.done()){
            *it_res = !equal(*it_lhs,*it_rhs);
            it_lhs++;
            it_rhs++;
            it_res++;
        }

        return res;
    }
}
#endif