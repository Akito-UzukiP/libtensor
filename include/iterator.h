#ifndef ITERATOR_H
#define ITERATOR_H
#include "tensor_basic.h"


namespace ts{
// _Iterator 类的定义
template <typename T>
class Tensor<T>::_Iterator {
private:
    Tensor<T>* m_pTensor; // 指向张量的指针
    int _index; // 当前索引
    int* _indices; // 用于存储当前索引的向量，顺序与标准顺序不同，与_dim_order配套使用，在*运算符中重新排序为实际索引
    bool _done; // 是否完成
    int* _dim_order; // 维度顺序，用于计算索引，例如:[0,2,3,1]，就是先遍历第0维，然后第2维，第3维，第1维
    int _dim_order_size; // 维度顺序的长度
    bool* _dim_order_entered; // 用于跟踪是否进入了一个新的维度
public:
    _Iterator(){
        m_pTensor = nullptr;
        _index = 0;
        _indices = nullptr;
        _done = true;
        _dim_order = nullptr;
        _dim_order_size = 0;
        _dim_order_entered = nullptr;
    }


    _Iterator(Tensor<T>* pTensor, int* dim_order, int dim_order_size) {
        m_pTensor = pTensor;
        _index = 0;
        _indices = new int[m_pTensor->m_nDim];
        std::fill(_indices, _indices + m_pTensor->m_nDim, 0);
        _done = false;
        _dim_order = new int[dim_order_size];
        _dim_order_size = dim_order_size;
        _dim_order_entered = new bool[dim_order_size];
        for(int i = 0;i<dim_order_size;i++){
            _dim_order[i] = *(dim_order+i);
        }
    }
    _Iterator(Tensor<T>* pTensor){
        m_pTensor = pTensor;
        _index = 0;
        _indices = new int[m_pTensor->m_nDim];
        std::fill(_indices, _indices + m_pTensor->m_nDim, 0);
        _done = false;
        _dim_order = new int[m_pTensor->m_nDim];
        _dim_order_size = m_pTensor->m_nDim;
        _dim_order_entered = new bool[m_pTensor->m_nDim];
        for(int i = 0;i<m_pTensor->m_nDim;i++){
            _dim_order[i] = i;
        }
    }
    _Iterator(Tensor<T>* pTensor, std::initializer_list<int> dim_order){
        m_pTensor = pTensor;
        _index = 0;
        _indices = new int[m_pTensor->m_nDim];
        std::fill(_indices, _indices + m_pTensor->m_nDim, 0);
        _done = false;
        _dim_order = new int[dim_order.size()];
        _dim_order_size = dim_order.size();
        _dim_order_entered = new bool[dim_order.size()];
        for(int i = 0;i<dim_order.size();i++){
            _dim_order[i] = *(dim_order.begin()+i);
        }

    }
    _Iterator(const _Iterator& other){
        m_pTensor = other.m_pTensor;
        _index = other._index;

        _indices = new int[m_pTensor->m_nDim];
        for(int i = 0;i<m_pTensor->m_nDim;i++){
            _indices[i] = other._indices[i];
        }
        _done = other._done;

        _dim_order = new int[other._dim_order_size];
        _dim_order_size = other._dim_order_size;
        for(int i = 0;i<other._dim_order_size;i++){
            _dim_order[i] = other._dim_order[i];
        }

        _dim_order_entered = new bool[other._dim_order_size];
    }

    _Iterator& operator=(const _Iterator& other){
        if(this == &other){
            return *this;
        }
        m_pTensor = other.m_pTensor;
        _index = other._index;
        if(_indices != nullptr){
            delete[] _indices;
        }
        _indices = new int[m_pTensor->m_nDim];
        for(int i = 0;i<m_pTensor->m_nDim;i++){
            _indices[i] = other._indices[i];
        }
        _done = other._done;
        if(_dim_order != nullptr){
            delete[] _dim_order;
        }
        _dim_order = new int[other._dim_order_size];
        _dim_order_size = other._dim_order_size;
        for(int i = 0;i<other._dim_order_size;i++){
            _dim_order[i] = other._dim_order[i];
        }
        if(_dim_order_entered != nullptr){
            delete[] _dim_order_entered;
        }
        _dim_order_entered = new bool[other._dim_order_size];
        return *this;
    }

    ~_Iterator(){
        delete[] _indices;
        delete[] _dim_order;
        delete[] _dim_order_entered;
    }

    bool operator==(const _Iterator& other){
        return _index == other._index && m_pTensor == other.m_pTensor;
    }
    bool operator!=(const _Iterator& other){
        return !(operator==(other));
    }

    _Iterator& operator++(){
        _index++;
        // 按照_dim_order更新索引
        std::fill(_dim_order_entered, _dim_order_entered + m_pTensor->m_nDim, true);
        for(int dim = m_pTensor->m_nDim-1;dim>=0;dim--){
            if(_indices[dim] < m_pTensor->m_dims[_dim_order[dim]]-1){ // _indices的顺序与_dim_order相同，与标准顺序不同
                _indices[dim]++;
                std::fill(_dim_order_entered+dim+1,_dim_order_entered+m_pTensor->m_nDim,false);
                break;
            }else{
                if(dim == 0){
                    _done = true;
                }
                _indices[dim] = 0;
            }
        }
        return *this;
    }


    _Iterator operator++(int){
        _Iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    _Iterator next(){
        _Iterator tmp = *this;
        ++(*this);
        return tmp;
    }





    T& operator*(){
        int index = m_pTensor->m_start_index;
        for(int i = 0;i<m_pTensor->m_nDim;i++){
            index += _indices[i]*m_pTensor->m_strides[_dim_order[i]];
        }
        return m_pTensor->m_pData.get()[index];
    }

    bool done(){
        return _done;
    }

    bool hasNext(){
        return !done();
    }



};




// _Const_Iterator 类的定义
template <typename T>
class Tensor<T>::_Const_Iterator {
private:
    const Tensor<T>* m_pTensor; // 指向张量的指针
    int _index; // 当前索引
    int* _indices; // 用于存储当前索引的向量，顺序与标准顺序不同，与_dim_order配套使用，在*运算符中重新排序为实际索引
    bool _done; // 是否完成
    int* _dim_order; // 维度顺序，用于计算索引，例如:[0,2,3,1]，就是先遍历第0维，然后第2维，第3维，第1维
    int _dim_order_size; // 维度顺序的长度
    bool* _dim_order_entered; // 用于跟踪是否进入了一个新的维度
public:
    _Const_Iterator(){
        m_pTensor = nullptr;
        _index = 0;
        _indices = nullptr;
        _done = true;
        _dim_order = nullptr;
        _dim_order_size = 0;
        _dim_order_entered = nullptr;
    }
    _Const_Iterator(const Tensor<T>* pTensor, int* dim_order, int dim_order_size) {
        m_pTensor = pTensor;
        _index = 0;

        _indices = new int[m_pTensor->m_nDim];
        std::fill(_indices, _indices + m_pTensor->m_nDim, 0);
        _done = false;

        _dim_order = new int[dim_order_size];
        _dim_order_size = dim_order_size;

        _dim_order_entered = new bool[dim_order_size];
        for(int i = 0;i<dim_order_size;i++){
            _dim_order[i] = *(dim_order+i);
        }
    }
    _Const_Iterator(const Tensor<T>* pTensor){
        m_pTensor = pTensor;
        _index = 0;

        _indices = new int[m_pTensor->m_nDim];
        std::fill(_indices, _indices + m_pTensor->m_nDim, 0);
        _done = false;

        _dim_order = new int[m_pTensor->m_nDim];
        _dim_order_size = m_pTensor->m_nDim;

        _dim_order_entered = new bool[m_pTensor->m_nDim];
        for(int i = 0;i<m_pTensor->m_nDim;i++){
            _dim_order[i] = i;
        }
    }
    _Const_Iterator(const Tensor<T>* pTensor, std::initializer_list<int> dim_order){
        m_pTensor = pTensor;
        _index = 0;

        _indices = new int[m_pTensor->m_nDim];
        std::fill(_indices, _indices + m_pTensor->m_nDim, 0);
        _done = false;

        _dim_order = new int[dim_order.size()];
        _dim_order_size = dim_order.size();

        _dim_order_entered = new bool[dim_order.size()];
        for(int i = 0;i<dim_order.size();i++){
            _dim_order[i] = *(dim_order.begin()+i);
        }
    }
    _Const_Iterator(const _Const_Iterator& other){
        m_pTensor = other.m_pTensor;
        _index = other._index;

        _indices = new int[m_pTensor->m_nDim];
        for(int i = 0;i<m_pTensor->m_nDim;i++){
            _indices[i] = other._indices[i];
        }
        _done = other._done;

        _dim_order = new int[other._dim_order_size];
        _dim_order_size = other._dim_order_size;
        for(int i = 0;i<other._dim_order_size;i++){
            _dim_order[i] = other._dim_order[i];
        }

        _dim_order_entered = new bool[other._dim_order_size];
    }

    _Const_Iterator& operator=(const _Const_Iterator& other){
        if(this == &other){
            return *this;
        }
        m_pTensor = other.m_pTensor;
        _index = other._index;
        if(_indices != nullptr){
            delete[] _indices;
        }
        _indices = new int[m_pTensor->m_nDim];
        for(int i = 0;i<m_pTensor->m_nDim;i++){
            _indices[i] = other._indices[i];
        }
        _done = other._done;
        if(_dim_order != nullptr){
            delete[] _dim_order;
        }
        _dim_order = new int[other._dim_order_size];
        _dim_order_size = other._dim_order_size;
        for(int i = 0;i<other._dim_order_size;i++){
            _dim_order[i] = other._dim_order[i];
        }
        if(_dim_order_entered != nullptr){
            delete[] _dim_order_entered;
        }
        _dim_order_entered = new bool[other._dim_order_size];
        return *this;
    }

    ~_Const_Iterator(){
        delete[] _indices;
        delete[] _dim_order;
        delete[] _dim_order_entered;
    }

    bool operator==(const _Const_Iterator& other){
        return _index == other._index && m_pTensor == other.m_pTensor;
    }
    bool operator!=(const _Const_Iterator& other){
        return !(operator==(other));
    }

    _Const_Iterator& operator++(){
        _index++;
        // 按照_dim_order更新索引
        std::fill(_dim_order_entered, _dim_order_entered + m_pTensor->m_nDim, true);
        for(int dim = m_pTensor->m_nDim-1;dim>=0;dim--){
            if(_indices[dim] < m_pTensor->m_dims[_dim_order[dim]]-1){ // _indices的顺序与_dim_order相同，与标准顺序不同
                _indices[dim]++;
                std::fill(_dim_order_entered+dim+1,_dim_order_entered+m_pTensor->m_nDim,false);
                break;
            }else{
                if(dim == 0){
                    _done = true;
                }
                _indices[dim] = 0;
            }
        }
        return *this;
    }


    _Const_Iterator operator++(int){
        _Const_Iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    _Const_Iterator next(){
        _Const_Iterator tmp = *this;
        ++(*this);
        return tmp;
    }





    T& operator*(){
        int index = m_pTensor->m_start_index;
        for(int i = 0;i<m_pTensor->m_nDim;i++){
            index += _indices[i]*m_pTensor->m_strides[_dim_order[i]];
        }
        return m_pTensor->m_pData.get()[index];
    }

    bool done(){
        return _done;
    }

    bool hasNext(){
        return !done();
    }


};
}
#endif