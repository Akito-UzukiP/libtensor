#ifndef TS_TENSOR_BASIC_H
#define TS_TENSOR_BASIC_H

#include <vector>
#include <string>
#include <typeinfo>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <typeinfo>
#include <random>
#include <memory>
#include "tensor_type_def.h"
namespace ts {
    template <typename T>
    class Tensor {
        private:
            int m_nDim; // Dimension count
            long m_total_size; // Total size of the tensor
            int*  m_dims; // Dimension sizes
            int m_start_index; // Start index
            //T *m_pData; // Element array,看起来得重写一个data类，提供一些基本的操作，或者干脆就用智能指针接手
            std::shared_ptr<T[]> m_pData;

            long* m_strides; // Strides,

        public:
            Tensor();  // Default constructor
            Tensor(const int* dims, const int nDim); // Constructor
            Tensor(const std::vector<int>& dims); // Constructor
            Tensor(T* pData, const std::initializer_list<int>& dims); // Constructor
            Tensor(T* pData, const int* dims, const int nDim); // Hard Copy Constructor
            Tensor(std::initializer_list<T> l, std::initializer_list<int> dims); // Constructor
            Tensor(const Tensor<T>& t, bool shallow_copy = true); // Shallow Copy constructor
            Tensor<T>& operator=(const Tensor<T>& t); // Copy assignment operator
            ~Tensor(); // Destructor
            
            std::string size() const; 
            std::vector<int> shape() const; 
            int total_size() const; 
            std::string type() const; 
            std::string stride() const; 
            T* data_ptr() const; 

           inline T& getPointerAtIndex(const int* dims); 

           // elementwise操作
           template <typename U, typename Func>
           friend void unary_elementwise_operation(const Tensor<U>& src, Tensor<U>& res, Func func);

           template <typename U, typename Func>
           friend void binary_elementwise_operation(const Tensor<U>& lhs, const Tensor<U>& rhs, Tensor<U>& res, Func func);

            //print方法

            template <typename U>
            friend std::ostream & operator<<(std::ostream & os, const Tensor<U> & m);

            // 重载运算符
            T& operator()(const std::initializer_list<int> indices) const;  
            T& operator()(const int* indices) const;  
            T& operator()(const std::vector<int> indices) const;
            T& operator()(const int i , const int j, const int k) const; 
            T& operator()(const int i , const int j) const; 
            Tensor<T> operator()(const int dim_index) const;  
            Tensor<T> operator()(const int dim_index, const std::initializer_list<int> indices) const; 

            // Contiguous操作 
            // 检测是否连续
            bool is_contiguous() const;
            // 返回内存连续化的版本
            Tensor<T> contiguous() const;
            // View操作，如果内存连续则共享原内存，若内存不连续则重排后新建内存
            Tensor<T> view(const std::initializer_list<int>& dims) const; 
            Tensor<T> view(const int* dims, const int nDim) const; 
            template <typename U>
            friend Tensor<U> view(const Tensor<U>& org, const std::initializer_list<int>& dims);

            // Transpose操作
            Tensor<T> transpose(const int dim1, const int dim2) const; 
            template <typename U>
            friend Tensor<U> transpose(const Tensor<U>& org, const int dim1, const int dim2); 
            
            // Permute操作
            Tensor<T> permute(const std::initializer_list<int>& dims) const; 
            template <typename U>
            friend Tensor<U> permute(const Tensor<U>& org, const std::initializer_list<int>& dims); 
            // Slice操作
            // TODO 在上面的operator()中实现

            // Concat操作
            // TODO
            template <typename U>
            friend Tensor<U> concat(const Tensor<U>& t1, const Tensor<U>& t2, const int axis); 

            // tile操作
            template <typename U>
            friend Tensor<U> repeat_along_axis(const Tensor<U>& org,const int axis,const int count);

            template <typename U>
            friend Tensor<U> tile(const Tensor<U>& org, const std::initializer_list<int>& counts); 
            template <typename U>
            friend Tensor<U> tile(const Tensor<U>& org, const int* counts, const int nCount);
            template <typename U>
            friend Tensor<U> tile(const Tensor<U>& org, const std::vector<int>& counts);

            // BroadCast操作
            template <typename U>
            friend std::vector<int> get_broadcast_shape(const std::vector<Tensor<U>>& tensors); 
            template <typename U>
            friend std::vector<int> get_broadcast_shape(const std::initializer_list<Tensor<U>>& tensors); 
            template <typename U>
            friend Tensor<U> broadcast(const Tensor<U>& org, const std::initializer_list<int>& dims); 
            template <typename U>
            friend Tensor<U> broadcast(const Tensor<U>& org, const int* dims, const int nDim); 
            template <typename U>
            friend Tensor<U> broadcast(const Tensor<U>& org, const std::vector<int>& dims); 

            // Mutate操作
            // TODO
            Tensor<T>& operator=(std::initializer_list<T> l); // 用列表中的元素替换张量中的元素


            // Squeeze和Unsqueeze操作
            Tensor<T> squeeze(const int dim) const; 
            template <typename U>
            friend Tensor<U> squeeze(const Tensor<U>& org, const int dim); 

            Tensor<T> unsqueeze(const int dim) const; 
            template <typename U>
            friend Tensor<U> unsqueeze(const Tensor<U>& org, const int dim); 


            template <typename U>
            friend Tensor<U> operator+(const Tensor<U>& t1, const Tensor<U>& t2); 
            template <typename U>
            friend Tensor<U> operator-(const Tensor<U>& t1, const Tensor<U>& t2); 
            template <typename U>
            friend Tensor<U> operator*(const Tensor<U>& t1, const Tensor<U>& t2); 
            template <typename U>
            friend Tensor<U> operator/(const Tensor<U>& t1, const Tensor<U>& t2); 
            template <typename U>
            friend Tensor<U> log(const Tensor<U>& lhs); 

            Tensor<T> add(const Tensor<T>& t) const; 
            template <typename U>
            friend Tensor<U> add(const Tensor<U>& t1, const Tensor<U>& t2); 
            Tensor<T> sub(const Tensor<T>& t) const; 
            template <typename U>
            friend Tensor<U> sub(const Tensor<U>& t1, const Tensor<U>& t2); 
            Tensor<T> mul(const Tensor<T>& t) const; 
            template <typename U>
            friend Tensor<U> mul(const Tensor<U>& t1, const Tensor<U>& t2); 
            Tensor<T> div(const Tensor<T>& t) const; 
            template <typename U>
            friend Tensor<U> div(const Tensor<U>& t1, const Tensor<U>& t2); 

            
            Tensor<T> matmul(const Tensor<T>& t) const; 
            template <typename U>
            friend Tensor<U> matmul(const Tensor<U>& t1, const Tensor<U>& t2); 

            // 基础的reduction计算
            Tensor<T> sum(const int dim);
            template <typename U>
            friend Tensor<U> sum(const Tensor<U>& t, const int dim);
            Tensor<T> mean(const int dim); 
            template <typename U>
            friend Tensor<U> mean(const Tensor<U>& t, const int dim); 
            Tensor<T> max(const int dim);
            template <typename U>
            friend Tensor<U> max(const Tensor<U>& t, const int dim);
            Tensor<T> min(const int dim);
            template <typename U>
            friend Tensor<U> min(const Tensor<U>& t, const int dim);
            //比较，有: gt ge lt le eq ne
            Tensor<bool> gt(const Tensor<T>& t);
            template <typename U>
            friend Tensor<bool> gt(const Tensor<U>& t1, const Tensor<U>& t2);
            Tensor<bool> operator>(const Tensor<T>& t); 
            Tensor<bool> ge(const Tensor<T>& t);
            template <typename U>
            friend Tensor<bool> ge(const Tensor<U>& t1, const Tensor<U>& t2);
            Tensor<bool> operator>=(const Tensor<T>& t); 
            Tensor<bool> lt(const Tensor<T>& t); 
            template <typename U>
            friend Tensor<bool> lt(const Tensor<U>& t1, const Tensor<U>& t2); 
            Tensor<bool> operator<(const Tensor<T>& t); 
            Tensor<bool> le(const Tensor<T>& t);
            template <typename U>
            friend Tensor<bool> le(const Tensor<U>& t1, const Tensor<U>& t2);
            Tensor<bool> operator<=(const Tensor<T>& t); 
            Tensor<bool> eq(const Tensor<T>& t); 
            template <typename U>
            friend Tensor<bool> eq(const Tensor<U>& t1, const Tensor<U>& t2); 
            Tensor<bool> operator==(const Tensor<T>& t); 
            Tensor<bool> ne(const Tensor<T>& t); 
            template <typename U>
            friend Tensor<bool> ne(const Tensor<U>& t1, const Tensor<U>& t2); 
            Tensor<bool> operator!=(const Tensor<T>& t); 

            void serialize(const std::string& filename) const;
            template <typename U>
            friend void serialize(const Tensor<U>& tensor, const std::string& filename);


            static Tensor<T> deserialize(const std::string& filename);


            class _Iterator;
            class _Const_Iterator;

    };

    // Default constructor
    template <typename T>
    Tensor<T>::Tensor() {
        m_nDim = 0;
        m_total_size = 0;
        m_dims = nullptr;
        m_pData = nullptr;
        m_strides = nullptr;
        m_start_index = 0;
        
    }

    template <typename T>
    Tensor<T>::Tensor(const int* dims, const int nDim){
        m_nDim = nDim;
        m_total_size = 1;
        m_dims = new int[m_nDim];
        m_strides = new long[m_nDim];
        m_start_index = 0;
        for(int i = 0;i<nDim;i++){
            m_dims[i] = dims[i];
            m_total_size *= m_dims[i];
        }
        for(int i = nDim-1;i>=0;i--){
            if(i == nDim-1){
                m_strides[i] = 1;
            }else{
                m_strides[i] = m_strides[i+1]*m_dims[i+1];
            }
        }
        m_pData = std::shared_ptr<T[]>(new T[m_total_size]);
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<int>& dims){
        m_nDim = dims.size();
        m_total_size = 1;
        m_dims = new int[m_nDim];
        m_strides = new long[m_nDim];
        m_start_index = 0;
        for(int i = 0;i<dims.size();i++){
            m_dims[i] = dims[i];
            m_total_size *= m_dims[i];
        }
        for(int i = dims.size()-1;i>=0;i--){
            if(i == dims.size()-1){
                m_strides[i] = 1;
            }else{
                m_strides[i] = m_strides[i+1]*m_dims[i+1];
            }
        }
        m_pData = std::shared_ptr<T[]>(new T[m_total_size]);
    }

    template <typename T>
    Tensor<T>::Tensor(T* pData, const std::initializer_list<int>& dims){
        m_nDim = dims.size();
        m_total_size = 1;
        m_dims = new int[m_nDim];
        m_strides = new long[m_nDim];
        m_start_index = 0;
        for(int i = 0;i<dims.size();i++){
            m_dims[i] = *(dims.begin()+i);
            m_total_size *= m_dims[i];
        }
        for(int i = dims.size()-1;i>=0;i--){
            if(i == dims.size()-1){
                m_strides[i] = 1;
            }else{
                m_strides[i] = m_strides[i+1]*m_dims[i+1];
            }
        }
        m_pData = std::shared_ptr<T[]>(new T[m_total_size]);
        for(int i = 0;i<m_total_size;i++){
            m_pData.get()[i] = pData[i];
        }
    }
    template <typename T>
    Tensor<T>::Tensor(T* pData, const int* dims, const int nDim){
        m_nDim = nDim;
        m_total_size = 1;
        m_dims = new int[m_nDim];
        m_strides = new long[m_nDim];
        m_start_index = 0;
        for(int i = 0;i<nDim;i++){
            m_dims[i] = dims[i];
            m_total_size *= m_dims[i];
        }
        for(int i = nDim-1;i>=0;i--){
            if(i == nDim-1){
                m_strides[i] = 1;
            }else{
                m_strides[i] = m_strides[i+1]*m_dims[i+1];
            }
        }
        m_pData = std::shared_ptr<T[]>(new T[m_total_size]);
        for(int i = 0;i<m_total_size;i++){
            m_pData.get()[i] = pData[i];
        }
    }

    template <typename T>
    Tensor<T>::Tensor(std::initializer_list<T> l, std::initializer_list<int> dims){
        Tensor<T> t = Tensor<T>(dims);
        T* data_ptr = t.m_pData.get();
        int i = 0;
        for(auto it = l.begin();it != l.end();it++){
            data_ptr[i] = *it;
            i++;
        }
    }


    // Shallow Copy constructor
    template <typename T> 
    Tensor<T>::Tensor(const Tensor<T>& other, bool shallow_copy){
        m_nDim = other.m_nDim;
        m_total_size = other.m_total_size;
        m_dims = new int[m_nDim];
        m_strides = new long[m_nDim];
        m_start_index = other.m_start_index;
        for(int i = 0;i<m_nDim;i++){
            m_dims[i] = other.m_dims[i];
            m_strides[i] = other.m_strides[i];
        }
        if(shallow_copy){
            m_pData = other.m_pData;
        }else{
            m_pData = std::shared_ptr<T[]>(new T[m_total_size]);
            for(int i = 0;i<m_total_size;i++){
                m_pData.get()[i] = other.m_pData.get()[i];
            }
        }
    }

    // Shallow Copy assignment operator
    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other){
        if(this == &other){
            return *this;
        }
        m_nDim = other.m_nDim;
        m_total_size = other.m_total_size;
        if(m_dims != nullptr){
            delete[] m_dims;
        }
        if(m_strides != nullptr){
            delete[] m_strides;
        }
        m_dims = new int[m_nDim];
        m_strides = new long[m_nDim];
        m_start_index = other.m_start_index;
        for(int i = 0;i<m_nDim;i++){
            m_dims[i] = other.m_dims[i];
            m_strides[i] = other.m_strides[i];
        }
        m_pData = other.m_pData;
        return *this;
    }

    // Destructor
    template <typename T>
    Tensor<T>::~Tensor() {
        delete[] m_dims;
        delete[] m_strides;
    }

    // Return the size of each dimension
    template <typename T>
    std::string Tensor<T>::size() const {
        std::string output = "[";
        for(int i = 0;i<m_nDim;i++){
            output += std::to_string(m_dims[i]);
            if(i<m_nDim-1){
                output+=",";
            }
        }
        output += "]";
        return output;
    }

    template <typename T>
    std::vector<int> Tensor<T>::shape() const {
        std::vector<int> output;
        for(int i = 0;i<m_nDim;i++){
            output.push_back(m_dims[i]);
        }
        return output;
    }

    // Return the total_size of the tensor
    template <typename T>
    int Tensor<T>::total_size() const {
        return m_total_size;
    }
    
    // Return the stride of each dimension
    template <typename T>
    std::string Tensor<T>::stride() const {
        std::string output = "[";
        for(int i = 0;i<m_nDim;i++){
            output += std::to_string(m_strides[i]);
            if(i<m_nDim-1){
                output+=",";
            }
        }
        output += "]";
        return output;
    }

    // Return the type of the elements
    template <typename T>
    std::string Tensor<T>::type() const {
        return typeid(T).name();
    }

    // Return a pointer to the elements
    template <typename T>
    T* Tensor<T>::data_ptr() const {
        return m_pData.get();
    }

    template <typename T>
    T& Tensor<T>::getPointerAtIndex(const int* dims){
        int index = m_start_index;
        int multiplier = 1;
        for (int i = m_nDim - 1; i >= 0; --i) {
            index += dims[i] * multiplier;
            multiplier *= m_dims[i];
        }
        return m_pData.get()[index];
    }

    template <typename U>
    std::ostream &operator<<(std::ostream &os, const Tensor<U> &m) {
        if (m.m_nDim == 0) {
            os << "tensor([])";
            return os;
        }

        std::vector<bool> print_ellipsis(m.m_nDim, false); // 用于跟踪是否打印省略号
        for(int i = 0;i<m.m_nDim;i++){
            if(m.m_dims[i] > 10){
                print_ellipsis[i] = true;
            }
        }
        if constexpr (std::is_floating_point<U>::value) {
            os << std::fixed << std::setprecision(4);
        }
        int max_length = 0;
        if constexpr(std::is_integral<U>::value || std::is_floating_point<U>::value){
            for (int i = 0; i < m.total_size(); ++i) {
                max_length = std::max(max_length, static_cast<int>(std::to_string(m.m_pData.get()[i]).length()));
            }
        }


        std::vector<int> indices(m.m_nDim, 0);  // 用于存储当前索引的向量
        std::vector<bool> dimensionEntered(m.m_nDim, false); // 用于跟踪是否进入了一个新的维度
        bool done = false;
        os<< "tensor(";
        while (!done) {
            // 遍历维度
            for (int dim = 0; dim < m.m_nDim; dim++) {
                if (!dimensionEntered[dim]) {
                    os << "[";
                    dimensionEntered[dim] = true;
                }
            }

            // 计算当前索引下的值
            int index = m.m_start_index;
            for (int i = 0; i < m.m_nDim; ++i) {
                index += indices[i] * m.m_strides[i];
            }

            if constexpr(std::is_integral<U>::value || std::is_floating_point<U>::value){
                os << std::setw(max_length) << m.m_pData.get()[index];
            }else if constexpr(typeid(U) == typeid(bool)){
                if(m.m_pData.get()[index]){
                    os << "True";
                }else{
                    os << "False";
                }

            }else{
                os << m.m_pData.get()[index];
            }
            std::string prefix_space = "        "; // 用于打印每一行的前缀空格
            // 更新索引并检查是否完成
            for (int dim = m.m_nDim - 1; dim >= 0; dim--) { // 从最内层往外更新，如果最内层到头了就更新上一层 ，break保证不会碰到未满的层的外层
                if (indices[dim] < m.m_dims[dim] - 1) { 
                    os << ", ";
                    for(int i = 0;i<dim;i++){
                        prefix_space += " ";
                    }
                    if(indices[dim] == 2 && print_ellipsis[dim]){
                        if(dim == m.m_nDim-1)os << " ..., ";
                        if(dim < m.m_nDim-1)os << "\n" << prefix_space << "...";
                        indices[dim] = m.m_dims[dim] - 3;
                    }else{
                        indices[dim]++;
                    }
                    if(dim == m.m_nDim-2) os << "\n" << prefix_space;
                    if(dim < m.m_nDim-2) os << "\n\n"<< prefix_space;
                    std::fill(dimensionEntered.begin() + dim + 1, dimensionEntered.end(), false);
                    break;
                } else {
                    os << "]";
                    if (dim == 0) done = true; // 如果最外层都到头了就结束
                    indices[dim] = 0; // 如果没到头就把当前层的index置0，然后继续更新上一层
                }
            }
        }
        os << ")";

        return os;
    }

    template <typename T>
    Tensor<T> zeros(const std::initializer_list<int>& dims){
        int total_size = 1;
        int nDim = dims.size();
        int* dim = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim[i] = *(dims.begin()+i);
            total_size *= dim[i];
        }
        T* data = new T[total_size];
        for(int i = 0;i<total_size;i++){
            data[i] = 0;
        }
        Tensor<T> t = Tensor<T>(data,dims);
        delete[] data;
        delete[] dim;
        return t;
    }
    template <typename T>
    Tensor<T> zeros(const int* dims, const int nDim){
        int total_size = 1;
        int* dim = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim[i] = dims[i];
            total_size *= dim[i];
        }
        T* data = new T[total_size];
        for(int i = 0;i<total_size;i++){
            data[i] = 0;
        }
        Tensor<T> t = Tensor<T>(data,dims,nDim);
        delete[] data;
        delete[] dim;
        return t;
    }

    template <typename T>
    Tensor<T> zeros(const std::vector<int> dims){
        int total_size = 1;
        int nDim = dims.size();
        int* dim = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim[i] = dims[i];
            total_size *= dim[i];
        }
        T* data = new T[total_size];
        for(int i = 0;i<total_size;i++){
            data[i] = 0;
        }
        int* dim2 = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim2[i] = dims[i];
        }
        Tensor<T> t = Tensor<T>(data,dim2,nDim);
        delete[] data;
        delete[] dim;
        return t;
    }

    template <typename T>
    Tensor<T> ones(const std::initializer_list<int>& dims){
        int total_size = 1;
        int nDim = dims.size();
        int* dim = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim[i] = *(dims.begin()+i);
            total_size *= dim[i];
        }
        T* data = new T[total_size];
        for(int i = 0;i<total_size;i++){
            data[i] = 1;
        }
        Tensor<T> t = Tensor<T>(data,dims);
        delete[] data;
        delete[] dim;
        return t;
    }




    template <typename T>
    Tensor<T> rand(const std::initializer_list<int>& dims){
        int total_size = 1;
        int nDim = dims.size();
        int* dim = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim[i] = *(dims.begin()+i);
            total_size *= dim[i];
        }
        T* data = new T[total_size];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 10);
        for(int i = 0;i<total_size;i++){
            data[i] = dis(gen);
        }
        Tensor<T> t = Tensor<T>(data,dims);
        delete[] data;
        delete[] dim;
        return t;

    }

    template <typename T>
    Tensor<T> rand(const std::vector<int>& dims){
        int total_size = 1;
        int nDim = dims.size();
        int* dim = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim[i] = *(dims.begin()+i);
            total_size *= dim[i];
        }
        T* data = new T[total_size];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 10);
        for(int i = 0;i<total_size;i++){
            data[i] = dis(gen);
        }
        int* dim2 = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim2[i] = dims[i];
        }
        Tensor<T> t = Tensor<T>(data,dim2,nDim);
        delete[] data;
        delete[] dim;
        return t;        
    }

    template <typename T>
    Tensor<T> eye(const int i){
        int total_size = i*i;
        int* dim = new int[2]{i,i};
        T* data = new T[total_size];
        for(int j = 0;j<total_size;j++){
            data[j] = 0;
        }
        for(int j = 0;j<i;j++){
            data[j*i+j] = 1;
        }
        Tensor<T> t = Tensor<T>(data,{i,i});
        delete[] data;
        delete[] dim;
        return t;
    }

    template <typename T>
    Tensor<T> full(const std::initializer_list<int>& dims, T value){
        int total_size = 1;
        int nDim = dims.size();
        int* dim = new int[nDim];
        for(int i = 0;i<nDim;i++){
            dim[i] = *(dims.begin()+i);
            total_size *= dim[i];
        }
        T* data = new T[total_size];
        std::fill(data, data + total_size, value);  // 使用 std::fill 替代循环
        Tensor<T> t = Tensor<T>(data,dims);
        delete[] data;
        delete[] dim;
        return t;
    }

    template <typename T>
    Tensor<T> arange(const int start, const int end, const int stride = 1){
        int total_size = (end-start)/stride;
        int* dim = new int[1]{total_size};
        T* data = new T[total_size];
        for(int i = 0;i<total_size;i++){
            data[i] = start+i*stride;
        }
        Tensor<T> t = Tensor<T>(data,{total_size});
        delete[] data;
        delete[] dim;
        return t;
    }

    template <typename T>
    T& Tensor<T>::operator()(const std::initializer_list<int> indices) const{
        int index = m_start_index;
        for(int i = 0;i<m_nDim;i++){
            index += *(indices.begin()+i)*m_strides[i];
        }
        return m_pData.get()[index];
    }

    template <typename T>
    T& Tensor<T>::operator()(const int* indices) const{
        int index = m_start_index;
        for(int i = 0;i<m_nDim;i++){
            index += *(indices+i)*m_strides[i];
        }
        return m_pData.get()[index];
    }

    template <typename T>
    T& Tensor<T>::operator()(const std::vector<int> indices) const{
        int index = m_start_index;
        for(int i = 0;i<m_nDim;i++){
            index += indices[i]*m_strides[i];
        }
        return m_pData.get()[index];
    }

    template <typename T>
    T& Tensor<T>::operator()(const int i, const int j, const int k) const{
        if(m_nDim != 3){
            throw std::invalid_argument("Dimensions of lhs must be 3.");
        }
        int index = m_start_index;
        index += i*m_strides[0];
        index += j*m_strides[1];
        index += k*m_strides[2];
        return m_pData.get()[index];
    }

    template <typename T>
    T& Tensor<T>::operator()(const int i, const int j) const{
        if(m_nDim != 2){
            throw std::invalid_argument("Dimensions of lhs must be 2.");
        }
        int index = m_start_index;
        index += i*m_strides[0];
        index += j*m_strides[1];
        return m_pData.get()[index];
    }
    

}
#endif // TS_TENSOR_H
