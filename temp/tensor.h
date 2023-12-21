#ifndef TS_TENSOR_H
#define TS_TENSOR_H

#include <vector>
#include <string>
#include <typeinfo>
#include <iostream>
#include <random>
#include <memory>
namespace ts {
    template <typename T>
    class Tensor {
        private:
            int m_nDim; // Dimension count
            int m_total_size; // Total size of the tensor
            int*  m_dims; // Dimension sizes
            int m_start_index; // Start index
            //T *m_pData; // Element array,看起来得重写一个data类，提供一些基本的操作，或者干脆就用智能指针接手
            std::shared_ptr<T[]> m_pData;

            /*
            * m_strides 数组用于存储张量每个维度上的步长信息。
            * “步长”指的是，在沿着某一维度移动到下一个元素时，需要在内存中跨越的字节数。
            * 例如，如果一个二维张量（矩阵）的类型为 int（假设每个 int 为 4 字节），
            * 那么在行方向上的步长就是该行所有元素所占字节数总和。
            * 这个信息对于快速定位多维张量中的元素非常重要，
            * 特别是在进行如转置、切片等不改变数据排列但改变视图的操作时。
            * 通过这种方式，我们可以有效地重用相同的数据缓冲区，而不必进行数据的实际复制，
            * 这对于提高性能和减少内存使用非常有帮助。
            */

            int* m_strides; // Strides,

        public:
            Tensor();  // Default constructor
            Tensor(T* pData, const std::initializer_list<int>& dims); // Constructor
            Tensor(T* pData, const int* dims, const int nDim); // Constructor
            Tensor(const Tensor<T>& t); // Shallow Copy constructor
            Tensor<T>& operator=(const Tensor<T>& t); // Copy assignment operator
            ~Tensor(); // Destructor
            
            std::string size() const; // Return the size of each dimension
            int total_size() const; // Return the total size of the tensor
            std::string type() const; // Return the type of the elements
            std::string stride() const; // Return the stride of each dimension
            T* data_ptr() const; // Return a pointer to the elements

           // inline T& Tensor<T>::getPointerAtIndex(int oneDIndex); // 将一维索引转化为实际多维索引处的引用
           inline T& getPointerAtIndex(const int* dims); // 将多维数组索引转化为实际多维索引处的引用，保证dims的长度与m_nDim相同

            //print方法

            template <typename U>
            friend std::ostream & operator<<(std::ostream & os, const Tensor<U> & m);

            // 重载运算符
            T& operator()(std::initializer_list<int> indices);  // 返回指定索引的元素
            Tensor<T> operator()(int dim_index);  // 返回指定维度(n-1维)
            Tensor<T> operator()(int dim_index, std::initializer_list<int> indices); // 返回指定维度的指定索引的切片
            Tensor<T> operator()(std::initializer_list<int> dim_target, std::initializer_list<int> indices); // 返回指定维度的指定索引的切片

            // View操作
            Tensor<T> view(const std::initializer_list<int>& dims); // 返回一个新的张量，该张量与原张量共享数据，但形状不同
            template <typename U>
            friend Tensor<U> view(const Tensor<U> org, const std::initializer_list<int>& dims); // 返回一个新的张量，该张量与原张量共享数据，但形状不同

            // Transpose操作
            Tensor<T> transpose(const int dim1, const int dim2); // 返回一个新的张量，该张量与原张量共享数据，但stride和dims被交换
            template <typename U>
            friend Tensor<U> transpose(const Tensor<U> org, const int dim1, const int dim2); // 返回一个新的张量，该张量与原张量共享数据，但形状不同
            
            // Permute操作
            Tensor<T> permute(const std::initializer_list<int>& dims); // 返回一个新的张量，该张量与原张量共享数据，但维度顺序不同
            template <typename U>
            friend Tensor<U> permute(const Tensor<U> org, const std::initializer_list<int>& dims); // 返回一个新的张量，该张量与原张量共享数据，但维度顺序不同
            // Slice操作
            // TODO 在上面的operator()中实现

            // Concat操作
            // TODO
            template <typename U>
            friend Tensor<U> concat(const Tensor<U> t1, const Tensor<U> t2, const int axis); // 返回一个新的张量，且是t1和t2在axis维度上的拼接，并且新创建内存空间

            // Mutate操作
            // TODO
            Tensor<T>& operator=(std::initializer_list<T> l); // 用列表中的元素替换张量中的元素
    };

    // Default constructor
    template <typename T>
    Tensor<T>::Tensor() {
        m_nDim = 0;
        m_total_size = 0;
        m_dims = nullptr;
        m_pData = nullptr;
    }

    template <typename T>
    Tensor<T>::Tensor(T* pData, const std::initializer_list<int>& dims){
        m_nDim = dims.size();
        m_total_size = 1;
        m_dims = new int[m_nDim];
        m_strides = new int[m_nDim];
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
        m_strides = new int[m_nDim];
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

    // Shallow Copy constructor
    template <typename T> 
    Tensor<T>::Tensor(const Tensor<T>& other){
        m_nDim = other.m_nDim;
        m_total_size = other.m_total_size;
        m_dims = new int[m_nDim];
        m_strides = new int[m_nDim];
        m_start_index = other.m_start_index;
        for(int i = 0;i<m_nDim;i++){
            m_dims[i] = other.m_dims[i];
            m_strides[i] = other.m_strides[i];
        }
        m_pData = other.m_pData;
    }

    // Shallow Copy assignment operator
    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other){
        if(this == &other){
            return *this;
        }
        m_nDim = other.m_nDim;
        m_total_size = other.m_total_size;
        m_dims = new int[m_nDim];
        m_strides = new int[m_nDim];
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
            os << "[]";
            return os;
        }

        std::vector<int> indices(m.m_nDim, 0);  // 用于存储当前索引的向量
        std::vector<bool> dimensionEntered(m.m_nDim, false); // 用于跟踪是否进入了一个新的维度
        bool done = false;

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
            os << m.m_pData.get()[index];

            // 更新索引并检查是否完成
            for (int dim = m.m_nDim - 1; dim >= 0; dim--) { // 从最内层往外更新，如果最内层到头了就更新上一层 ，break保证不会碰到未满的层的外层
                if (indices[dim] < m.m_dims[dim] - 1) { 
                    indices[dim]++;
                    os << ", ";
                    std::fill(dimensionEntered.begin() + dim + 1, dimensionEntered.end(), false);
                    break;
                } else {
                    os << "]";
                    if (dim == 0) done = true; // 如果最外层都到头了就结束
                    indices[dim] = 0; // 如果没到头就把当前层的index置0，然后继续更新上一层
                }
            }
        }

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
    // template <typename T>
    // Tensor<T> ones(const int* dims, const int nDim){
    //     int total_size = 1;
    //     int* dim = new int[nDim];
    //     for(int i = 0;i<nDim;i++){
    //         total_size *= *(dims.begin+i);
    //     }
    //     T* data = new T[total_size];
    //     for(int i = 0;i<total_size;i++){
    //         data[i] = 1;
    //     }
    //     Tensor<T> t = Tensor<T>(data,dims);
    //     delete[] data;
    //     return t;
    // }




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
        for(int i = 0;i<total_size;i++){
            data[i] = value;
        }
        Tensor<T> t = Tensor<T>(data,dims);
        delete[] data;
        delete[] dim;
        return t;
    }

    template <typename T>
    Tensor<T> arange(const int start, const int end, const int stride){
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
    T& Tensor<T>::operator()(std::initializer_list<int> indices){
        int index = m_start_index;
        int multiplier = 1;
        for (int i = m_nDim - 1; i >= 0; --i) {
            index += *(indices.begin()+i) * multiplier;
            multiplier *= m_dims[i];
        }
        return m_pData.get()[index];
    }
    

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
        return Tensor<T>(m_pData.get(),dims);
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
        return Tensor<U>(org.m_pData.get(),dims);
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
        return t;

    }


}
#endif // TS_TENSOR_H
