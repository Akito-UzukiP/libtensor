#ifndef TS_TENSOR_BASIC_H
#define TS_TENSOR_BASIC_H

#include <vector>
#include <string>
#include <typeinfo>
#include <iostream>
#include <iomanip>
#include <type_traits>
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
            Tensor(const int* dims, const int nDim); // Constructor
            Tensor(const std::vector<int>& dims); // Constructor
            Tensor(T* pData, const std::initializer_list<int>& dims); // Constructor
            Tensor(T* pData, const int* dims, const int nDim); // Hard Copy Constructor
            Tensor(const Tensor<T>& t, bool shallow_copy = true); // Shallow Copy constructor
            Tensor<T>& operator=(const Tensor<T>& t); // Copy assignment operator
            ~Tensor(); // Destructor
            
            std::string size() const; // Return the size of each dimension
            std::vector<int> shape() const; // Return the size of each dimension
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
            T& operator()(int* indices);  // 返回指定索引的元素，须保证indices的长度与m_nDim相同，否则会出现未定义行为
            T& operator()(std::vector<int> indices) const;
            T& operator()(int i , int j, int k); // 返回指定索引的元素，须保证m_nDim为3，否则会出现未定义行为
            T& operator()(int i , int j); // 返回指定索引的元素，须保证m_nDim为2，否则会出现未定义行为
            Tensor<T> operator()(int dim_index);  // 返回指定维度(n-1维)
            Tensor<T> operator()(int dim_index, std::initializer_list<int> indices); // 返回指定维度的指定索引的切片
          //  Tensor<T> operator()(std::initializer_list<int> dim_target, std::initializer_list<int> indices); // 返回指定维度的指定索引的切片

            // View操作
            Tensor<T> view(const std::initializer_list<int>& dims) const; // 返回一个新的张量，该张量与原张量共享数据，但形状不同
            //Tensor<T> view(const std::vector<int>& dims) const; // 返回一个新的张量，该张量与原张量共享数据，但形状不同
            Tensor<T> view(const int* dims, const int nDim) const; // 返回一个新的张量，该张量与原张量共享数据，但形状不同
            template <typename U>
            friend Tensor<U> view(const Tensor<U>& org, const std::initializer_list<int>& dims); // 返回一个新的张量，该张量与原张量共享数据，但形状不同

            // Transpose操作
            Tensor<T> transpose(const int dim1, const int dim2); // 返回一个新的张量，该张量与原张量共享数据，但stride和dims被交换
            template <typename U>
            friend Tensor<U> transpose(const Tensor<U>& org, const int dim1, const int dim2); // 返回一个新的张量，该张量与原张量共享数据，但形状不同
            
            // Permute操作
            Tensor<T> permute(const std::initializer_list<int>& dims); // 返回一个新的张量，该张量与原张量共享数据，但维度顺序不同
            template <typename U>
            friend Tensor<U> permute(const Tensor<U>& org, const std::initializer_list<int>& dims); // 返回一个新的张量，该张量与原张量共享数据，但维度顺序不同
            // Slice操作
            // TODO 在上面的operator()中实现

            // Concat操作
            // TODO
            template <typename U>
            friend Tensor<U> concat(const Tensor<U>& t1, const Tensor<U>& t2, const int axis); // 返回一个新的张量，且是t1和t2在axis维度上的拼接，并且新创建内存空间

            // tile操作
            template <typename U>
            friend Tensor<U> repeat_along_axis(const Tensor<U>& org,const int axis,const int count);

            template <typename U>
            friend Tensor<U> tile(const Tensor<U>& org, const std::initializer_list<int>& counts); 
            template <typename U>
            friend Tensor<U> tile(const Tensor<U>& org, const int* counts, const int nCount);



            // Mutate操作
            // TODO
            Tensor<T>& operator=(std::initializer_list<T> l); // 用列表中的元素替换张量中的元素


            // Squeeze和Unsqueeze操作
            Tensor<T> squeeze(const int dim) const; // 返回一个新的张量，该张量与原张量共享数据，但在dim维度上的大小为1的维度被删除
            template <typename U>
            friend Tensor<U> squeeze(const Tensor<U>& org, const int dim); // 返回一个新的张量，该张量与原张量共享数据，但在dim维度上的大小为1的维度被删除

            Tensor<T> unsqueeze(const int dim) const; // 返回一个新的张量，该张量与原张量共享数据，但在dim维度上增加一个大小为1的维度
            template <typename U>
            friend Tensor<U> unsqueeze(const Tensor<U>& org, const int dim); // 返回一个新的张量，该张量与原张量共享数据，但在dim维度上增加一个大小为1的维度


            template <typename U>
            friend Tensor<U> operator+(const Tensor<U>& t1, const Tensor<U>& t2); // 张量加法
            template <typename U>
            friend Tensor<U> operator-(const Tensor<U>& t1, const Tensor<U>& t2); // 张量减法
            template <typename U>
            friend Tensor<U> operator*(const Tensor<U>& t1, const Tensor<U>& t2); // 张量乘法
            template <typename U>
            friend Tensor<U> operator/(const Tensor<U>& t1, const Tensor<U>& t2); // 张量除法
            template <typename U>
            friend Tensor<U> log(const Tensor<U>& lhs); // 张量取对数

            Tensor<T> add(const Tensor<T>& t); // 张量加法
            template <typename U>
            friend Tensor<U> add(const Tensor<U>& t1, const Tensor<U>& t2); // 张量加法
            Tensor<T> sub(const Tensor<T>& t); // 张量减法
            template <typename U>
            friend Tensor<U> sub(const Tensor<U>& t1, const Tensor<U>& t2); // 张量减法
            Tensor<T> mul(const Tensor<T>& t); // 张量乘法
            template <typename U>
            friend Tensor<U> mul(const Tensor<U>& t1, const Tensor<U>& t2); // 张量乘法
            Tensor<T> div(const Tensor<T>& t); // 张量除法
            template <typename U>
            friend Tensor<U> div(const Tensor<U>& t1, const Tensor<U>& t2); // 张量除法

            // 矩阵乘法
            Tensor<T> matmul(const Tensor<T>& t); // 矩阵乘法
            template <typename U>
            friend Tensor<U> matmul(const Tensor<U>& t1, const Tensor<U>& t2); // 矩阵乘法

            // 基础的reduction计算
            Tensor<T> sum(const int dim); // 按照指定维度求和
            template <typename U>
            friend Tensor<U> sum(const Tensor<U>& t, const int dim); // 按照指定维度求和
            Tensor<T> mean(const int dim); // 按照指定维度求平均
            template <typename U>
            friend Tensor<U> mean(const Tensor<U>& t, const int dim); // 按照指定维度求平均
            Tensor<T> max(const int dim); // 按照指定维度求最大值
            template <typename U>
            friend Tensor<U> max(const Tensor<U>& t, const int dim); // 按照指定维度求最大值
            Tensor<T> min(const int dim); // 按照指定维度求最小值
            template <typename U>
            friend Tensor<U> min(const Tensor<U>& t, const int dim); // 按照指定维度求最小值
            // Tensor<T> argmax(const int dim); // 按照指定维度求最大值的索引
            // Tensor<T> argmin(const int dim); // 按照指定维度求最小值的索引
            // Tensor<T> norm(const int dim); // 按照指定维度求范数
            // Tensor<T> norm(const int dim, const int p); // 按照指定维度求p范数  
            // Tensor<T> norm(const int dim, const int p, const int keepdim); // 按照指定维度求p范数，keepdim表示是否保持维度

            //比较，有: gt ge lt le eq ne
            Tensor<bool> gt(const Tensor<T>& t); // 大于
            template <typename U>
            friend Tensor<bool> gt(const Tensor<U>& t1, const Tensor<U>& t2); // 大于
            Tensor<bool> operator>(const Tensor<T>& t); // 等于
            Tensor<bool> ge(const Tensor<T>& t); // 大于等于
            template <typename U>
            friend Tensor<bool> ge(const Tensor<U>& t1, const Tensor<U>& t2); // 大于等于
            Tensor<bool> operator>=(const Tensor<T>& t); // 等于
            Tensor<bool> lt(const Tensor<T>& t); // 小于
            template <typename U>
            friend Tensor<bool> lt(const Tensor<U>& t1, const Tensor<U>& t2); // 小于
            Tensor<bool> operator<(const Tensor<T>& t); // 等于
            Tensor<bool> le(const Tensor<T>& t); // 小于等于
            template <typename U>
            friend Tensor<bool> le(const Tensor<U>& t1, const Tensor<U>& t2); // 小于等于
            Tensor<bool> operator<=(const Tensor<T>& t); // 等于
            Tensor<bool> eq(const Tensor<T>& t); // 等于
            template <typename U>
            friend Tensor<bool> eq(const Tensor<U>& t1, const Tensor<U>& t2); // 等于
            Tensor<bool> operator==(const Tensor<T>& t); // 等于
            Tensor<bool> ne(const Tensor<T>& t); // 不等于
            template <typename U>
            friend Tensor<bool> ne(const Tensor<U>& t1, const Tensor<U>& t2); // 不等于
            Tensor<bool> operator!=(const Tensor<T>& t); // 不等于

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
    }

    template <typename T>
    Tensor<T>::Tensor(const int* dims, const int nDim){
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
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<int>& dims){
        m_nDim = dims.size();
        m_total_size = 1;
        m_dims = new int[m_nDim];
        m_strides = new int[m_nDim];
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
    Tensor<T>::Tensor(const Tensor<T>& other, bool shallow_copy){
        m_nDim = other.m_nDim;
        m_total_size = other.m_total_size;
        m_dims = new int[m_nDim];
        m_strides = new int[m_nDim];
        m_start_index = other.m_start_index;
        for(int i = 0;i<m_nDim;i++){
            m_dims[i] = other.m_dims[i];
            m_strides[i] = other.m_strides[i];
        }
        if(shallow_copy){
            m_pData = other.m_pData;
        }else{
            m_pData = std::shared_ptr<T[]>(new T[m_total_size]);
            
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
    T& Tensor<T>::operator()(std::initializer_list<int> indices){
        int index = m_start_index;
        for(int i = 0;i<m_nDim;i++){
            index += *(indices.begin()+i)*m_strides[i];
        }
        return m_pData.get()[index];
    }

    template <typename T>
    T& Tensor<T>::operator()(int* indices){
        int index = m_start_index;
        for(int i = 0;i<m_nDim;i++){
            index += *(indices+i)*m_strides[i];
        }
        return m_pData.get()[index];
    }

    template <typename T>
    T& Tensor<T>::operator()(std::vector<int> indices) const{
        int index = m_start_index;
        for(int i = 0;i<m_nDim;i++){
            index += indices[i]*m_strides[i];
        }
        return m_pData.get()[index];
    }

    template <typename T>
    T& Tensor<T>::operator()(int i, int j, int k){
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
    T& Tensor<T>::operator()(int i, int j){
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
