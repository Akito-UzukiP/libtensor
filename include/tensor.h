#ifndef TS_TENSOR_H
#define TS_TENSOR_H

#include <vector>
#include <string>
#include <typeinfo>
#include <iostream>
#include <random>
namespace ts {
    template <typename T>
    class Tensor {
        private:
            int m_nDim; // Dimension count
            int m_total_size; // Total size of the tensor
            int*  m_dims; // Dimension sizes
            T *m_pData; // Element array


        public:
            Tensor();  // Default constructor
            Tensor(T* pData, const std::initializer_list<int>& dims); // Constructor
            ~Tensor(); // Destructor
            
            std::string size() const; // Return the size of each dimension
            std::string type() const; // Return the type of the elements
            T* data_ptr() const; // Return a pointer to the elements
            //print方法
            void printTensor(std::ostream & os, int dim, int* indices) const;
            template <typename U>
            friend std::ostream & operator<<(std::ostream & os, const Tensor<U> & m);
            // 特殊初始化方法
            static Tensor<T> zeros(const std::initializer_list<int>& dims);
            static Tensor<T> ones(const std::initializer_list<int>& dims);
            static Tensor<T> rand(const std::initializer_list<int>& dims);
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
        for(int i = 0;i<dims.size();i++){
            m_dims[i] = *(dims.begin()+i);
            m_total_size *= m_dims[i];
        }
        m_pData = new T[m_total_size];
        for(int i = 0;i<m_total_size;i++){
            m_pData[i] = pData[i];
        }


    }


    // Destructor
    template <typename T>
    Tensor<T>::~Tensor() {
        delete[] m_pData;
        delete[] m_dims;
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

    // Return the type of the elements
    template <typename T>
    std::string Tensor<T>::type() const {
        return typeid(T).name();
    }

    // Return a pointer to the elements
    template <typename T>
    T* Tensor<T>::data_ptr() const {
        return m_pData;
    }
    template <typename T>
    void Tensor<T>::printTensor(std::ostream & os, int dim, int* indices) const {
        if (dim == m_nDim) {
            // Calculate the index in the linear array
            int index = 0;
            int multiplier = 1;
            for (int i = m_nDim - 1; i >= 0; --i) {
                index += indices[i] * multiplier;
                multiplier *= m_dims[i];
            }
            os << m_pData[index];
            return;
        }

        os << "[";
        for (int i = 0; i < m_dims[dim]; ++i) {
            indices[dim] = i;
            printTensor(os, dim + 1, indices);
            if (i < m_dims[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
    }
    template <typename U>
    std::ostream & operator<<(std::ostream & os, const Tensor<U> & m){
        if(m.m_nDim == 0){
            os << "[]";
            return os;
        }
        int* indices = new int[m.m_nDim]();
        m.printTensor(os,0,indices);
        delete[] indices;
        return os;
    }

    template <typename T>
    Tensor<T> Tensor<T>::zeros(const std::initializer_list<int>& dims){
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
    Tensor<T> Tensor<T>::ones(const std::initializer_list<int>& dims){
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
    Tensor<T> Tensor<T>::rand(const std::initializer_list<int>& dims){
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

    }
}
#endif // TS_TENSOR_H
