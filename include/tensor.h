#ifndef TS_TENSOR_H
#define TS_TENSOR_H

#include <vector>
#include <string>
#include <typeinfo>

namespace ts {
    template <typename T>
    class Tensor {
        private:
            int m_nDim; // Dimension count
            int m_total_size; // Total size of the tensor
            std::vector<int> m_dims; // Dimension sizes
            T *m_pData; // Element array


        public:
            Tensor();  // Default constructor
            Tensor(const Tensor &obj); // Copy constructor
            Tensor &operator=(const Tensor &obj); // Assignment operator
            Tensor(T* pData, int nDim, const std::vector<int>& dims); // Parameterized constructor
            Tensor(T* pData, int nDim, const int* dims); // Parameterized constructor
            Tensor(T* pData,  const std::vector<int>& dims); // Parameterized constructor
            ~Tensor(); // Destructor
            
            std::vector<int> size() const; // Return the size of each dimension
            std::string type() const; // Return the type of the elements
            T* data_ptr() const; // Return a pointer to the elements
            // ... other member functions and data ...
    };

    // Default constructor
    template <typename T>
    Tensor<T>::Tensor() {
        m_nDim = 0;
        m_total_size = 0;
        m_dims = std::vector<int>();
        m_pData = nullptr;
    }

    // Copy constructor
    template <typename T>
    Tensor<T>::Tensor(const Tensor &obj) {
        m_nDim = obj.m_nDim;
        m_dims = obj.m_dims;
        m_total_size = obj.m_total_size;
        m_pData = new T[m_total_size];
        std::copy(obj.m_pData, obj.m_pData + m_total_size, m_pData);
    }

    // Assignment operator
    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor &obj) {
        if (this != &obj) {
            delete[] m_pData;
            m_nDim = obj.m_nDim;
            m_dims = obj.m_dims;
            m_total_size = obj.m_total_size;
            m_pData = new T[m_total_size];
            std::copy(obj.m_pData, obj.m_pData + m_total_size, m_pData);
        }
        return *this;
    }

    // Parameterized constructor
    template <typename T>
    Tensor<T>::Tensor(T* pData, int nDim, const std::vector<int>& dims) {
        m_nDim = nDim;
        m_dims = dims;
        m_total_size = 1;
        for(int i = 0;i<nDim;i++){
            m_total_size *= dims[i];
        }
        m_pData = new T[m_total_size];
        std::copy(pData, pData + m_total_size, m_pData);
    }
    template <typename T>
    Tensor<T>::Tensor(T* pData, int nDim, const int* dims) {
        m_nDim = nDim;
        m_dims = std::vector<int>(dims, dims + nDim);
        m_total_size = 1;
        for(int i = 0;i<nDim;i++){
            m_total_size *= dims[i];
        }
        m_pData = new T[m_total_size];
        std::copy(pData, pData + m_total_size, m_pData);
    }

    template <typename T>
    Tensor<T>::Tensor(T* pData,  const std::vector<int>& dims) {
        m_nDim = dims.size();
        m_dims = dims;
        m_total_size = 1;
        for(int i = 0;i<m_nDim;i++){
            m_total_size *= dims[i];
        }
        m_pData = new T[m_total_size];
        std::copy(pData, pData + m_total_size, m_pData);
    }

    // Destructor
    template <typename T>
    Tensor<T>::~Tensor() {
        delete[] m_pData;
    }

    // Return the size of each dimension
    template <typename T>
    std::vector<int> Tensor<T>::size() const {
        return m_dims;
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

}

#endif // TS_TENSOR_H
