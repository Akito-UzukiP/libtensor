// #include "tensor.h"
// #include <iostream>
// using namespace ts;

// // Default constructor
// template <typename T>
// Tensor<T>::Tensor() {
//     m_nDim = 0;
//     m_dims = std::vector<int>();
//     m_pData = nullptr;
// }

// // Copy constructor
// template <typename T>
// Tensor<T>::Tensor(const Tensor &obj) {
//     m_nDim = obj.m_nDim;
//     m_dims = obj.m_dims;
//     m_pData = new T[obj.m_dims[0]];
//     for (int i = 0; i < obj.m_dims[0]; i++) {
//         m_pData[i] = obj.m_pData[i];
//     }
// }

// // Assignment operator
// template <typename T>
// Tensor<T> &Tensor<T>::operator=(const Tensor &obj) {
//     if (this != &obj) {
//         m_nDim = obj.m_nDim;
//         m_dims = obj.m_dims;
//         m_pData = new T[obj.m_dims[0]];
//         for (int i = 0; i < obj.m_dims[0]; i++) {
//             m_pData[i] = obj.m_pData[i];
//         }
//     }
//     return *this;
// }
// template <typename T>
// Tensor<T>::Tensor(T* pData, int nDim, const std::vector<int>& dims) {
//     m_nDim = nDim;
//     m_dims = dims;
//     m_pData = new T[dims[0]];
//     for (int i = 0; i < dims[0]; i++) {
//         m_pData[i] = pData[i];
//     }
// }

// template <typename T>
// Tensor<T>::Tensor(T* pData,  const std::vector<int>& dims) {
//     m_nDim = dims.size();
//     m_dims = dims;
//     m_pData = new T[dims[0]];
//     for (int i = 0; i < dims[0]; i++) {
//         m_pData[i] = pData[i];
//     }
// }

// // Destructor
// template <typename T>
// Tensor<T>::~Tensor() {
//     delete[] m_pData;
// }

// // Return the size of each dimension
// template <typename T>
// std::vector<int> Tensor<T>::size() const {
//     return m_dims;
// }

// // Return the type of the elements
// template <typename T>
// std::string Tensor<T>::type() const {
//     return typeid(T).name();
// }

// // Return a pointer to the elements
// template <typename T>
// T* Tensor<T>::data_ptr() const {
//     return m_pData;
// }
