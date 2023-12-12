// 请确保这个文件只被包含在 tensor.h 中
namespace ts {

    // Default constructor
    template <typename T>
    Tensor<T>::Tensor() {
        m_nDim = 0;
        m_dims = std::vector<int>();
        m_pData = nullptr;
    }

    // Copy constructor
    template <typename T>
    Tensor<T>::Tensor(const Tensor &obj) {
        m_nDim = obj.m_nDim;
        m_dims = obj.m_dims;
        m_pData = new T[obj.m_dims.size()];
        std::copy(obj.m_pData, obj.m_pData + obj.m_dims.size(), m_pData);
    }

    // Assignment operator
    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor &obj) {
        if (this != &obj) {
            delete[] m_pData;
            m_nDim = obj.m_nDim;
            m_dims = obj.m_dims;
            m_pData = new T[obj.m_dims.size()];
            std::copy(obj.m_pData, obj.m_pData + obj.m_dims.size(), m_pData);
        }
        return *this;
    }

    // Parameterized constructor
    template <typename T>
    Tensor<T>::Tensor(T* pData, int nDim, const std::vector<int>& dims) {
        m_nDim = nDim;
        m_dims = dims;
        m_pData = new T[dims.size()];
        std::copy(pData, pData + dims.size(), m_pData);
    }

    template <typename T>
    Tensor<T>::Tensor(T* pData,  const std::vector<int>& dims) {
        m_nDim = dims.size();
        m_dims = dims;
        m_pData = new T[dims.size()];
        std::copy(pData, pData + dims.size(), m_pData);
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

    // ... 其他成员函数的实现 ...

}
