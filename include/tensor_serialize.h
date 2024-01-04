#ifndef TENSOR_SERIALIZE_H
#define TENSOR_SERIALIZE_H
#include "tensor_basic.h"
#include "tensor_operation.h"
#include "fstream"
#include "iostream"
#include "memory"
// enum class TensorType{
//     SHORT,
//     INT,
//     LONG,
//     FLOAT,
//     DOUBLE,
//     BOOL
// };
// 序列化定义：
namespace ts{

    template <typename T>
    void Tensor<T>::serialize(const std::string& filename) const{
        std::ofstream out(filename, std::ios::binary);
        if(!out.is_open()){
            std::cout << "open file failed" << std::endl;
            return;
        }
        char type;
        if(std::is_same<T, int16>::value){
            type = 's';
        }else if(std::is_same<T, int32>::value){
            type = 'i';
        }else if(std::is_same<T, int64>::value){
            type = 'l';
        }else if(std::is_same<T, float32>::value){
            type = 'f';
        }else if(std::is_same<T, float64>::value){
            type = 'd';
        }else if(std::is_same<T, bool>::value){
            type = 'b';
        }else{
            std::cout << "unsupported type" << std::endl;
            return;
        }
        /*
         * 内存结构:[type(1)][m_nDim(4)][m_total_size(8)][m_dims(4*m_nDim)][m_strides(8*m_nDim)][m_pData(m_total_size*sizeof(T))]
         * 先contiguous再写入
        */
        Tensor<T> contiguous_tensor = this->contiguous();
        out.write(&type, sizeof(char));
        out.write((char*)&contiguous_tensor.m_nDim, sizeof(int));
        out.write((char*)&contiguous_tensor.m_total_size, sizeof(long));
        out.write((char*)contiguous_tensor.m_dims, sizeof(int)*contiguous_tensor.m_nDim);
        out.write((char*)contiguous_tensor.m_strides, sizeof(long)*contiguous_tensor.m_nDim);
        out.write((char*)contiguous_tensor.m_pData.get(), contiguous_tensor.m_total_size*sizeof(T));

        out.close();

    }


    template <typename T>
    void serialize(const Tensor<T>& tensor, const std::string& filename){
        tensor.serialize(filename);
    }


    template <typename T>
    static Tensor<T> deserialize(const std::string& filename){
        std::ifstream in(filename, std::ios::binary);
        if(!in.is_open()){
            std::cout << "open file failed" << std::endl;
            return Tensor<T>();
        }
        char type;
        in.read(&type, sizeof(char));
        int nDim;
        in.read((char*)&nDim, sizeof(int));
        long total_size;
        in.read((char*)&total_size, sizeof(long));
        int* dims = new int[nDim];
        in.read((char*)dims, sizeof(int)*nDim);
        long* strides = new long[nDim];
        in.read((char*)strides, sizeof(long)*nDim);
        T* data = new T[total_size];
        in.read((char*)data, sizeof(T)*total_size);
        in.close();
        Tensor<T> tensor(data, dims,nDim);
        return tensor;
    }
};
#endif