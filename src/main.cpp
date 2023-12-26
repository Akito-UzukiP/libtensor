
#include "tensor.h"
#include "iostream"
int main(){
    ts::Tensor<int> a = ts::arange<int>(0,120).view({2,3,4,5});
    ts::Tensor<int> b = ts::arange<int>(120,240).view({2,3,5,4});
    std::cout<<a.matmul(b)<<std::endl;
    std::cout<<ts::matmul(a,b)<<std::endl;

}