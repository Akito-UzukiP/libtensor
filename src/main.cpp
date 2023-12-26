
#include "tensor.h"
#include "iostream"
int main(){
    ts::Tensor<int> a = ts::arange<int>(0,120).view({2,3,4,5});
    ts::Tensor<int> b = ts::arange<int>(120,240).view({2,3,5,4});
    // std::cout<<a.matmul(b)<<std::endl;
    // std::cout<<ts::matmul(a,b)<<std::endl;
    // std::cout<<ts::mean(a,0)<<std::endl;
    // std::cout<<a.mean(1)<<std::endl;
    std::cout<<ts::max(a,1)<<std::endl;
    std::cout<<a.max(1)<<std::endl;
    // std::cout<<ts::min(a,0)<<std::endl;
    // std::cout<<a.min(1)<<std::endl;

}