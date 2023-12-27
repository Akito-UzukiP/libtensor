
#include "tensor.h"
#include "iostream"
int main(){
    ts::Tensor<int> a = ts::arange<int>(0,120).view({3,4,5,2});
    ts::Tensor<int> b = ts::arange<int>(0,6).view({2,3});
    std::cout<<ts::tile(b,{2,3,1})<<std::endl;

}