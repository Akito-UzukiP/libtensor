#include "tensor.h"
#include "iostream"
int main(){
    ts::Tensor<double> t1 = ts::Tensor<double>(new double[8]{1,2,3,4,5,6,7,8},{2,2,2});
    std::cout<<t1<<std::endl;
    std::cout<<t1.size()<<" "<<t1.type()<<" "<<t1.data_ptr()<<std::endl;
    ts::Tensor<double> t2 = ts::Tensor<double>::zeros({2,2,2});
    std::cout<<t2<<std::endl;
}