
#include "tensor_calculation.h"
#include "iostream"
int main(){
    ts::Tensor<double> t1 = ts::arange<double>(0,6,1);
    t1 = t1.view({3,2});
    std::cout<<t1.size()<<" "<<t1.type()<<" "<<t1.data_ptr()<<" "<<t1.stride()<<std::endl;
    std::cout<<t1<<std::endl;
    t1 = t1.view({1,1,1,1,1,1,6});
    t1 = t1.sub(t1);
    std::cout<<t1.size()<<" "<<t1.type()<<" "<<t1.data_ptr()<<" "<<t1.stride()<<std::endl;
    std::cout<<t1<<std::endl;

}