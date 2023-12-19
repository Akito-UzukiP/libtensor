#include "tensor.h"
#include "iostream"
int main(){
    ts::Tensor<double> t1 = ts::Tensor<double>(new double[8]{1,2,3,4,5,6,7,8},{2,2,2});
    std::cout<<t1<<std::endl;
    std::cout<<t1.size()<<" "<<t1.type()<<" "<<t1.data_ptr()<<std::endl;
    ts::Tensor<double> t2 = ts::zeros<double>({2,2,2});
    std::cout<<t2<<std::endl;
    ts::Tensor<double> t3 = ts::ones<double>({2,2,2});
    std::cout<<t3<<std::endl;
    ts::Tensor<double> t4 = ts::rand<double>({2,2,2});
    std::cout<<t4<<std::endl;
    ts::Tensor<double> t5 = ts::eye<double>(3);
    std::cout<<t5<<std::endl;
    ts::Tensor<double> t6 = ts::full<double>({2,2,2},3.45);
    std::cout<<t6<<std::endl;
}