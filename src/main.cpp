
#include "tensor.h"
#include "iostream"
int main(){
    ts::Tensor<double> t1 = ts::arange<double>(1,7,1);
    t1 = t1.view({3,2});
    std::cout << t1 << std::endl;
    std::cout << t1-t1 << std::endl;
    std::cout << t1+t1 << std::endl;
    std::cout << t1*t1 << std::endl;
    std::cout << t1/t1 << std::endl;

    std::cout<< ts::add(t1,t1) << std::endl;
    std::cout<< ts::sub(t1,t1) << std::endl;
    std::cout<< ts::mul(t1,t1) << std::endl;
    std::cout<< ts::div(t1,t1) << std::endl;

    std::cout<< t1.add(t1) << std::endl;
    std::cout<< t1.sub(t1) << std::endl;
    std::cout<< t1.mul(t1) << std::endl;
    std::cout<< t1.div(t1) << std::endl;
}