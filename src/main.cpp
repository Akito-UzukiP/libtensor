#include "tensor.h"
#include "iostream"
int main(){
    ts::Tensor<double> t1 = ts::Tensor<double>(new double[8]{1,2,3,4,5,6,7,8},{2,2,2});
    std::cout<<t1<<std::endl;
    std::cout<<t1.size()<<" "<<t1.type()<<" "<<t1.data_ptr()<<" "<<t1.stride()<<std::endl;
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
    t6({0,0,0}) = 1;
    std::cout<<t6<<std::endl;
    ts::Tensor<double> t7 = t1.view({4,2});
    std::cout<<t7<<std::endl;
    ts::Tensor<double> t8 = ts::view(t1,{8,1});
    std::cout<<t8<<std::endl;
    ts::Tensor<double> t9 = ts::transpose(t1,0,2);
    std::cout<<t9<<std::endl;
    ts::Tensor<double> t10 = t1.transpose(0,1);
    std::cout<<t10<<std::endl;
    ts::Tensor<double> t11 = ts::arange<double>(0,60,1);
    std::cout<<t11<<std::endl;
    ts::Tensor<double> t12 = t11.view({3,4,5});
    std::cout<<t12<<std::endl;
    ts::Tensor<double> t13 = t12.permute({1,0,2});
    std::cout<<t13<<std::endl;
    ts::Tensor<double> t14 = ts::permute(t13,{1,0,2});
    std::cout<<t14<<std::endl;
    ts::Tensor<double> t15 = t14(0);
    std::cout<<t15<<std::endl;
    std::cout<<t15.size()<<" "<<t15.type()<<" "<<t15.data_ptr()<<" "<<t15.stride()<<" "<<t15.total_size()<<std::endl;
    ts::Tensor<double> t16 = t14(0,{1,4});
    std::cout<<t16<<std::endl;
    std::cout<<t16.size()<<" "<<t16.type()<<" "<<t16.data_ptr()<<" "<<t16.stride() << " " <<t16.total_size()<<std::endl;
    t14(1,{1,3}) = {0,1,2,3,4};
    std::cout<<t14<<std::endl;

    ts::Tensor<double> t17 = ts::Tensor<double>(new double[8]{1,2,3,4,5,6,7,8},{2,2,2});
    ts::Tensor<double> t18 = ts::Tensor<double>(new double[12]{11,22,33,44,55,66,77,88,99,1010,1111,1212},{2,3,2});
    ts::Tensor<double> t19 = ts::concat(t17,t18,1);
    std::cout<<t19<<std::endl;
}