#include <iostream>
#include <vector>
#include "tensor.h"


int main(){
    int a[2][3] = {{1,2,3},{4,5,6}};
    std::vector<int> b = {3};
    ts::Tensor<int> c(a, 1, b);
    std::cout << c.type() << std::endl;
    std::cout << c.size()[0] << std::endl;
    std::cout << c.data_ptr()[0] << std::endl;
    std::cout << c.data_ptr()[1] << std::endl;
    std::cout << c.data_ptr()[2] << std::endl;
    std::cout << c.data_ptr() << std::endl;
    return 0;
}