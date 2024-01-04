#include <iostream>
#include "tensor.h"
#include <assert.h>
#include <immintrin.h>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
void testConstructionAndAssignment() {
    ts::Tensor<int> a;
    ts::Tensor<int> b = ts::arange<int>(0, 10).view({2, 5});
    ts::Tensor<int> c = b; // 测试复制构造函数
    ts::Tensor<int> d;
    d = b; // 测试赋值运算符
    //assert(a.data_ptr() != c.data_ptr()); // 确保深复制
    assert(b(0, 0) == c(0, 0)); // 检查内容
}

void testElementAccess() {
    ts::Tensor<int> a = ts::arange<int>(0, 6).view({2, 3});
    // 测试元素访问
    assert(a(1, 2) == 5);
    // 测试修改元素
    a(1, 2) = 10;
    assert(a(1, 2) == 10);
}

void testArithmeticOperations() {
    ts::Tensor<int> a = ts::arange<int>(0, 6).view({2, 3});// [[0,1,2],[3,4,5]]
    ts::Tensor<int> b = ts::arange<int>(0, 6).view({2, 3});
    ts::Tensor<int> c = a + b;
    // 确认加法结果
    assert(c(1, 2) == 10);
    c = a-b;
    assert(c(1,2) == 0);
    c = a*b;
    assert(c(1,2) == 25);
    // 其他算术操作...
}

void testMatrixOperations() {
    ts::Tensor<int> a = ts::arange<int>(0, 6).view({2, 3});
    ts::Tensor<int> b = ts::arange<int>(0, 6).view({3, 2});
    ts::Tensor<int> c = a.matmul(b);
    ts::Tensor<int> d = ts::einsum<int>("xy,yz->xz",{a,b});
    std::cout<<(c == d)<<std::endl;
    std::cout<<ts::einsum<int>("ij->",{a})<<std::endl;
    ts::Tensor<double> e = ts::rand<double>({3,4,5});
    ts::Tensor<double> f = ts::rand<double>({3,4,5}).transpose(1,2);
    ts::Tensor<double> g = e.matmul(f);
    ts::Tensor<double> h = ts::einsum<double>("nij,njk->nik",{e,f});
    ts::Tensor<bool> all_false = ts::full<bool>({3,4,4},false);
    ts::Tensor<bool> all_true = ts::full<bool>({3,4,4}, true);
    std::cout<<(g == h) << std::endl;
    std::cout<<(g >= h) << std::endl;
    std::cout<<(g > h) << std::endl;


    // 测试矩阵乘法结果
    // ...
}

int main(){
    ts::Tensor<int> a = ts::arange<int>(0, 6).view({2, 3}).transpose(0,1);
    ts::serialize<int>(a, "test.txt");
    ts::Tensor<int> b = ts::deserialize<int>("test.txt");
    std::cout<<b<<std::endl;
    return 0;
}
