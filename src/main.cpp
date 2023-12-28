#include <iostream>
#include "tensor.h"
#include <assert.h>

void testConstructionAndAssignment() {
    ts::Tensor<int> a;
    ts::Tensor<int> b = ts::arange<int>(0, 10).view({2, 5});
    ts::Tensor<int> c = b; // 测试复制构造函数
    ts::Tensor<int> d;
    d = b; // 测试赋值运算符
    // 测试内容是否相同
    //assert(b.data_ptr() != c.data_ptr()); // 确保深复制
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
    ts::Tensor<int> a = ts::arange<int>(0, 6).view({2, 3});
    ts::Tensor<int> b = ts::arange<int>(0, 6).view({2, 3});
    ts::Tensor<int> c = a + b;
    // 确认加法结果
    assert(c(1, 2) == 10);
    // 其他算术操作...
}

void testMatrixOperations() {
    ts::Tensor<int> a = ts::arange<int>(0, 6).view({2, 3});
    ts::Tensor<int> b = ts::arange<int>(0, 6).view({3, 2});
    ts::Tensor<int> c = a.matmul(b);
    // 测试矩阵乘法结果
    // ...
}

int main() {
    // testConstructionAndAssignment();
    // testElementAccess();
    // testArithmeticOperations();
    // testMatrixOperations();

    // std::cout << "All tests passed!" << std::endl;
    ts::Tensor<int> b = ts::arange<int>(0, 16).view({4, 4});
    std::cout << ts::einsum<int>("ii->",{b}) << std::endl; // 预期输出：0+5+10+15 = 30
    ts::Tensor<int> c = ts::arange<int>(0, 12).view({3, 4});
    std::cout << ts::einsum<int>("ij->i",{c}) << std::endl; // 预期输出：每行元素之和
    ts::Tensor<int> d = ts::arange<int>(0, 12).view({3, 4});
    std::cout << ts::einsum<int>("ij->j",{d}) << std::endl; // 预期输出：每列元素之和
    ts::Tensor<int> e = ts::arange<int>(0, 6).view({2, 3});
    std::cout << ts::einsum<int>("ij->ji",{e}) << std::endl; // 预期输出：矩阵的转置
    ts::Tensor<int> f = ts::arange<int>(0, 4).view({2, 2});
    ts::Tensor<int> g = ts::arange<int>(0, 4).view({2, 2});
    std::cout << ts::einsum<int>("ij,ij->",{f, g}) << std::endl; // 预期输出：对应元素乘积之和
    ts::Tensor<int> j = ts::arange<int>(0, 4).view({2, 2});
    ts::Tensor<int> k = ts::arange<int>(0, 4).view({2, 2});
    std::cout << ts::einsum<int>("ij,ji->", {j, k}) << std::endl;
    ts::Tensor<int> h = ts::arange<int>(0, 27).view({3, 3, 3});
    ts::Tensor<int> i = ts::arange<int>(0, 27).view({3, 3, 3});
    std::cout << ts::einsum<int>("ijk,ikl->ijl", {h, i}) << std::endl;

    ts::Tensor<int> l = ts::arange<int>(0, 24).view({2, 3, 4});
    ts::Tensor<int> m = ts::arange<int>(0, 12).view({4,3});
    std::cout << ts::einsum<int>("ijk,kl->ijl", {l, m}) << std::endl;
    ts::Tensor<double> A = ts::rand<double>({3,3});
    ts::Tensor<double> B = ts::rand<double>({3,3});
    ts::Tensor<double> C = ts::rand<double>({3,3});
    std::cout << ts::einsum<double>("ij,jk,kl->il", {A, B, C}) << std::endl;


    // ts::Tensor<double> D = ts::rand<double>({100,50,20,20});
    // std::cout<<D<<std::endl;



    return 0;
}
