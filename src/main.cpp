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
    testConstructionAndAssignment();
    testElementAccess();
    testArithmeticOperations();
    testMatrixOperations();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
