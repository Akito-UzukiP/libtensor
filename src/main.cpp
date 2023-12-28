#include <iostream>
#include "tensor.h"
#include <assert.h>
#include <immintrin.h>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <ctime>

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
    // srand(time(0));
    // std::vector<int> shape(5);
    // for (int i = 0; i < 5; ++i) {
    //     shape[i] = rand()%50 + 1;
    //     std::cout<<shape[i]<<std::endl;
    // }

    // ts::Tensor<double> a = ts::rand<double>(shape);
    // ts::Tensor<double> b = ts::rand<double>(shape);
    // ts::Tensor<double> c = ts::zeros<double>(shape);
    // auto start = std::chrono::high_resolution_clock::now();
    // typename ts::Tensor<double>::_Const_Iterator iter(a);
    // typename ts::Tensor<double>::_Const_Iterator iter2(b);
    // typename ts::Tensor<double>::_Iterator iter3(c);
    // int i = 0;
    // for (; !iter.done(); ++iter, ++iter2, ++iter3) {
    //     *iter3 = *iter * *iter2;
    //     ++i;
    // }
    // std::cout<<i<<std::endl;
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed2 = end - start;
    // std::cout << "Time taken by multiplication outside function: " << elapsed2.count() << " s\n";
    // start = std::chrono::high_resolution_clock::now();
    // c = a*b;
    // end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken by multiplication: " << elapsed.count() << " s\n";
    ts::Tensor<int> a = ts::arange<int>(1, 26).view({5, 5});
    ts::Tensor<int> b(a,false);
    b = b.transpose(0,1);
    //b = b.transpose(0,1);
    std::cout<<a * b<<std::endl;
    std::cout<<a - b<<std::endl;
    std::cout<<a + b<<std::endl;
    std::cout<<a / b<<std::endl;
    // std::cout<<a.matmul(b)<<std::endl;
    // typename ts::Tensor<double>::_Const_Iterator iter(a);

    // iter.reset();
    // iter2.reset();
    // iter3.reset();
    // start = std::chrono::high_resolution_clock::now();
    // alignas(32) double a1[8];
    // alignas(32) double b1[8];
    // alignas(32) double c1[8];
    // int i = 0;
    // for (; !iter.done(); ++iter, ++iter2) {
    //     a1[i] = *iter;
    //     b1[i] = *iter2;
    //     ++i;
    //     if (i == 8) {
    //         __m256d a2 = _mm256_load_pd(a1);
    //         __m256d b2 = _mm256_load_pd(b1);
    //         __m256d c2 = _mm256_mul_pd(a2, b2);
    //         _mm256_store_pd(c1, c2);
    //         for (int j = 0; j < 8; ++j) {
    //             *iter3 = c1[j];
    //             ++iter3;
    //         }
    //         i = 0;
    //     }
    // }

    // end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed3 = end - start;
    // std::cout << "Time taken by iterator with stride: " << elapsed3.count() << " s\n";
    // start = std::chrono::high_resolution_clock::now();
    // double da = 1.23456;
    // double db = 2.34567;
    // double dc;
    // for (int i = 0; i < 2097152; ++i) {
    //     dc = da * db;
    // }
    // end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed4 = end - start;
    // std::cout << "Time taken by double: " << elapsed4.count() << " s\n";

    // ts::Tensor<int> test = ts::arange<int>(0, 10).view({2, 5});
    // std::cout<< test * test <<std::endl;
    return 0;
}
