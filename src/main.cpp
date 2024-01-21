#include <iostream>
#include "../include/tensor.h"
#include <assert.h>
#include <immintrin.h>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include "./benchmark/api.hpp"
#include "gtest/gtest.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
void testConstructionAndAssignment() { //任务：1.测试构造函数
    // 测试默认构造函数
    ts::Tensor<float> defaultTensor;
    assert(defaultTensor.total_size() == 0);

    // 测试使用维度数组构造函数
    int dims[] = {2, 3};
    ts::Tensor<float> tensorWithDims(dims, 2);
    assert(tensorWithDims.total_size() == 6);
    assert(tensorWithDims.shape() == std::vector<int>({2, 3}));

    // 测试使用向量初始化构造函数
    std::vector<int> vecDims = {2, 3, 4};
    ts::Tensor<float> tensorWithVecDims(vecDims);
    assert(tensorWithVecDims.total_size() == 24);
    assert(tensorWithVecDims.shape() == vecDims);

    // 测试使用数据和维度列表构造函数
    float data[] = {1, 2, 3, 4, 5, 6};
    ts::Tensor<float> tensorWithData(data, {2, 3});
    assert(tensorWithData.total_size() == 6);
    assert(tensorWithData.shape() == std::vector<int>({2, 3}));
    assert(tensorWithData.data_ptr()[3] == data[3]); // 确认数据指针

    // 测试使用初始化列表构造函数
    ts::Tensor<float> tensorWithInitList({1, 2, 3, 4, 5, 6}, {2, 3});
    std::cout<<tensorWithInitList.total_size()<<std::endl;
    assert(tensorWithInitList.total_size() == 6);
    assert(tensorWithInitList.shape() == std::vector<int>({2, 3}));

    ts::Tensor<float> tensor4 = ts::rand<float>({2,3,4});
    std::cout<<tensor4<<std::endl;

    ts::Tensor<float> tensor5 = ts::zeros<float>({2,3,4});
    std::cout<<tensor5<<std::endl;
    ts::Tensor<float> tensor6 = ts::ones<float>({4,5,6});
    std::cout<<tensor6<<std::endl;
    ts::Tensor<float> tensor7 = ts::full<float>({4,5,6}, 3.14);
    std::cout<<tensor7<<std::endl;

    ts::Tensor<double> tensor8 = ts::eye<double>(5);
    std::cout<<tensor8<<std::endl;
    //t(1);

    return;
}

void elementAccessTest() {// Item2:测试元素访问, Item3:证明元素访问是返回引用
    // 创建Tensor对象
    float data[] = {1, 2, 3, 4, 5, 6};
    ts::Tensor<float> tensor(data, {2, 3});
    std::cout<<tensor<<std::endl;

    // 测试size函数
    std::string expectedSize = "[2,3]";
    assert(tensor.size() == expectedSize);

    // 测试shape函数
    std::vector<int> expectedShape = {2, 3};
    assert(tensor.shape() == expectedShape);

    // 测试total_size函数
    int expectedTotalSize = 6;
    assert(tensor.total_size() == expectedTotalSize);

    // 测试type函数
    std::string expectedType = typeid(float).name(); // 依赖于编译器具体的name mangling
    assert(tensor.type() == expectedType);

    // index
    float* a = &tensor(1, 2);
    std::cout<<a<<std::endl;
    std::cout<<&tensor.data_ptr()[5]<<std::endl;

    // slice
    ts::Tensor<float> tensor5 = ts::slice<float>(tensor,{{0, 2}, {1, 3}});
    std::cout<<tensor5<<std::endl;
    std::cout<<&tensor5.data_ptr()[0]<<std::endl;
    std::cout<<&tensor.data_ptr()[0]<<std::endl;

    // 测试stride函数
    // 假设连续内存布局, strides应为 {3, 1} (对于2x3的Tensor)
    std::string expectedStride = "[3,1]";
    assert(tensor.stride() == expectedStride);
    ts::Tensor<float> tensor2 = tensor.transpose(0, 1);
    assert(tensor2.stride() == "[1,3]");
    std::cout<<tensor2<<std::endl;
    ts::Tensor<float> tensor3 = tensor2.contiguous();
    std::cout<<tensor3.stride()<<std::endl;
    assert(tensor3.stride() == "[2,1]");



    return;
}

void testMutate(){
    ts::Tensor<double> t = ts::Tensor<double>({0.1,1.2,3.4,5.6,7.8,2.2,3.1,4.5,6.7,8.9,4.9,5.2,6.3,7.4,8.5},{3,5});
    std::cout<<t<<std::endl;
    std::cout<<t.data_ptr()<<std::endl;
    t(1) = 1;
    t(2,{2,4}) = {1,2};
    std::cout<<t<<std::endl;
    std::cout<<t.data_ptr()<<std::endl;
}

void testTensorOperations(){
    ts::Tensor<int> t1 = ts::ones<int>({2,2,3});
    ts::Tensor<int> t2 = ts::zeros<int>({1,2,3});
    ts::Tensor<int> t3 = ts::concat<int>({t1,t2},0);
    std::cout<<t3<<std::endl;
    std::cout<<t3.data_ptr()<<std::endl;
    ts::Tensor<int> t4 = t3.transpose(0,1);
    ts::Tensor<int> t5 = t3.permute({2,1,0});
    std::cout<<t4<<std::endl;
    std::cout<<t4.data_ptr()<<std::endl;
    std::cout<<t5<<std::endl;
    std::cout<<t5.data_ptr()<<std::endl;
    ts::Tensor<int> t6 = t3.view({6,3});
    std::cout<<t6<<std::endl;
    std::cout<<t6.data_ptr()<<std::endl;
    ts::Tensor<int> t7 = t5.view({6,3});
    std::cout<<t7<<std::endl;
    std::cout<<t7.data_ptr()<<std::endl;


}

void testBasicArithmeticOperations() {
    ts::Tensor<int> a = ts::arange<int>(1000, 1361).view({3,4,5,6});
    ts::Tensor<int> b = ts::arange<int>(2000, 2121).view({4,5,6});
    ts::Tensor<int> c = a + b;
    ts::Tensor<int> d = a - b;
    ts::Tensor<int> e = a * b;
    ts::Tensor<int> f = a / b;
    ts::Tensor<int> g = ts::add<int>(a, b);
    ts::Tensor<int> h = ts::sub<int>(a, b);
    ts::Tensor<int> i = ts::mul<int>(a, b);
    ts::Tensor<int> j = ts::div<int>(a, b);
    ts::Tensor<int> k = a.add(b);
    ts::Tensor<int> l = a.sub(b);
    ts::Tensor<int> m = a.mul(b);
    ts::Tensor<int> n = a.div(b);

    
    // 确认加法结果
    // 其他算术操作...
}

void testMathOperations() {
    ts::Tensor<double> t1 = ts::Tensor<double>({0.1,1.2,2.2,3.1,4.9,5.2},{3,2});
    ts::Tensor<double> t2 = ts::Tensor<double>({0.2,1.3,2.3,3.2,4.8,5.1},{3,2});
    std::cout<< t1 + t2 <<std::endl << ts::add<double>(t1,1) << std::endl;

    auto t = t1;
    std::cout<< ts::sum<double>(t,0) << std::endl << t.sum(1) << std::endl;
    ts::Tensor<double> t3 = ts::Tensor<double>({0.1,1.2,2.2,3.1,4.9,5.2}, {3,2});
    ts::Tensor<double> t4 = ts::Tensor<double>({0.2,1.2,2.2,3.2,4.8,5.2}, {3,2});
    std::cout << (t3 == t4) << std::endl;
}


void testEinsum(){
    ts::Tensor< int> t1 = ts::Tensor< int>({1,2,3,4,5,6},{3,2});
    ts::Tensor< int> t2 = ts::Tensor< int>({1,2,3,4,5,6},{2,3});
    ts::Tensor< int> t3 = ts::Tensor< int>({1,2,3},{3});
    ts::Tensor< int> t4 = ts::Tensor< int>({4,5,6},{3});
    ts::Tensor< int> t5 = ts::Tensor< int>({7,8},{2});
    ts::Tensor< int> t6 = ts::full< int>({3,3}, 1);

    // 1) Extracting elements along diagonal, ‘ii->i’
    std::cout << ts::einsum< int>("ii->i", {t6}) << std::endl;

    // 2) Transpose, ‘ij->ji’
    std::cout << ts::einsum< int>("ij->ji", {t1}) << std::endl;

    // 3) Permute, ‘ij->ji’
    ts::Tensor<int> t99 = ts::arange<int>(0,1*2*3*4*5).view({1, 2, 3, 4, 5});
    std::cout<<t99.size()<<std::endl;
    ts::Tensor<int> t992 =  ts::einsum< int>("abcde->acdeb", {t99});
    std::cout<<t992.size()<<std::endl;

    // 4) Reduce sum, ‘ij->’
    std::cout << ts::einsum< int>("ij->", {t1}) << std::endl;

    // 5) Sum along dimension, ‘ij->j’
    std::cout << ts::einsum< int>("ij->j", {t1}) << std::endl;

    // 6) Matrix and vector mul, ‘ik, k->i’
    std::cout << ts::einsum< int>("ik,k->i", {t1, t5}) << std::endl;

    // 7) Matrix mul, ‘ik, kj->ij’
    std::cout << ts::einsum< int>("ik,kj->ij", {t1, t2}) << std::endl;

    // 8) Dot product, ‘i,i->’
    std::cout << ts::einsum< int>("i,i->", {t3, t4}) << std::endl;

    // 9) Pointwise mul and reduce sum, ‘ij,ij->’
    t2 = ts::arange<int>(0,6).view({3, 2});
    std::cout << ts::einsum< int>("ij,ij->", {t1, t2}) << std::endl;

    // 10) Outer product, ‘i,j->ij’
    std::cout << ts::einsum< int>("i,j->ij", {t3, t4}) << std::endl;

    // 11) Batch matrix mul, ‘ijk,ikl->ijl’
    // Assuming t6 and t7 are 3D tensors
    t6 = ts::arange<int>(0,12).view({3,2,2});
    ts::Tensor< int> t7 = ts::arange< int>(0,3*2*5).view({3, 2, 5});  
    std::cout << ts::einsum< int>("ijk,ikl->ijl", {t6, t7}) << std::endl;

    // 12) Tensor contraction, ‘pqrs,tuqvr->pstuv’
    // Assuming t8 and t9 are higher order tensors
    ts::Tensor< int> t8 = ts::arange< int>(0,120).view({2, 3, 4, 5});  
    ts::Tensor< int> t9 = ts::arange< int>(0,6*3*3*4*4).view({6, 3, 3, 4, 4});  
    std::cout << ts::einsum< int>("pqrs,tuqvr->pstuv", {t8, t9}) << std::endl; // p=2,q=3,r=4,s=5,t=6,u=3,v=4

    // 13) Bilinear transformation, 'ik,jkl,il->ij' i = 4, j = 5, k = 6, l = 7
    ts::Tensor< int> t10 = ts::arange< int>(0,4*6).view({4, 6});  
    ts::Tensor< int> t11 = ts::arange< int>(0,5*6*7).view({5, 6, 7});  
    ts::Tensor< int> t12 = ts::arange< int>(0,4*7).view({4, 7});
    std::cout << ts::einsum< int>("ik,jkl,il->ij", {t10, t11, t12}) << std::endl;
}




void testSerialization(){
    ts::Tensor< double> t1 = ts::Tensor<double>({0.1,1.2,2.2,3.1,4.9,5.2},{3,2});
    ts::serialize(t1, "t1.bin");
    ts::Tensor<double> t2 = ts::deserialize<double>("t1.bin");
    std::cout<<(t1 == t2)<<std::endl;
}
void testComparasions() {
    ts::Tensor<double> t1 = ts::Tensor<double>({0.1,1.2,2.2,3.1,4.9,5.2},{3,2});
    ts::Tensor<double> t2 = ts::Tensor<double>({0.2,1.3,2.3,3.2,4.8,5.1},{3,2});
    ts::Tensor<bool> t3 = ts::eq<double>(t1, t2);
    std::cout<<"eq:"<<t3<<std::endl;
    ts::Tensor<bool> t4 = ts::ne<double>(t1, t2);
    std::cout<<"ne:"<<t4<<std::endl;
    ts::Tensor<bool> t5 = ts::gt<double>(t1, t2);
    std::cout<<"gt:"<<t5<<std::endl;
    ts::Tensor<bool> t6 = ts::ge<double>(t1, t2);
    std::cout<<"ge:"<<t6<<std::endl;
    ts::Tensor<bool> t7 = ts::lt<double>(t1, t2);
    std::cout<<"lt:"<<t7<<std::endl;
    ts::Tensor<bool> t8 = ts::le<double>(t1, t2);
    std::cout<<"le:"<<t8<<std::endl;

}


int main(){
    // testConstructionAndAssignment(); // 1
    // elementAccessTest(); // 2 3 4 5
    // testMutate(); // 6 7
    // testTensorOperations(); // 8 9 10 11  注意：如果先transpose/permute了再view，需要先contiguous，此时内存不相同
    // testBasicArithmeticOperations(); //12 实现了broadcast
    // testMathOperations(); //14 实现了broadcast
    // testEinsum();
    testSerialization();
    // testComparasions();
    return 0;
}
