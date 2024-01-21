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

    // 测试使用初始化列表构造函数
    ts::Tensor<float> tensorWithInitList({1, 2, 3, 4, 5, 6}, {2, 3});
    assert(tensorWithInitList.total_size() == 6);
    assert(tensorWithInitList.shape() == std::vector<int>({2, 3}));

    std::cout<< "随机数测试："<<std::endl;
    ts::Tensor<float> tensor4 = ts::rand<float>({2,3,4});
    std::cout<<tensor4<<std::endl;

    std::cout<< "全零、全一、全3.14测试："<<std::endl;
    ts::Tensor<float> tensor5 = ts::zeros<float>({2,3,4});
    std::cout<<tensor5<<std::endl;
    ts::Tensor<float> tensor6 = ts::ones<float>({4,5,6});
    std::cout<<tensor6<<std::endl;
    ts::Tensor<float> tensor7 = ts::full<float>({4,5,6}, 3.14);
    std::cout<<tensor7<<std::endl;

    std::cout<< "单位矩阵测试："<<std::endl;
    ts::Tensor<double> tensor8 = ts::eye<double>(5);
    std::cout<<tensor8<<std::endl;
    ts::Tensor<int> tensor9 = ts::eye<int>(5,3);
    std::cout<<tensor9<<std::endl;
    //t(1);

    return;
}

void elementAccessTest() {// Item2:测试元素访问, Item3:证明元素访问是返回引用
    // 创建Tensor对象
    float data[] = {1, 2, 3, 4, 5, 6};
    ts::Tensor<float> tensor(data, {2, 3});
    std::cout<< "测试元素访问："<<std::endl;
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
    std::cout<< "位于(1,2)的元素："<<std::endl;
    float* a = &tensor(1, 2);
    std::cout<<a<<std::endl;
    std::cout<<&tensor.data_ptr()[5]<<std::endl;

    // slice
    std::cout<< "slice测试："<<std::endl;
    ts::Tensor<float> tensor5 = ts::slice<float>(tensor,{{0, 2}, {1, 3}});
    std::cout<<tensor5<<std::endl;
    std::cout<<&tensor5.data_ptr()[0]<<std::endl;
    std::cout<<&tensor.data_ptr()[0]<<std::endl;

    // 测试stride函数
    // 假设连续内存布局, strides应为 {3, 1} (对于2x3的Tensor)
    std::string expectedStride = "[3,1]";
    assert(tensor.stride() == expectedStride);
    ts::Tensor<float> tensor2 = tensor.transpose(0, 1);
    std::cout<< "congtiguous测试："<<std::endl;
    assert(tensor2.stride() == "[1,3]");
    std::cout<<tensor2<<std::endl;
    ts::Tensor<float> tensor3 = tensor2.contiguous();
    std::cout<<tensor3.stride()<<std::endl;
    assert(tensor3.stride() == "[2,1]");



    return;
}

void testMutate(){
    ts::Tensor<double> t = ts::Tensor<double>({0.1,1.2,3.4,5.6,7.8,2.2,3.1,4.5,6.7,8.9,4.9,5.2,6.3,7.4,8.5},{3,5});
    std::cout<<"原始矩阵："<<std::endl;
    std::cout<<t<<std::endl;
    std::cout<<t.data_ptr()<<std::endl;
    t(1) = 1;
    t(2,{2,4}) = {1,2};
    std::cout<<"修改后矩阵："<<std::endl;
    std::cout<<t<<std::endl;
    std::cout<<t.data_ptr()<<std::endl;
}

void testTensorOperations(){
    ts::Tensor<int> t1 = ts::ones<int>({2,2,3});
    ts::Tensor<int> t2 = ts::zeros<int>({1,2,3});
    ts::Tensor<int> t3 = ts::concat<int>({t1,t2},0);
    std::cout<<"由t1和t2拼接而成的t3："<<std::endl;
    std::cout<<t3<<std::endl;
    std::cout<<t3.data_ptr()<<std::endl;
    ts::Tensor<int> t4 = t3.transpose(0,1);
    ts::Tensor<int> t5 = t3.permute({2,1,0});
    std::cout<<"t3的转置："<<std::endl;
    std::cout<<t4<<std::endl;
    std::cout<<t4.data_ptr()<<std::endl;
    std::cout<<"t3的置换："<<std::endl;
    std::cout<<t5<<std::endl;
    std::cout<<t5.data_ptr()<<std::endl;
    std::cout<<"t3的view："<<std::endl;
    ts::Tensor<int> t6 = t3.view({6,3});
    std::cout<<t6<<std::endl;
    std::cout<<t6.data_ptr()<<std::endl;
    std::cout<<"t5的view(由于内存不连续，复制并contiguous后再view)："<<std::endl;
    ts::Tensor<int> t7 = t5.view({6,3});
    std::cout<<t7<<std::endl;
    std::cout<<t7.data_ptr()<<std::endl;


}

void testBasicArithmeticOperations() {
    ts::Tensor<double> a = ts::arange<double>(1, 1+2*2*3*4).view({2,2,3,4});
    ts::Tensor<double> b = ts::arange<double>(2, 2+2*3*4).view({2,3,4}); // Broadcast
    std::cout<<"原始矩阵："<<std::endl;
    std::cout<<a<<std::endl;
    std::cout<<b<<std::endl;
    ts::Tensor<double> c = a + b;
    ts::Tensor<double> d = a - b;
    ts::Tensor<double> e = a * b;
    ts::Tensor<double> f = a / b;
    std::cout<<"加法："<<std::endl;
    std::cout<<c<<std::endl;
    std::cout<<"减法："<<std::endl;
    std::cout<<d<<std::endl;
    std::cout<<"乘法："<<std::endl;
    std::cout<<e<<std::endl;
    std::cout<<"除法："<<std::endl;
    std::cout<<f<<std::endl;
    // ts::Tensor<int> g = ts::add<int>(a, b);
    // ts::Tensor<int> h = ts::sub<int>(a, b);
    // ts::Tensor<int> i = ts::mul<int>(a, b);
    // ts::Tensor<int> j = ts::div<int>(a, b);
    // ts::Tensor<int> k = a.add(b);
    // ts::Tensor<int> l = a.sub(b);
    // ts::Tensor<int> m = a.mul(b);
    // ts::Tensor<int> n = a.div(b);

    
    // 确认加法结果
    // 其他算术操作...
}

void testMatmulOperations() {
    std::cout<<"测试矩阵乘法："<<std::endl;
    ts::Tensor<int> a = ts::arange<int>(0, 2*3*4*5).view({2, 3, 4, 5});
    ts::Tensor<int> b = ts::arange<int>(0, 2*3*4*5).view({2, 3, 5, 4});
    ts::Tensor<int> c = ts::matmul<int>(a, b);
    std::cout<<c<<std::endl;
}


void testEinsum(){
    ts::Tensor< int> t1 = ts::Tensor< int>({1,2,3,4,5,6},{3,2});
    ts::Tensor< int> t2 = ts::Tensor< int>({1,2,3,4,5,6},{2,3});
    ts::Tensor< int> t3 = ts::Tensor< int>({1,2,3},{3});
    ts::Tensor< int> t4 = ts::Tensor< int>({4,5,6},{3});
    ts::Tensor< int> t5 = ts::Tensor< int>({7,8},{2});
    ts::Tensor< int> t6 = ts::full< int>({3,3}, 1);

    std::cout<<" 1) Extracting elements along diagonal, ‘ii->i’" <<std::endl;
    std::cout << ts::einsum< int>("ii->i", {t6}) << std::endl;

    std::cout<<"2) Transpose, 'ij->ji'" <<std::endl;
    std::cout<<t1<<std::endl;
    std::cout << ts::einsum< int>("ij->ji", {t1}) << std::endl;

    std::cout<<"3) Permutation, 'abcde->acdeb'" <<std::endl;
    ts::Tensor<int> t99 = ts::arange<int>(0,1*2*3*4*5).view({1, 2, 3, 4, 5});
    std::cout<<t99.size()<<std::endl;
    ts::Tensor<int> t992 =  ts::einsum< int>("abcde->acdeb", {t99});
    std::cout<<t992.size()<<std::endl;

    std::cout<<"4) Sum along dimension, ‘ij->’" <<std::endl;
    std::cout << ts::einsum< int>("ij->", {t1}) << std::endl;

    std::cout<<"5) Column sum, ‘ij->j’" <<std::endl;
    std::cout << ts::einsum< int>("ij->j", {t1}) << std::endl;

    std::cout<<"6) Row sum, ‘ij->i’" <<std::endl;
    std::cout << ts::einsum< int>("ik,k->i", {t1, t5}) << std::endl;

    std::cout<<"7) Matrix vector multiplication, ‘ij,j->i’" <<std::endl;
    std::cout << ts::einsum< int>("ik,kj->ij", {t1, t2}) << std::endl;

    std::cout<<"8) Matrix matrix multiplication, ‘ij,jk->ik’" <<std::endl;
    std::cout << ts::einsum< int>("i,i->", {t3, t4}) << std::endl;

    std::cout<<"9) Vector outer product, ‘i,j->ij’" <<std::endl;
    t2 = ts::arange<int>(0,6).view({3, 2});
    std::cout << ts::einsum< int>("ij,ij->", {t1, t2}) << std::endl;

    std::cout<<"10) Vector dot product, ‘i,i->’" <<std::endl;
    std::cout << ts::einsum< int>("i,j->ij", {t3, t4}) << std::endl;

    std::cout<<"11) Tensor contraction, ‘ij,jk->ik’" <<std::endl;
    t6 = ts::arange<int>(0,12).view({3,2,2});
    ts::Tensor< int> t7 = ts::arange< int>(0,3*2*5).view({3, 2, 5});  
    std::cout << ts::einsum< int>("ijk,ikl->ijl", {t6, t7}) << std::endl;

    std::cout<<"12) Tensor contraction, ‘pqrs,tuqvr->pstuv’" <<std::endl;
    ts::Tensor< int> t8 = ts::arange< int>(0,120).view({2, 3, 4, 5});  
    ts::Tensor< int> t9 = ts::arange< int>(0,6*3*3*4*4).view({6, 3, 3, 4, 4});  
    std::cout << ts::einsum< int>("pqrs,tuqvr->pstuv", {t8, t9}) << std::endl; // p=2,q=3,r=4,s=5,t=6,u=3,v=4

    std::cout<<"13) Tensor contraction, ‘ij,jk->ik’" <<std::endl;
    ts::Tensor< int> t10 = ts::arange< int>(0,4*6).view({4, 6});  
    ts::Tensor< int> t11 = ts::arange< int>(0,5*6*7).view({5, 6, 7});  
    ts::Tensor< int> t12 = ts::arange< int>(0,4*7).view({4, 7});
    std::cout << ts::einsum< int>("ik,jkl,il->ij", {t10, t11, t12}) << std::endl;
}




void testSerialization(){
    ts::Tensor< double> t1 = ts::Tensor<double>({0.1,1.2,2.2,3.1,4.9,5.2},{3,2});
    ts::serialize(t1, "t1.bin");
    ts::Tensor<double> t2 = ts::deserialize<double>("t1.bin");
    std::cout<<"序列化前后是否相等："<<std::endl;
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
    std::cout<<" 第一部分：测试构造函数"<<std::endl;
    testConstructionAndAssignment(); // 1
    std::cout<<" 第二部分：测试元素访问"<<std::endl;
    elementAccessTest(); // 2 3 4 5
    std::cout<<" 第三部分：测试Mutate操作"<<std::endl;
    testMutate(); // 6 7
    std::cout<<" 第四部分：测试Transpose,Permute,Contiguous操作"<<std::endl;
    testTensorOperations(); // 8 9 10 11  注意：如果先transpose/permute了再view，需要先contiguous，此时内存不相同
    std::cout<<" 第五部分：测试基本算术操作"<<std::endl;
    testBasicArithmeticOperations(); //12 实现了broadcast
    std::cout<<" 第六部分：测试矩阵乘法操作"<<std::endl;
    testMatmulOperations(); //14 实现了broadcast
    std::cout<<" 第七部分：测试比较操作"<<std::endl;
    testComparasions();
    std::cout<<" 第八部分：测试einsum操作"<<std::endl;
    testEinsum();
    std::cout<<" 第九部分：测试序列化和反序列化操作"<<std::endl;
    testSerialization();
    return 0;
}
