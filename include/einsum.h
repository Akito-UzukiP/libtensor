#ifndef EINSUM_H
#define EINSUM_H
#include "tensor_basic.h"
#include "iterator.h"
#include "tensor_operation.h"
#include "tensor_calculation.h"
#include <string>
#include <map>
#include <cassert>

namespace ts{

    // 实现einsum的函数
    template <typename T>
    Tensor<T> einsum(const std::string& expr, const std::vector<Tensor<T>>& args) {
        std::map<char, int> dims; // 用于存储每个维度的大小
        std::map<char, int> cur_index; // 用于迭代的当前索引
        std::vector<char> dim_iter_order; // 用于迭代的维度顺序
        std::vector<std::vector<char>> input_dims; // 每个输入Tensor的维度对应的字母
        std::vector<char> output_part; // 输出Tensor的维度对应的字母
        bool is_input_part = true;
        bool is_scalar_output = false;
        std::vector<char> input_part;
        for(char c : expr) {
            if(c == ' '){
                continue;
            }
            if(is_input_part){
                if(c == '-' || c == ','){
                    input_dims.push_back(input_part);
                    input_part.clear();
                }else if(c == '>'){
                    is_input_part = false;
                }else{
                    input_part.push_back(c);
                }
            }else{
                output_part.push_back(c);
            }
        }
        if(output_part.size() == 0){
            is_scalar_output = true;
        }

        for(int i = 0; i < input_dims.size(); i++){
            for(int j = 0;j<input_dims[i].size();j++){
                char c = input_dims[i][j];
                if(dims.find(c) == dims.end()){
                    dim_iter_order.push_back(c);
                    dims[c] = args[i].shape()[j];
                    cur_index[c] = 0;
                }else{
                    assert(dims[c] == args[i].shape()[j]);
                }
            }
        }
        // 在这里可以创建output矩阵
        std::vector<int> output_shape;
        for(char c : output_part){
            if(dims.find(c) == dims.end()){
                assert(is_scalar_output);
            }else{
                output_shape.push_back(dims[c]);
            }
        }
        Tensor<T> output = zeros<T>(output_shape);


        //std::vector<bool> dim_entered(input_dims.size(), false);
        bool done = false;
        while(!done){
            // 创建一个用于存储当前索引的vector
            std::vector<int> index_for_each_input;

            // 对每个输入Tensor计算索引
            // for(const auto& input_dim : input_dims){
            //     std::vector<int> indices;
            //     for(char dim : input_dim){
            //         indices.push_back(cur_index[dim]);
            //     }
            //     index_for_each_input.push_back(tensor(indices)); // 使用indices获取tensor的值
            // }
            for(int i = 0; i < input_dims.size(); i++){
                std::vector<int> indices;
                for(char dim : input_dims[i]){
                    indices.push_back(cur_index[dim]);
                }
                index_for_each_input.push_back(args[i](indices)); // 使用indices获取tensor的值
            }

            // 执行计算
            T temp_result = 1; // 假设是乘法操作
            for(auto value : index_for_each_input){
                temp_result *= value;
            }

            // 根据output_part计算输出索引
            std::vector<int> output_indices;
            for(char dim : output_part){
                output_indices.push_back(cur_index[dim]);
            }

            // 将结果累加到输出Tensor的相应位置
            output(output_indices) += temp_result;

            // 更新索引
            for(int i = dim_iter_order.size() - 1; i >= 0; i--){
                if(cur_index[dim_iter_order[i]] < dims[dim_iter_order[i]] - 1){
                    cur_index[dim_iter_order[i]]++;
                    break;
                }else{
                    cur_index[dim_iter_order[i]] = 0;
                    if(i == 0){
                        done = true;
                    }
                }
            }
        }

        // 此时 output Tensor 包含了结果






        // ... 根据dims和args实现einsum逻辑
        // 创建输出张量
        // 迭代所有索引组合
        // 计算并填充输出张量
        // 返回输出张量
        return output;
    }

    
}

#endif