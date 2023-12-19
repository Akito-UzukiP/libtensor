#include <iostream>
#include <vector>

// 基本情况：处理非数组类型
template <typename T>
void printArrayDimensions(const T& arr) {
    std::cout << "基本类型，非数组" << std::endl;
}

// 递归情况：处理数组类型
template <typename T>
void printArrayDimensions(const std::vector<T>& arr) {
    std::cout << "数组大小: " << arr.size() << std::endl;
    if (!arr.empty()) {
        printArrayDimensions(arr[0]);
    }
}

int main() {
    std::vector<std::vector<std::vector<int>>> arr = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    printArrayDimensions(arr);
    return 0;
}
