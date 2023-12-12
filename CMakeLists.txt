cmake_minimum_required(VERSION 3.10)
project(LibTensor VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 包含的头文件目录
include_directories(include)

# 启用测试
enable_testing()

# 添加测试可执行文件
add_executable(test_tensor tests/test_tensor.cpp)

# 如果您有其他库要链接到测试可执行文件，请在这里添加
# 例如，如果您使用了外部测试框架，可能需要链接该框架到 test_tensor
# target_link_libraries(test_tensor PRIVATE some_other_libraries)

# 添加测试
add_test(NAME TestTensor COMMAND test_tensor)
