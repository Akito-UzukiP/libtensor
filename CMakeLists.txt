# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.0)

# Project's name
project(libtensor)

# Set the C++ standard you are using (e.g., C++11, C++14, C++17, C++20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS -O3)

# The following folder will be included
include_directories(${PROJECT_SOURCE_DIR}/include)

# Can manually add the sources using the set command as follows:
# set(SOURCES src/main.cpp src/other_source.cpp)

# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

# Add executable based on the source files
add_executable(libtensor ${SOURCES})

target_compile_options(libtensor PRIVATE -mavx -mavx2)

# If you have any libraries to link (e.g., if you have some libs in a 'lib' directory)
# target_link_libraries(YourExecutableName ${PROJECT_SOURCE_DIR}/lib/your_lib.a)
