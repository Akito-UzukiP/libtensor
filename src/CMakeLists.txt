add_library(${PROJECT_NAME} SHARED "")

file(GLOB_RECURSE SOURCES *.cpp *.cxx *.cc *.c)
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
# 添加 main.cpp 文件
add_executable(main main.cpp)

# # 如果有库文件，链接它们
# target_link_libraries(main your_library)
