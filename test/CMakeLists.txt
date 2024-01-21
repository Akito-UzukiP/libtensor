include(FetchContent)

FetchContent_Declare(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG 0.7.7
)
FetchContent_Declare(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG 0.24.7
)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.13.0
)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(xtl)
FetchContent_MakeAvailable(xtensor)
FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(googletest)

set(TARGET benchmark)

file(GLOB_RECURSE TEST_SRC *.cpp)
add_executable(${TARGET} ${TEST_SRC})
target_link_libraries(
        ${TARGET}
        PRIVATE
        ${PROJECT_NAME}
        xtensor
        spdlog::spdlog
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(${TARGET})
