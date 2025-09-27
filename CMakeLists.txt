cmake_minimum_required(VERSION 3.10.0)
project(Task3 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)


add_executable(Task3 main.cpp)

target_include_directories(Task3 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(Task3 PRIVATE ${OpenCV_LIBS} Ceres::ceres)

if (MSVC)
    target_compile_options(Task3 PRIVATE /W4)
else()
    target_compile_options(Task3 PRIVATE -Wall -Wextra -Wpedantic -O2)
endif()