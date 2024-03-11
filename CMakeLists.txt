cmake_minimum_required(VERSION 3.5.0)
project(cugrad VERSION 0.1.0 LANGUAGES CUDA CXX)

include_directories(${CMAKE_SOURCE_DIR}/include)

# cugrad lib
add_library(cugrad SHARED src/cugrad.cpp src/kernels.cu src/tensor.cu)
set_target_properties(cugrad PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# examples/linear_neural_net
add_executable(linear_neural_net examples/linear_neural_net/main.cpp)
target_link_libraries(linear_neural_net cugrad)

# examples/partials
add_executable(partials examples/partials/main.cpp)
target_link_libraries(partials cugrad)

# examples/raw_tensor
add_executable(raw_tensor examples/raw_tensor/main.cpp)
target_link_libraries(raw_tensor cugrad)

# examples/raw_tensor_gpu
add_executable(raw_tensor_gpu examples/raw_tensor_gpu/main.cpp)
target_link_libraries(raw_tensor_gpu cugrad)

# examples/summation
add_executable(summation examples/summation/main.cpp)
target_link_libraries(summation cugrad)