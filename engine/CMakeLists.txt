cmake_minimum_required(VERSION 3.22)

# Capture the CLI/cache value before enabling CUDA (3.22 validates it there).
set(FLINT_CUDA_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
if(NOT FLINT_CUDA_ARCH)
    set(FLINT_CUDA_ARCH 120a)
endif()
if(CMAKE_VERSION VERSION_LESS 3.27 OR FLINT_CUDA_ARCH MATCHES "[a-z]$")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
else()
    set(CMAKE_CUDA_ARCHITECTURES "${FLINT_CUDA_ARCH}")
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
project(flint_engine LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
# CMake 3.22's NVIDIA compiler table stops at C++17. NVCC 12+ supports C++20.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12
   AND NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
endif()
set(CMAKE_BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_library(smoke_op OBJECT src/ops/smoke_add.cu)
set_target_properties(smoke_op PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    target_compile_options(smoke_op PRIVATE
        "-gencode=arch=compute_${FLINT_CUDA_ARCH},code=sm_${FLINT_CUDA_ARCH}")
endif()

add_library(engine_loader STATIC src/loader/safetensors.cpp src/loader/arenas.cpp)
set_target_properties(engine_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(engine_loader PUBLIC src)
target_link_libraries(engine_loader PUBLIC CUDA::cudart)

add_library(engine STATIC src/runtime/engine_api.cpp $<TARGET_OBJECTS:smoke_op>)
set_target_properties(engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(engine PUBLIC include PRIVATE src)
target_link_libraries(engine PRIVATE engine_loader CUDA::cudart)

add_library(forward SHARED src/model/forward_entry.cpp $<TARGET_OBJECTS:smoke_op>)
set_target_properties(forward PROPERTIES PREFIX "" OUTPUT_NAME "forward")
target_include_directories(forward PRIVATE src)
target_link_libraries(forward PRIVATE CUDA::cudart)

add_library(forge_smoke STATIC src/forge/forge_smoke.cpp)
target_include_directories(forge_smoke PUBLIC src)
target_link_libraries(forge_smoke PRIVATE engine_loader CUDA::cudart ${CMAKE_DL_LIBS})

add_executable(forge-host src/forge/forge_host.cpp)
target_link_libraries(forge-host PRIVATE forge_smoke)
add_dependencies(forge-host forward)

enable_testing()
add_executable(engine_tests tests/test_engine_smoke.cpp)
target_link_libraries(engine_tests PRIVATE engine forge_smoke)
add_dependencies(engine_tests forward)
add_test(NAME engine_smoke COMMAND engine_tests --forward $<TARGET_FILE:forward>)

add_executable(loader_tests tests/test_loader.cpp)
target_link_libraries(loader_tests PRIVATE engine_loader)
add_test(NAME loader_tests COMMAND loader_tests)
