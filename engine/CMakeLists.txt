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

function(flint_configure_cuda_target target)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${FLINT_CUDA_ARCH},code=sm_${FLINT_CUDA_ARCH}>")
    endif()
endfunction()

add_library(smoke_op OBJECT src/ops/smoke_add.cu)
set_target_properties(smoke_op PROPERTIES POSITION_INDEPENDENT_CODE ON)
flint_configure_cuda_target(smoke_op)

add_library(engine_loader_obj OBJECT src/loader/safetensors.cpp src/loader/arenas.cpp)
set_target_properties(engine_loader_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(engine_loader_obj PRIVATE src)
target_link_libraries(engine_loader_obj PRIVATE CUDA::cudart)
add_library(engine_loader STATIC $<TARGET_OBJECTS:engine_loader_obj>)
target_include_directories(engine_loader PUBLIC src)
target_link_libraries(engine_loader PUBLIC CUDA::cudart)

add_library(engine_ops OBJECT
    src/ops/exl3_linear.cu src/ops/norm_embed.cu src/ops/sconv.cu
    src/ops/dspark.cu src/ops/attention.cu src/ops/moe.cu src/ops/loop_res.cu
    src/ops/gemm_cublas.cpp)
set_target_properties(engine_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(engine_ops PUBLIC src)
target_link_libraries(engine_ops PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(engine_ops PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=false>")
flint_configure_cuda_target(engine_ops)

add_library(engine_model OBJECT src/model/dspark_forward.cpp src/model/inkling2_forward.cpp src/model/submodules.cpp)
set_target_properties(engine_model PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(engine_model PUBLIC engine_ops CUDA::cudart CUDA::cublas)

add_library(engine STATIC src/runtime/engine_api.cpp src/runtime/engine_runtime.cpp
    src/runtime/sampler.cpp src/runtime/prefix_cache.cpp
    $<TARGET_OBJECTS:smoke_op> $<TARGET_OBJECTS:engine_loader_obj>
    $<TARGET_OBJECTS:engine_ops> $<TARGET_OBJECTS:engine_model>)
set_target_properties(engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(engine PUBLIC include PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(engine PRIVATE Threads::Threads CUDA::cublas CUDA::cudart)

add_library(forward SHARED src/forge/forward_entry.cpp)
set_target_properties(forward PROPERTIES PREFIX "" OUTPUT_NAME "forward")
target_include_directories(forward PRIVATE src)
target_link_libraries(forward PRIVATE engine_model engine_ops CUDA::cudart)
add_library(stub_forward SHARED src/forge/stub_forward_entry.cpp src/ops/stub_layers.cu $<TARGET_OBJECTS:smoke_op>)
set_target_properties(stub_forward PROPERTIES PREFIX "" OUTPUT_NAME "stub_forward")
target_include_directories(stub_forward PRIVATE src)
target_link_libraries(stub_forward PRIVATE CUDA::cudart)
flint_configure_cuda_target(stub_forward)

# Keep CUDA source-line attribution for nsys/ncu without bloating Release binaries.
foreach(target engine_ops forward)
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
endforeach()

add_library(forge_smoke STATIC src/forge/forge_smoke.cpp)
target_include_directories(forge_smoke PUBLIC src)
target_link_libraries(forge_smoke PRIVATE engine_loader CUDA::cudart ${CMAKE_DL_LIBS})

add_library(forge_replay STATIC
    src/forge/forward_module.cpp src/forge/replay_metrics.cpp
    src/forge/replay.cpp src/runtime/sampler.cpp src/forge/safetensors_writer.cpp)
target_link_libraries(forge_replay PUBLIC engine_loader PRIVATE ${CMAKE_DL_LIBS})
add_executable(forge-host src/forge/forge_host.cpp src/forge/forge_daemon.cpp)
target_link_libraries(forge-host PRIVATE forge_smoke forge_replay)
add_dependencies(forge-host forward stub_forward)

enable_testing()
add_executable(engine_tests tests/test_engine_smoke.cpp)
target_link_libraries(engine_tests PRIVATE engine forge_smoke)
add_dependencies(engine_tests stub_forward)
add_test(NAME engine_smoke COMMAND engine_tests --forward $<TARGET_FILE:stub_forward>)

add_executable(loader_tests tests/test_loader.cpp)
target_link_libraries(loader_tests PRIVATE engine_loader)
add_test(NAME loader_tests COMMAND loader_tests)

add_executable(forge_tests tests/test_forge_daemon.cpp)
target_link_libraries(forge_tests PRIVATE forge_replay)
add_dependencies(forge_tests forge-host)
add_test(NAME forge_tests COMMAND forge_tests $<TARGET_FILE:forge-host> $<TARGET_FILE:stub_forward>)
set_tests_properties(forge_tests PROPERTIES TIMEOUT 60)

add_executable(ops_tests tests/test_ops_golden.cpp)
target_link_libraries(ops_tests PRIVATE engine_model engine_ops engine_loader)
add_test(NAME ops_tests COMMAND ops_tests)

add_executable(e2e_tests tests/test_e2e_golden.cpp)
target_link_libraries(e2e_tests PRIVATE engine_loader ${CMAKE_DL_LIBS})
add_dependencies(e2e_tests forward)
add_test(NAME e2e_tests COMMAND e2e_tests --forward $<TARGET_FILE:forward>)

add_executable(sampler_tests tests/test_sampler.cpp src/runtime/sampler.cpp)
target_include_directories(sampler_tests PRIVATE src)
add_test(NAME sampler_tests COMMAND sampler_tests)
add_executable(prefix_cache_tests tests/test_prefix_cache.cpp src/runtime/prefix_cache.cpp)
target_include_directories(prefix_cache_tests PRIVATE src)
target_link_libraries(prefix_cache_tests PRIVATE engine_loader)
add_test(NAME prefix_cache_tests COMMAND prefix_cache_tests)
add_executable(runtime_tests tests/test_runtime_gpu.cpp)
target_include_directories(runtime_tests PRIVATE src)
target_link_libraries(runtime_tests PRIVATE engine CUDA::cudart)
add_test(NAME runtime_tests COMMAND runtime_tests)

add_executable(mtp_tests tests/test_mtp_golden.cpp)
target_include_directories(mtp_tests PRIVATE src)
target_link_libraries(mtp_tests PRIVATE engine CUDA::cudart CUDA::cublas)
add_test(NAME mtp_tests COMMAND mtp_tests)
