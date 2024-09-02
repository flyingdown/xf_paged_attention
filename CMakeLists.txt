# cmake .. -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc -DAMDGPU_TARGETS=gfx928 -DCMAKE_INSTALL_PREFIX=pa_install

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(paged-attention LANGUAGES CXX)

find_package(HIP REQUIRED)

file(
  GLOB_RECURSE
  PG_SOURCES_HIP
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/attention_kernels.cpp)

add_library(paged-attention SHARED
    paged_attn.cpp
    ${PG_SOURCES_HIP}
  )

target_include_directories(paged-attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention
    ${HIP_INCLUDE_DIRS})

target_link_libraries(paged-attention PRIVATE hip::device)

target_compile_options(paged-attention PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
  -Wno-deprecated-builtins
  -Wno-deprecated
  -DNDEBUG
  -DUSE_ROCM
  -U__HIP_NO_HALF_OPERATORS__
  -U__HIP_NO_HALF_CONVERSIONS__
  -fPIC
  -O3
  -std=c++17
  -D__HIP_PLATFORM_HCC__=1
  --offload-arch=gfx928
  #-mllvm 
  #-enable-num-vgprs-512=true
  >)

INSTALL(TARGETS paged-attention
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

INSTALL(FILES paged_attn.h DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
