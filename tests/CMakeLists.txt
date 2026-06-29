# Standalone CMake build for pipeline_runner, cross-compiled to wasm32.
# Configured by scripts/05_build_pipeline_runner.sh via emcmake against the
# stage-2 wasm MLIR install and the stage-3 StableHLO wasm build.
#
#   find_package(MLIR) resolves the full upstream MLIR/LLVM static-lib closure
#   through imported targets; the StableHLO libs are added explicitly (StableHLO
#   ships no CMake package config, so we point at its build/install tree).
cmake_minimum_required(VERSION 3.20)
project(pipeline_runner C CXX)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

# StableHLO locations passed in by the build script.
#   STABLEHLO_SOURCE_DIR : repo root (public headers)
#   STABLEHLO_BUILD_DIR  : cmake build tree (generated headers + static libs)
if(NOT STABLEHLO_SOURCE_DIR OR NOT STABLEHLO_BUILD_DIR)
  message(FATAL_ERROR "pass -DSTABLEHLO_SOURCE_DIR=... -DSTABLEHLO_BUILD_DIR=...")
endif()

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${STABLEHLO_SOURCE_DIR}
  ${STABLEHLO_BUILD_DIR}
)

add_executable(pipeline_runner pipeline_runner.c)

# StableHLO static libs (order matters for static linking: these before MLIR).
# pipeline_runner.c uses mlirRegisterAllStablehloPasses + mlirGetDialectHandle__
# stablehlo__ (both in StablehloCAPI) AND mlirGetDialectHandle__chlo__ (in the
# SEPARATE ChloCAPI — verified in stablehlo/integrations/c/CMakeLists.txt).
find_library(LIB_SHLO_CAPI   NAMES StablehloCAPI             PATHS ${STABLEHLO_BUILD_DIR} PATH_SUFFIXES lib REQUIRED NO_DEFAULT_PATH)
find_library(LIB_CHLO_CAPI   NAMES ChloCAPI                  PATHS ${STABLEHLO_BUILD_DIR} PATH_SUFFIXES lib REQUIRED NO_DEFAULT_PATH)
find_library(LIB_SHLO_LINALG NAMES StablehloLinalgTransforms PATHS ${STABLEHLO_BUILD_DIR} PATH_SUFFIXES lib REQUIRED NO_DEFAULT_PATH)
find_library(LIB_SHLO_OPS    NAMES StablehloOps              PATHS ${STABLEHLO_BUILD_DIR} PATH_SUFFIXES lib REQUIRED NO_DEFAULT_PATH)
find_library(LIB_CHLO_OPS    NAMES ChloOps                   PATHS ${STABLEHLO_BUILD_DIR} PATH_SUFFIXES lib REQUIRED NO_DEFAULT_PATH)

target_link_libraries(pipeline_runner PRIVATE
  ${LIB_SHLO_CAPI}
  ${LIB_CHLO_CAPI}
  ${LIB_SHLO_LINALG}
  ${LIB_SHLO_OPS}
  ${LIB_CHLO_OPS}
  # Upstream MLIR C-API: registration + IR + pass management. find_package(MLIR)
  # pulls the transitive LLVM/MLIR static-lib closure through these targets.
  MLIRCAPIRegisterEverything
  MLIRCAPIIR
)

# emscripten: let host libm resolve fmaxf/expf/tanhf at instantiate time
# (softmax/gelu emit these). See docs/STATUS.md "libm" note.
target_link_options(pipeline_runner PRIVATE
  "-sALLOW_TABLE_GROWTH" "-sWASM_BIGINT" "-sASSERTIONS"
  "-sNO_DISABLE_EXCEPTION_CATCHING" "-sALLOW_MEMORY_GROWTH"
  "-sNODERAWFS=1" "-sSTACK_SIZE=4MB"
)
