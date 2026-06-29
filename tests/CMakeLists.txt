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

# StableHLO static libs by DIRECT path. (find_library is unreliable here: the
# emscripten toolchain sets CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY, so it ignores
# absolute PATHS outside the sysroot.) Order matters for static linking: these
# before MLIR. pipeline_runner.c uses mlirRegisterAllStablehloPasses +
# mlirGetDialectHandle__stablehlo__ (StablehloCAPI) AND mlirGetDialectHandle__
# chlo__ (the SEPARATE ChloCAPI — see stablehlo/integrations/c/CMakeLists.txt).
set(SHLO_LIB "${STABLEHLO_BUILD_DIR}/lib")
# mlirRegisterAllStablehloPasses() pulls the full pass set, and StablehloCAPI
# embeds the interpreter dialect, so link ALL StableHLO/Chlo/Vhlo archives. lld
# (wasm-ld) resolves archives iteratively, so order/over-inclusion is fine; it
# only pulls referenced objects. StablehloCAPI first so its registrations win.
file(GLOB SHLO_ALL_LIBS
  "${SHLO_LIB}/libStablehloCAPI.a"
  "${SHLO_LIB}/libChloCAPI.a"
  "${SHLO_LIB}/libStablehlo*.a"
  "${SHLO_LIB}/libChlo*.a"
  "${SHLO_LIB}/libVhlo*.a"
  "${SHLO_LIB}/libVersion.a")   # vhlo::Version (separate target named "Version")
target_link_libraries(pipeline_runner PRIVATE
  ${SHLO_ALL_LIBS}
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
  # Tolerate host-only libc symbols pulled from LLVM support (e.g. wait4) that
  # are never reached on the lowering path. Same approach eudsl uses for libm.
  "-sERROR_ON_UNDEFINED_SYMBOLS=0"
)
