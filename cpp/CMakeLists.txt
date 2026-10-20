cmake_minimum_required(VERSION 3.13)
project(a311y3_audio_sentinel C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O2 -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ALSA：采集链路依赖
find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA REQUIRED alsa)

# NPU 运行时：默认关闭。拿到 NNSDK2 后：
#   cmake -B build -DUSE_ADLA=ON -DNNSDK_ROOT=/path/to/nnsdk2
option(USE_ADLA "Link against NNSDK2 for on-device inference" OFF)

add_library(mel_frontend STATIC src/mel_frontend.c)
target_include_directories(mel_frontend PUBLIC src)
target_link_libraries(mel_frontend PUBLIC m)

add_executable(audio_sentinel src/main.cpp)
target_link_libraries(audio_sentinel PRIVATE mel_frontend ${ALSA_LIBRARIES})
target_include_directories(audio_sentinel PRIVATE ${ALSA_INCLUDE_DIRS})
target_link_directories(audio_sentinel PRIVATE ${ALSA_LIBRARY_DIRS})

if(USE_ADLA)
  if(NOT NNSDK_ROOT)
    message(FATAL_ERROR "USE_ADLA=ON 需要 -DNNSDK_ROOT=<path>")
  endif()
  target_compile_definitions(audio_sentinel PRIVATE USE_ADLA=1)
  target_include_directories(audio_sentinel PRIVATE ${NNSDK_ROOT}/include)
  target_link_directories(audio_sentinel PRIVATE ${NNSDK_ROOT}/lib)
  target_link_libraries(audio_sentinel PRIVATE nnsdk2)
  message(STATUS "ADLA3 推理已启用：${NNSDK_ROOT}")
else()
  message(STATUS "未启用 ADLA3，score 使用能量占位（仅打通链路与打点）")
endif()

# 前端自测：把 C 侧 mel 导出成文本，供 python/mel_parity.py --c-dump 比对
add_executable(mel_dump src/mel_dump.c)
target_link_libraries(mel_dump PRIVATE mel_frontend m)
