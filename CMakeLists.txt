cmake_minimum_required(VERSION 3.16)
project(cv_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CVFS_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

add_library(cvfs src/image.cpp src/filters.cpp src/canny.cpp src/hough.cpp)
target_include_directories(cvfs PUBLIC include)

if(MSVC)
  target_compile_options(cvfs PRIVATE /W4)
else()
  target_compile_options(cvfs PRIVATE -Wall -Wextra -Wpedantic)
  if(CVFS_SANITIZE)
    target_compile_options(cvfs PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(cvfs PUBLIC -fsanitize=address,undefined)
  endif()
endif()

add_executable(cvfs_demo app/main.cpp)
target_link_libraries(cvfs_demo PRIVATE cvfs)

enable_testing()
add_executable(cvfs_tests tests/test_main.cpp)
target_link_libraries(cvfs_tests PRIVATE cvfs)
add_test(NAME unit_tests COMMAND cvfs_tests)
