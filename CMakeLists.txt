cmake_minimum_required (VERSION 3.14)
project (liftol)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(FetchContent)
include(ExternalProject)

find_package(OpenMP)
find_package(MKL REQUIRED)
find_package(JeMalloc REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${MKL_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})
include_directories(${JEMALLOC_INCLUDE_DIR})
# include_directories(SYSTEM src/competitor/hot/src/libs/hot/commons/include)
# include_directories(SYSTEM src/competitor/hot/src/libs/hot/rowex/include)
# include_directories(SYSTEM src/competitor/hot/src/libs/hot/single-threaded/include)
# include_directories(SYSTEM src/competitor/hot/src/libs/idx/content-helpers/include)

# include_directories("/home/wanghui/gre/GRE/eigen-3.4.0")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core)

add_compile_options(-faligned-new -march=native -mavx2 -g -O3)


add_executable(example_liftol
        ${CMAKE_CURRENT_SOURCE_DIR}/src/examples/example_liftol.cpp
    )

target_link_libraries(example_liftol PUBLIC OpenMP::OpenMP_CXX ${JEMALLOC_LIBRARIES} ${MKL_LIBRARIES} ${TBB_LIBRARIES})
