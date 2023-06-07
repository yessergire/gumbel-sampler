cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)
set(CPM_DOWNLOAD_VERSION 0.37.0) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

project(Main)

add_executable(Main src/main.cpp)
add_executable(GumbelPlot src/plots.cpp)
add_executable(eXample src/accept-reject-example.cpp)
add_executable(Sampler src/sampler.cpp src/Hungarian.cpp)

set(SCIPLOT_BUILD_EXAMPLES OFF)
set(SCIPLOT_BUILD_TESTS OFF)
set(SCIPLOT_BUILD_DOCS OFF)

CPMAddPackage(
  NAME sciplot
  VERSION 0.3.1
  GITHUB_REPOSITORY "sciplot/sciplot"
)

if(sciplot)
  add_library(sciplot)	 
  target_include_directories(sciplot ${sciplot_SOURCE_DIR})	 
endif()

find_package(Boost 1.81 REQUIRED COMPONENTS program_options)

include_directories(${Boost_INCLUDE_DIR})
link_libraries(${Boost_LIBRARIES})

target_link_libraries(GumbelPlot sciplot Boost::program_options)
target_link_libraries(Main Boost::program_options)
target_link_libraries(eXample sciplot)