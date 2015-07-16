cmake_minimum_required(VERSION 2.8)

project(hdf5)

find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(Boost REQUIRED COMPONENTS system)
find_package(MPI REQUIRED)

set (CMAKE_CXX_FLAGS "-std=c++11")

include_directories(  
    ${MPI_INCLUDE_PATH}
    ${HDF5_INCLUDE_DIRS}
    ${boost_INCLUDE_DIRS}
    )

include_directories(..)

add_executable(hdf5_tests test-hdf.cpp )
target_link_libraries(hdf5_tests ${HDF5_LIBRARIES} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY} )

