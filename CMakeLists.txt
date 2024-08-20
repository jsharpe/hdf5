cmake_minimum_required(VERSION 3.3)

project(hdf5)

macro(use_cxx11)
    set (CMAKE_CXX_STANDARD 11)
endmacro(use_cxx11)

use_cxx11()

find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(Boost REQUIRED COMPONENTS system)
find_package(MPI REQUIRED)
find_package(GTest REQUIRED)

enable_testing()

include_directories(
  ${GTEST_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
  ${MPI_INCLUDE_PATH}
  ${boost_INCLUDE_DIRS}
  )

include_directories(.)

add_executable(hdf5_tests test-hdf.cpp)

target_link_libraries(hdf5_tests
  ${Boost_LIBRARIES}
  ${GTEST_BOTH_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${MPI_C_LIBRARIES}
  )

add_test(NAME hdf5
	COMMAND hdf5_tests --gtest_output=xml)
