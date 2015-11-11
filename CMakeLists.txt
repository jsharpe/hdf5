cmake_minimum_required(VERSION 3.3)

project(hdf5)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

use_cxx11()

find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(Boost REQUIRED COMPONENTS system)
find_package(MPI REQUIRED)

include_directories(  
    ${MPI_INCLUDE_PATH}
    ${HDF5_INCLUDE_DIRS}
    ${boost_INCLUDE_DIRS}
    )

include_directories(..)

add_executable(hdf5_tests test-hdf.cpp )
target_link_libraries(hdf5_tests ${HDF5_LIBRARIES} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES})

add_test(NAME hdf5
	COMMAND hdf5_tests)

enable_testing()
