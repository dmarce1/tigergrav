# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)

# Libraries linked via full path no longer produce linker search paths.
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)

project(tigergrav CXX)

# Search path for CMake modules to be loaded by include() and find_package()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Silo REQUIRED)
find_package(Boost REQUIRED)


set(tigergrav_source_files
	src/expansion.cpp
	src/gravity.cpp
	src/initialize.cpp
	src/main.cpp
	src/new.cpp
	src/output.cpp
	src/options.cpp
	src/rand.cpp
	src/range.cpp
	src/tree.cpp
)

set(test_source_files
	src/rand.cpp
	src/test.cpp
)

set(tigergrav_header_files
	tigergrav/cuda_export.hpp
	tigergrav/defs.hpp
	tigergrav/gravity.hpp
	tigergrav/initialize.hpp
	tigergrav/multipole.hpp
	tigergrav/output.hpp
	tigergrav/options.hpp
	tigergrav/particle.hpp
	tigergrav/position.hpp
	tigergrav/rand.hpp
	tigergrav/range.hpp
	tigergrav/taylor.hpp
	tigergrav/time.hpp
	tigergrav/tree.hpp
	tigergrav/vect.hpp
)


add_executable( tigergrav_single ${tigergrav_source_files}  ${Silo_LIBRARY} )
target_include_directories(tigergrav_single PUBLIC ${PROJECT_SOURCE_DIR} ~/local/boost/include ~/local/silo/include)
target_compile_options(tigergrav_single PUBLIC "-march=native")
#target_compile_options(tigergrav_single PUBLIC "-std=c++17")
target_link_libraries( tigergrav_single /home/dmarce1/local/boost/lib/libboost_program_options.a )
target_link_libraries( tigergrav_single /home/dmarce1/local/silo/lib/libsiloh5.a )
target_link_libraries( tigergrav_single /home/dmarce1/local/hdf5/lib/libhdf5.a )
target_link_libraries( tigergrav_single z)
target_link_libraries( tigergrav_single dl)
target_link_libraries( tigergrav_single pthread)



add_hpx_executable(
  tigergrav
  DEPENDENCIES
    Silo::silo Boost::boost
  SOURCES
    ${tigergrav_source_files}
  HEADERS
    ${tigergrav_header_files}
 )

add_hpx_executable(
  test
  DEPENDENCIES
    Silo::silo Boost::boost
  SOURCES
    ${test_source_files}
  HEADERS
    ${tigergrav_header_files}
 )


target_include_directories(tigergrav PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tigergrav PUBLIC "-DUSE_HPX")
target_compile_options(tigergrav PUBLIC "-march=native")

target_include_directories(test PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(test PUBLIC "-march=native")

