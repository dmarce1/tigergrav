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
enable_language(CUDA)

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
	src/output.cpp
	src/options.cpp
	src/rand.cpp
	src/range.cpp
	src/tree.cpp
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




add_hpx_executable(
  tigergrav
  DEPENDENCIES
    Silo::silo Boost::boost
  SOURCES
    ${tigergrav_source_files}
  HEADERS
    ${tigergrav_header_files}
 )



target_include_directories(tigergrav PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tigergrav PUBLIC "-march=native")

