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

project(tigergrav CXX C)

# Search path for CMake modules to be loaded by include() and find_package()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Silo REQUIRED)
find_package(Boost REQUIRED)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "-g -O3 -march=native")

set(tigergrav_source_files
	src/cosmo.cpp
	src/gravity_cuda.cu
	src/gravity.cpp
	src/gravity_work.cpp
	src/groups.cpp
	src/initialize.cpp
	src/load.cpp
	src/main.cpp
	src/map.cpp
	src/new.cpp
	src/output.cpp
	src/options.cpp
	src/part_vect.cpp
	src/rand.cpp
	src/range.cpp
	src/tree.cpp
)

set(test_source_files
	src/rand.cpp
	src/test.cpp
)

set(tigergrav_header_files
	tiegrgrav/cosmo.hpp
	tigergrav/defs.hpp
	tigergrav/gravity_cuda.hpp
	tigergrav/gravity.hpp
	tigergrav/gravity_work.hpp
	tigergrav/green.hpp
	tigergrav/groups.hpp
	tigergrav/initialize.hpp
	tigergrac/load.hpp
	tigergrav/map.hpp
	tigergrav/multipole.hpp
	tigergrav/output.hpp
	tigergrav/options.hpp
	tigergrav/part_vect.hpp
	tigergrav/particle.hpp
	tigergrav/position.hpp
	tigergrav/rand.hpp
	tigergrav/range.hpp
	tigergrav/taylor.hpp
	tigergrav/time.hpp
	tigergrav/tree.hpp
	tigergrav/vect.hpp
)


#add_executable( tigergrav_single ${tigergrav_source_files}  ${Silo_LIBRARY} )
#target_include_directories(tigergrav_single PUBLIC ${PROJECT_SOURCE_DIR} ~/local/boost/include ~/local/silo/include)
#target_compile_options(tigergrav_single PUBLIC "-O3 -march=native")
#target_compile_options(tigergrav_single PUBLIC "-std=c++17")
#target_link_libraries( tigergrav_single /home/dmarce1/local/boost/lib/libboost_program_options.a )
#target_link_libraries( tigergrav_single /home/dmarce1/local/silo/lib/libsiloh5.a )
#target_link_libraries( tigergrav_single /home/dmarce1/local/hdf5/lib/libhdf5.a )
#target_link_libraries( tigergrav_single z)
#target_link_libraries( tigergrav_single dl)
#target_link_libraries( tigergrav_single pthread)



add_library(hpx_lite 
	hpx_lite/src/hpx.cpp
	hpx_lite/src/id_type.cpp
	hpx_lite/src/thread.cpp
	hpx_lite/src/future.cpp
	hpx_lite/src/mutex.cpp
	hpx_lite/src/serialize.cpp
)

add_executable( tigergrav_lite
    ${tigergrav_source_files}
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

add_hpx_executable(
  test
  DEPENDENCIES
    Silo::silo Boost::boost
  SOURCES
    ${test_source_files}
  HEADERS
    ${tigergrav_header_files}
 )


target_link_libraries (tigergrav_lite boost_program_options tcmalloc hpx_lite siloh5 hwloc dl z hdf5)


target_include_directories(hpx_lite PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite)
target_include_directories(tigergrav PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tigergrav PUBLIC "-DUSE_HPX")
#target_compile_options(tigergrav_lite PUBLIC "-O3")
#target_compile_options(tigergrav PUBLIC "-O3")
#target_compile_options(tigergrav PUBLIC "-march=native")
target_include_directories(tigergrav_lite PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite)
target_compile_options(tigergrav_lite PUBLIC "-DHPX_LITE")
#target_compile_options(tigergrav_lite PUBLIC "-march=native")
target_compile_options(tigergrav_lite PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --gpu-architecture=compute_75;
			-Xptxas;
			-v
                        >)
target_compile_options(tigergrav PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --gpu-architecture=compute_75;
                        >)

target_include_directories(test PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(test PUBLIC -O3 -march=native)

