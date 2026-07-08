cmake_minimum_required(VERSION 3.10.2)
project(prog3_pf_epic1_feature4_v2026_1)

set(CMAKE_CXX_STANDARD 23)
if(UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

include_directories(./include ./src)
file(GLOB_RECURSE source_files CONFIGURE_DEPENDS
        src/*.cpp
        src/*.cc
        src/*.cxx)

add_executable(${PROJECT_NAME}
        main.cpp
        ${source_files})

if(UNIX AND NOT APPLE)
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(${PROJECT_NAME} TBB::tbb)
    endif()
endif()

add_subdirectory(autograder)

