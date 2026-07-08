cmake_minimum_required(VERSION 3.10.2)
project(prog3_pf_epic1_feature4_v2026_1_test)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_DEBUG})
if(UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

# Work Folders
include_directories(.)
include_directories(tests)
include_directories(../include)
include_directories(../src)
include_directories(../tools/catch)

# Project files
file(GLOB_RECURSE project_files CONFIGURE_DEPENDS
        tests/question_*/test_*/*.cpp
        tests/test-main.cpp
        ../src/*.cpp
        ../src/*.cc
        ../src/*.cxx
        )

add_executable(${PROJECT_NAME}
        ${project_files}
    )

if(UNIX AND NOT APPLE)
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(${PROJECT_NAME} TBB::tbb)
    endif()
endif()

# Test files
file(GLOB_RECURSE test_files CONFIGURE_DEPENDS
        tests/question_*/test_*/*.in
        tests/question_*/test_*/*.txt
    )
foreach(full_test_file_name ${test_files})
    configure_file(${full_test_file_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach(full_test_file_name)
