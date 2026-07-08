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

# =========================
# Tests del proyecto final
# =========================

include(CTest)
enable_testing()

function(add_prog3_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    target_compile_features(${TEST_NAME} PRIVATE cxx_std_23)

    target_include_directories(${TEST_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )

    # Si el profesor dejó TBB configurado, lo usamos solo si existe el target.
    # Esto evita romper el build cuando TBB no está instalado.
    if(TARGET TBB::tbb)
        target_link_libraries(${TEST_NAME} PRIVATE TBB::tbb)
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# -------------------------
# Epic 1
# -------------------------

add_prog3_test(
    epic1_feature1_eigen_backend_test
    tests/epic1/feature1_eigen_backend_test.cpp
)

add_prog3_test(
    epic1_feature2_numeric_ops_test
    tests/epic1/feature2_numeric_ops_test.cpp
)

add_prog3_test(
    epic1_feature3_forward_layers_test
    tests/epic1/feature3_forward_layers_test.cpp
)

add_prog3_test(
    epic1_feature4_computational_graph_test
    tests/epic1/feature4_computational_graph_test.cpp
)

# -------------------------
# Epic 2
# -------------------------
# Si tus archivos se llaman feature1_test.cpp, feature2_test.cpp, etc.,
# déjalos como están abajo.

add_prog3_test(
    epic2_feature1_monitoring_test
    tests/epic2/feature1_test.cpp
)

add_prog3_test(
    epic2_feature2_cnn_application_test
    tests/epic2/feature2_test.cpp
)

add_prog3_test(
    epic2_feature3_serialization_test
    tests/epic2/feature3_test.cpp
)

add_prog3_test(
    epic2_feature4_reporting_test
    tests/epic2/feature4_test.cpp
)
