cmake_minimum_required(VERSION 3.16)

project(qt-avif-image-plugin)

include(FeatureSummary)
find_package(ECM 5.89.0  NO_MODULE)
set_package_properties(ECM PROPERTIES TYPE REQUIRED DESCRIPTION "Extra CMake Modules." URL "https://commits.kde.org/extra-cmake-modules")
feature_summary(WHAT REQUIRED_PACKAGES_NOT_FOUND FATAL_ON_MISSING_REQUIRED_PACKAGES)


set(CMAKE_MODULE_PATH ${ECM_MODULE_PATH})

include(KDEInstallDirs)
include(KDEFrameworkCompilerSettings NO_POLICY_SCOPE)
include(KDECMakeSettings)


include(CheckIncludeFiles)

set(REQUIRED_QT_VERSION 5.12.0)
find_package(Qt${QT_MAJOR_VERSION}Gui ${REQUIRED_QT_VERSION} REQUIRED NO_MODULE)

find_package(libavif 1.0.0 CONFIG QUIET)
if(NOT libavif_FOUND)
  find_package(libavif 0.8.2 CONFIG QUIET)
endif()
if (TARGET avif)
  MESSAGE(STATUS "libavif found - we will use dynamic linking")
  set(BUILD_QAVIF_DYNAMIC TRUE)
else()
  MESSAGE(STATUS "libavif not found! Searching for local build.")
  set(BUILD_QAVIF_DYNAMIC FALSE)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/ext/libavif/build/libavif.a")
    message(FATAL_ERROR "libavif.a not found, run ${CMAKE_SOURCE_DIR}/ext/build_local_libaom_libavif.sh")
  endif()
endif()

add_definitions(-DQT_DISABLE_DEPRECATED_BEFORE=0x050f02)
add_definitions(-DKF_DISABLE_DEPRECATED_BEFORE_AND_AT=0x055900)
add_subdirectory(src)

feature_summary(WHAT ALL FATAL_ON_MISSING_REQUIRED_PACKAGES)
