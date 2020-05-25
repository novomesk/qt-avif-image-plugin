# NB: the desktop files are installed for the benefit of KImageIO in KDELibs4Support.

##################################

function(kimageformats_add_plugin plugin)
    set(options)
    set(oneValueArgs JSON)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(KIF_ADD_PLUGIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT KIF_ADD_PLUGIN_SOURCES)
        message(FATAL_ERROR "kimageformats_add_plugin called without SOURCES parameter")
    endif()
    get_filename_component(json "${KIF_ADD_PLUGIN_JSON}" REALPATH)
    if(NOT KIF_ADD_PLUGIN_JSON OR NOT EXISTS ${json})
        message(FATAL_ERROR "JSON file doesn't exist: ${json}")
    endif()

    add_library(${plugin} MODULE ${KIF_ADD_PLUGIN_SOURCES})
    set_property(TARGET ${plugin} APPEND PROPERTY AUTOGEN_TARGET_DEPENDS ${json})
    set_target_properties(${plugin} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/imageformats")
    target_link_libraries(${plugin} Qt5::Gui)
    install(TARGETS ${plugin} DESTINATION ${KDE_INSTALL_QTPLUGINDIR}/imageformats)
endfunction()

##################################

kimageformats_add_plugin(libqavif JSON "avif.json" SOURCES avif_qt.cpp)

if(BUILD_QAVIF_DYNAMIC)
  target_link_libraries(libqavif "avif")
else()
  target_include_directories(libqavif BEFORE PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/libavif/include>)
  target_link_libraries(libqavif "${CMAKE_SOURCE_DIR}/ext/libavif/build/libavif.a" "${CMAKE_SOURCE_DIR}/ext/libavif/ext/aom/build.libavif/libaom.a" "pthread")
endif()

install(FILES "${CMAKE_SOURCE_DIR}/share/kservices5/qimageioplugins/avif.desktop" "${CMAKE_SOURCE_DIR}/share/kservices5/qimageioplugins/avifs.desktop" DESTINATION ${KDE_INSTALL_KSERVICES5DIR}/qimageioplugins/)

##################################
