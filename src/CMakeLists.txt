# NB: the desktop files are installed for the benefit of KImageIO in KDELibs4Support.

##################################

function(kimageformats_add_plugin plugin)
    set(options)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(KIF_ADD_PLUGIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT KIF_ADD_PLUGIN_SOURCES)
        message(FATAL_ERROR "kimageformats_add_plugin called without SOURCES parameter")
    endif()

    add_library(${plugin} MODULE ${KIF_ADD_PLUGIN_SOURCES})
    set_target_properties(${plugin} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/imageformats")
    target_link_libraries(${plugin} Qt${QT_MAJOR_VERSION}::Gui)
    install(TARGETS ${plugin} DESTINATION ${KDE_INSTALL_QTPLUGINDIR}/imageformats)
endfunction()

##################################

kimageformats_add_plugin("libqavif${QT_MAJOR_VERSION}" SOURCES "main.cpp" "qavifhandler.cpp")

if(BUILD_QAVIF_DYNAMIC)
  target_link_libraries("libqavif${QT_MAJOR_VERSION}" "avif")
else()
  target_include_directories("libqavif${QT_MAJOR_VERSION}" BEFORE PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/libavif/include>)
  target_link_libraries("libqavif${QT_MAJOR_VERSION}" "${CMAKE_SOURCE_DIR}/ext/libavif/build/libavif.a" "${CMAKE_SOURCE_DIR}/ext/libavif/ext/aom/build.libavif/libaom.a" "pthread")
endif()

#install(FILES "${CMAKE_SOURCE_DIR}/share/kservices5/qimageioplugins/avif.desktop" "${CMAKE_SOURCE_DIR}/share/kservices5/qimageioplugins/avifs.desktop" DESTINATION ${KDE_INSTALL_KSERVICESDIR}/qimageioplugins/)

##################################
