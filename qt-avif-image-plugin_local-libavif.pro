TARGET = qavif

HEADERS = src/avif_qt_p.h
SOURCES = src/avif_qt.cpp
OTHER_FILES = src/avif.json

LIBS += ext/libavif/build/libavif.a ext/libavif/ext/aom/build.libavif/libaom.a

INCLUDEPATH += ext/libavif/include

PLUGIN_TYPE = imageformats
PLUGIN_CLASS_NAME = AVIFPlugin
load(qt_plugin)

CONFIG += release skip_target_version_ext strict_c++ warn_on
CONFIG -= separate_debug_info debug debug_and_release force_debug_info

QMAKE_TARGET_COMPANY = "Daniel Novomesky"
QMAKE_TARGET_PRODUCT = "qt-avif-image-plugin"
QMAKE_TARGET_DESCRIPTION = "Qt plug-in to allow Qt and KDE based applications to read/write AVIF images."
QMAKE_TARGET_COPYRIGHT = "Copyright (C) 2020 Daniel Novomesky"
