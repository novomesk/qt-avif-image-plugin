TARGET = qavif6

HEADERS = ../src/qavifhandler_p.h ../src/util_p.h
SOURCES = ../src/qavifhandler.cpp
OTHER_FILES = ../src/avif.json

SOURCES += ../src/main.cpp

LIBS += ../ext/libavif/build-ro/libavif.a ../ext/libavif/ext/dav1d/build/src/libdav1d.a ../ext/libavif/ext/libyuv/build/libyuv.a
unix:LIBS += -ldl

INCLUDEPATH += ../ext/libavif/include

TEMPLATE = lib

CONFIG += release skip_target_version_ext c++14 warn_on plugin
CONFIG -= separate_debug_info debug debug_and_release force_debug_info

win32:VERSION = 0.9.1
QMAKE_TARGET_COMPANY = "Daniel Novomesky"
QMAKE_TARGET_PRODUCT = "qt-avif-image-plugin"
QMAKE_TARGET_DESCRIPTION = "Qt plug-in to allow Qt and KDE based applications to read/write AVIF images."
QMAKE_TARGET_COPYRIGHT = "Copyright (C) 2020-2025 Daniel Novomesky"
QMAKE_TARGET_COMMENTS = "Build using Qt 6.8.1, read-only AVIF support"
