TARGET = qavif

HEADERS = src/qavifhandler_p.h src/util_p.h
SOURCES = src/qavifhandler.cpp
OTHER_FILES = src/avif.json

SOURCES += src/main.cpp

LIBS += -lavif

PLUGIN_TYPE = imageformats
PLUGIN_CLASS_NAME = QAVIFPlugin
load(qt_plugin)

CONFIG += release skip_target_version_ext c++14 warn_on
CONFIG -= separate_debug_info debug debug_and_release force_debug_info

QMAKE_TARGET_COMPANY = "Daniel Novomesky"
QMAKE_TARGET_PRODUCT = "qt-avif-image-plugin"
QMAKE_TARGET_DESCRIPTION = "Qt plug-in to allow Qt and KDE based applications to read/write AVIF images."
QMAKE_TARGET_COPYRIGHT = "Copyright (C) 2020-2022 Daniel Novomesky"
