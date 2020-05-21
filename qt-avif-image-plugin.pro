TARGET = qavif

HEADERS = src/avif_qt_p.h
SOURCES = src/avif_qt.cpp
OTHER_FILES = src/avif.json

LIBS += -lavif

CONFIG += skip_target_version_ext strict_c++

QMAKE_CFLAGS += -Wall -Wextra
QMAKE_CXXFLAGS += -Wall -Wextra

PLUGIN_TYPE = imageformats
PLUGIN_CLASS_NAME = AVIFPlugin
load(qt_plugin)
