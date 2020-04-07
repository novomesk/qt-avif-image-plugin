TARGET = qavif

HEADERS = avif_qt_p.h
SOURCES = avif_qt.cpp
OTHER_FILES = avif.json

CONFIG = release qt plugin skip_target_version_ext strict_c++

INCLUDEPATH += "../ext/libavif/include"
LIBS += ../ext/libavif/build/libavif.a ../ext/libavif/ext/aom/build.libavif/libaom.a

TEMPLATE = lib

QMAKE_CFLAGS += -Wall -Wextra
QMAKE_CXXFLAGS += -Wall -Wextra
