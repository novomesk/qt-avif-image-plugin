TARGET = qavif

HEADERS = avif_qt_p.h
SOURCES = avif_qt.cpp
OTHER_FILES = avif.json

CONFIG = release qt plugin skip_target_version_ext strict_c++

LIBS += -lavif

TEMPLATE = lib

QMAKE_CFLAGS += -Wall -Wextra
QMAKE_CXXFLAGS += -Wall -Wextra
