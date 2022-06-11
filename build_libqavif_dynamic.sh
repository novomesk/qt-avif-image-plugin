#!/bin/bash

if ! [ -x "$(command -v qmake)" ]; then
  if ! [ -x "$(command -v qmake5)" ]; then
    echo 'Error: qmake or qmake5 is not installed.' >&2
    exit 1
  else
    QMAKE_EXE=qmake5
  fi
else
  QMAKE_EXE=qmake
fi

if ! [ -x "$(command -v make)" ]; then
  echo 'Error: make is not installed.' >&2
  exit 1
fi

RELATIVE_PATH=`dirname "$BASH_SOURCE"`
cd "$RELATIVE_PATH"


$QMAKE_EXE qt-avif-image-plugin.pro

if ! [ -f Makefile ]; then
  echo 'qmake failed to produce Makefile' >&2
  exit 1
fi

make

if [ $? -eq 0 ]; then
  echo "SUCCESS! in order to install libqavif.so type as root:"
  echo "make install"
  exit 0
else
  echo 'Failed to build libqavif.so' >&2
  exit 1
fi
