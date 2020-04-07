#!/bin/bash

if ! [ -x "$(command -v qmake)" ]; then
  echo 'Error: qmake is not installed.' >&2
  exit 1
fi

if ! [ -x "$(command -v make)" ]; then
  echo 'Error: make is not installed.' >&2
  exit 1
fi

RELATIVE_PATH=`dirname "$BASH_SOURCE"`
cd "$RELATIVE_PATH"

SRC_FOLDER="src-dynamic"
cd $SRC_FOLDER

if ! [ -f Makefile ]; then

  qmake

  if ! [ -f Makefile ]; then
    echo 'qmake failed to produce Makefile' >&2
    exit 1
  fi
fi

make

if ! [ -f libqavif.so ]; then
  echo 'Failed to build libqavif.so' >&2
  exit 1
fi

echo "SUCCESS! libqavif is ready in $SRC_FOLDER"
exit 0
