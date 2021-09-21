#!/bin/bash

if ! [ -x "$(command -v cmake)" ]; then
  echo 'Error: cmake is not installed.' >&2
  exit 1
fi

if ! [ -x "$(command -v ninja)" ]; then
  echo 'Error: ninja is not installed.' >&2
  exit 1
fi

RELATIVE_PATH=`dirname "$BASH_SOURCE"`
cd "$RELATIVE_PATH"

if ! [ -f libavif/ext/libyuv/build/libyuv.a ]; then
  echo 'We are going to build libyuv.a'
  cd libavif/ext/libyuv
  mkdir -p build
  cd build

  cmake -G Ninja -DBUILD_SHARED_LIBS=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" ..
  ninja yuv

  if ! [ -f libyuv.a ]; then
    echo 'Error: libyuv.a build failed!' >&2
  fi

  cd ../../../..
fi

if ! [ -f libavif/ext/aom/build.libavif/libaom.a ]; then
  echo 'We are going to build libaom.a'
  cd libavif/ext/aom
  mkdir -p build.libavif
  cd build.libavif

  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCONFIG_AV1_DECODER=0 -DCONFIG_AV1_ENCODER=1 -DENABLE_DOCS=0 -DENABLE_EXAMPLES=0 -DENABLE_TESTDATA=0 -DENABLE_TESTS=0 -DENABLE_TOOLS=0 -DCONFIG_PIC=1 ..
  ninja

  if ! [ -f libaom.a ]; then
    echo 'Error: libaom.a build failed!' >&2
    exit 1
  fi
  cd ../../../..
fi

if ! [ -f libavif/ext/dav1d/build/src/libdav1d.a ]; then
 echo 'We are going to build libdav1d.a'
 cd libavif/ext/dav1d
 mkdir -p build
 cd build

 meson --default-library=static --buildtype release ..
 ninja

 if ! [ -f src/libdav1d.a ]; then
  echo 'Error: libdav1d.a build failed!' >&2
  exit 1
 fi
 cd ../../../..
fi

if ! [ -f libavif/build/libavif.a ]; then
  echo 'We are going to build libavif.a'
  cd libavif
  mkdir -p build
  cd build

  CFLAGS="-fPIC" cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DAVIF_CODEC_AOM=ON -DAVIF_LOCAL_AOM=ON -DAVIF_CODEC_AOM_DECODE=OFF -DAVIF_CODEC_AOM_ENCODE=ON -DAVIF_CODEC_DAV1D=ON -DAVIF_LOCAL_DAV1D=ON -DAVIF_LOCAL_LIBYUV=ON ..
  ninja

  if ! [ -f libavif.a ]; then
    echo 'Error: libavif.a build failed!' >&2
    exit 1
  fi
  cd ../..
fi

exit 0
