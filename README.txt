QT imageformats plug-in for AV1 Image File Format (AVIF)

This is a plug-in for QT based applications to enable them to open/save images in AVIF format.

Format support:
AVIF (AV1 Image File Format - image/avif): read/write support
AVIFS (AVIF image sequence): ready only

Software requirements:
yasm – needed to build optimized libaom
cmake – needed to build libaom and libavif
perl – needed to build libaom
nasm – needed to build optimized dav1d
meson – build system used by dav1d
qmake (or qt5-qmake) – to build the QT plugin
QT5 development packages (for example qtbase5-dev on Ubuntu)
qt5-image-formats-plugins (or qt5-imageformats, dev-qt/qtimageformats, …)
The plug-in use libavif library internally ( https://github.com/AOMediaCodec/libavif ).
Qt4 is not supported and the plug-in doesn’t compile with Qt older than Qt 5.12

Optional requirement:
Plug-in supports Color spaces and loading ICC profiles when built with Qt 5.14 or newer.


How to build the plug-in (two possibilities):

1) Dynamic linking with libavif (ONLY for systems with >=libavif 0.8.2 installed!!!)
Run:
./build_libqavif_dynamic.sh

2) Static linking (should work everywhere but resulting plug-in with be larger)
Run:
./build_libqavif_static.sh
The static version takes longer time to build because libaom and libavif libraries will be compiled locally.

The result of building process is libqavif.so

How to install:
Copy libqavif.so into the folder where other plug-ins from qt5-image-formats-plugins are installed. It could be one of the following locations:
Gentoo: /usr/lib/qt5/plugins/imageformats
Archlinux: /usr/lib/qt/plugins/imageformats/
Ubuntu: /usr/lib/x86_64-linux-gnu/qt5/plugins/imageformats

Further recommend steps:
If you don’t have "image/avif" mime type installed, copy qt-avif-image-plugin.xml to:
/usr/share/mime/packages
And run:
update-mime-database /usr/share/mime

AVIF Thumbnails in KDE’s dolphin
Copy avif.desktop, avifs.desktop to:
/usr/share/kservices5/qimageioplugins/

Update imagethumbnail.desktop (in /usr/share/kservices5/ ):
Add ;image/avif to the MimeType= list.

How to test:
Associate image/avif (*.avif) with gwenview and it should be able to display test images:
https://github.com/AOMediaCodec/av1-avif/tree/master/testFiles

Things good to know:

imagethumbnail.desktop belongs to kde-apps/kio-extras (in Gentoo Linux).
When the kio-extras is updated, configuration in imagethumbnail.desktop is overwritten.
If you want to see thumbnais in dolphin again, will need to add
;image/avif;
to the MimeType= list after kio-extras update again.

Before you update the plug-in, save your work!!!
If you overwrite the libqavif.so in running KDE environment, running Qt/KDE applications may crash.
Crash may not occur immediately; it may be after few minutes.
It is safer to delete the old libqavif.so first before installing the new one,
or to perform update when user’s Qt applications are not running.

If you are building libavif by yourself with libaom support, make sure that the libaom is recently fresh.
You need at least libaom v1.0.0-errata1-avif, v2.0.0 is recommended.
Do not use the old libaom v1.0.0 which is present in some distributions.

