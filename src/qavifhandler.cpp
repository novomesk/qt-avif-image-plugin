/*
 * QT plug-in to allow import/export in AVIF image format.
 * Author: Daniel Novomesky
 */

/*
This software uses libavif
URL: https://github.com/AOMediaCodec/libavif/

Copyright 2019 Joe Drago. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <QtGlobal>
#include <QThread>

#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
#include <QColorSpace>
#endif

#include "qavifhandler_p.h"


QAVIFHandler::QAVIFHandler() :
    m_parseState(ParseAvifNotParsed),
    m_quality(52),
    m_container_width(0),
    m_container_height(0),
    m_rawAvifData(AVIF_DATA_EMPTY),
    m_decoder(NULL),
    m_must_jump_to_next_image(false)
{
}

QAVIFHandler::~QAVIFHandler()
{
    if (m_decoder) {
        avifDecoderDestroy(m_decoder);
    }
}

bool QAVIFHandler::canRead() const
{
    if (m_parseState == ParseAvifNotParsed && !canRead(device())) {
        return false;
    }

    if (m_parseState != ParseAvifError) {
        setFormat("avif");
        return true;
    }
    return false;
}

bool QAVIFHandler::canRead(QIODevice *device)
{
    if (!device) {
        return false;
    }
    QByteArray header = device->peek(144);
    if (header.size() < 12) {
        return false;
    }

    avifROData input;
    input.data = (const uint8_t *) header.constData();
    input.size = header.size();

    if (avifPeekCompatibleFileType(&input)) {
        return true;
    }
    return false;
}

bool QAVIFHandler::ensureParsed() const
{
    if (m_parseState == ParseAvifSuccess) {
        return true;
    }
    if (m_parseState == ParseAvifError) {
        return false;
    }

    QAVIFHandler *that = const_cast<QAVIFHandler *>(this);

    return that->ensureDecoder();
}

bool QAVIFHandler::ensureDecoder()
{
    if (m_decoder) {
        return true;
    }

    m_rawData = device()->readAll();

    m_rawAvifData.data = (const uint8_t *) m_rawData.constData();
    m_rawAvifData.size = m_rawData.size();

    if (avifPeekCompatibleFileType(&m_rawAvifData) == AVIF_FALSE) {
        m_parseState = ParseAvifError;
        return false;
    }


    m_decoder = avifDecoderCreate();

    avifResult decodeResult;

    decodeResult = avifDecoderParse(m_decoder, &m_rawAvifData);
    if (decodeResult != AVIF_RESULT_OK) {
        qWarning("ERROR: Failed to parse input: %s\n", avifResultToString(decodeResult));

        avifDecoderDestroy(m_decoder);
        m_decoder = NULL;
        m_parseState = ParseAvifError;
        return false;
    }

    decodeResult = avifDecoderNextImage(m_decoder);

    if (decodeResult == AVIF_RESULT_OK) {

        m_container_width = m_decoder->image->width;
        m_container_height = m_decoder->image->height;

        if ((m_container_width > 32768) || (m_container_height > 32768)) {
            qWarning("AVIF image (%dx%d) is too large!", m_container_width, m_container_height);
            m_parseState = ParseAvifError;
            return false;
        }

        if ((m_container_width == 0) || (m_container_height == 0)) {
            qWarning("Empty image, nothing to decode");
            m_parseState = ParseAvifError;
            return false;
        }

        m_parseState = ParseAvifSuccess;
        if (decode_one_frame()) {
            return true;
        } else {
            m_parseState = ParseAvifError;
            return false;
        }
    } else {
        qWarning("ERROR: Failed to decode image: %s\n", avifResultToString(decodeResult));
    }

    avifDecoderDestroy(m_decoder);
    m_decoder = NULL;
    m_parseState = ParseAvifError;
    return false;
}

bool QAVIFHandler::decode_one_frame()
{
    if (!ensureParsed()) {
        return false;
    }

    bool loadalpha;

    if (m_decoder->image->alphaPlane) {
        loadalpha = true;
    } else {
        loadalpha = false;
    }

    QImage::Format resultformat;

    if (m_decoder->image->depth > 8) {
        resultformat = QImage::Format_RGBA64;
    } else {
        if (loadalpha) {
            resultformat = QImage::Format_RGBA8888;
        } else {
            resultformat = QImage::Format_RGB888;
        }
    }
    QImage result(m_decoder->image->width, m_decoder->image->height, resultformat);

    if (result.isNull()) {
        qWarning("Memory cannot be allocated");
        return false;
    }

#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
    if (m_decoder->image->icc.data && (m_decoder->image->icc.size > 0)) {
        result.setColorSpace(QColorSpace::fromIccProfile(QByteArray::fromRawData((const char *) m_decoder->image->icc.data, (int) m_decoder->image->icc.size)));
        if (! result.colorSpace().isValid()) {
            qWarning("Invalid QColorSpace created from ICC!\n");
        }
    } else {

        avifColorPrimaries primaries_to_load;
        avifTransferCharacteristics trc_to_load;

        if ((m_decoder->image->colorPrimaries == 2 /* AVIF_COLOR_PRIMARIES_UNSPECIFIED */) ||
                (m_decoder->image->colorPrimaries == 0 /* AVIF_COLOR_PRIMARIES_UNKNOWN */)) {
            primaries_to_load = (avifColorPrimaries) 1; // AVIF_COLOR_PRIMARIES_BT709
        } else {
            primaries_to_load = m_decoder->image->colorPrimaries;
        }
        if ((m_decoder->image->transferCharacteristics == 2 /* AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED */) ||
                (m_decoder->image->transferCharacteristics == 0 /* AVIF_TRANSFER_CHARACTERISTICS_UNKNOWN */)) {
            trc_to_load = (avifTransferCharacteristics) 13; // AVIF_TRANSFER_CHARACTERISTICS_SRGB
        } else {
            trc_to_load = m_decoder->image->transferCharacteristics;
        }

        float prim[8]; // outPrimaries: rX, rY, gX, gY, bX, bY, wX, wY
        avifColorPrimariesGetValues(primaries_to_load, prim);

        QPointF redPoint(prim[0], prim[1]);
        QPointF greenPoint(prim[2], prim[3]);
        QPointF bluePoint(prim[4], prim[5]);
        QPointF whitePoint(prim[6], prim[7]);


        QColorSpace::TransferFunction q_trc = QColorSpace::TransferFunction::Custom;
        float q_trc_gamma = 0.0f;

        switch (trc_to_load) {
        /* AVIF_TRANSFER_CHARACTERISTICS_BT470M */
        case 4:
            q_trc = QColorSpace::TransferFunction::Gamma;
            q_trc_gamma = 2.2f;
            break;
        /* AVIF_TRANSFER_CHARACTERISTICS_BT470BG */
        case 5:
            q_trc = QColorSpace::TransferFunction::Gamma;
            q_trc_gamma = 2.8f;
            break;
        /* AVIF_TRANSFER_CHARACTERISTICS_LINEAR */
        case 8:
            q_trc = QColorSpace::TransferFunction::Linear;
            break;
        /* AVIF_TRANSFER_CHARACTERISTICS_SRGB */
        case 13:
            q_trc =  QColorSpace::TransferFunction::SRgb;
            break;
        default:
            qWarning("CICP colorPrimaries: %d, transferCharacteristics: %d\nThe colorspace is unsupported by this plug-in yet.",
                     m_decoder->image->colorPrimaries, m_decoder->image->transferCharacteristics);
            q_trc = QColorSpace::TransferFunction::SRgb;
            break;
        }

        if (q_trc != QColorSpace::TransferFunction::Custom) {   //we create new colorspace using Qt
            switch (primaries_to_load) {
            /* AVIF_COLOR_PRIMARIES_BT709 */
            case 1:
                result.setColorSpace(QColorSpace(QColorSpace::Primaries::SRgb, q_trc, q_trc_gamma));
                break;
            /* AVIF_COLOR_PRIMARIES_SMPTE432 */
            case 12:
                result.setColorSpace(QColorSpace(QColorSpace::Primaries::DciP3D65, q_trc, q_trc_gamma));
                break;
            default:
                result.setColorSpace(QColorSpace(whitePoint, redPoint, greenPoint, bluePoint, q_trc, q_trc_gamma));
                break;
            }
        }

        if (! result.colorSpace().isValid()) {
            qWarning("Invalid QColorSpace created from NCLX/CICP!\n");
        }
    }
#endif

    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, m_decoder->image);
    rgb.rowBytes = result.bytesPerLine();
    rgb.pixels = result.bits();

    if (m_decoder->image->depth > 8) {
        rgb.depth = 16;
        rgb.format = AVIF_RGB_FORMAT_RGBA;

        if (!loadalpha) {
            resultformat = QImage::Format_RGBX64;
#if (QT_VERSION >= QT_VERSION_CHECK(5, 13, 0))
            if (m_decoder->image->yuvFormat == AVIF_PIXEL_FORMAT_YUV400) {
                resultformat = QImage::Format_Grayscale16;
            }
#endif
        }
    } else {
        rgb.depth = 8;
        if (loadalpha) {
            rgb.format = AVIF_RGB_FORMAT_RGBA;
            resultformat = QImage::Format_ARGB32;
        } else {
            rgb.format = AVIF_RGB_FORMAT_RGB;

            if (m_decoder->image->yuvFormat == AVIF_PIXEL_FORMAT_YUV400) {
                resultformat = QImage::Format_Grayscale8;
            } else {
                resultformat = QImage::Format_RGB32;
            }
        }
    }

    avifResult res = avifImageYUVToRGB(m_decoder->image, &rgb);
    if (res != AVIF_RESULT_OK) {
        qWarning("ERROR in avifImageYUVToRGB: %s\n", avifResultToString(res));
        return false;
    }

    if (m_decoder->image->transformFlags & AVIF_TRANSFORM_CLAP) {
        if ((m_decoder->image->clap.widthD > 0) && (m_decoder->image->clap.heightD > 0) &&
                (m_decoder->image->clap.horizOffD > 0) && (m_decoder->image->clap.vertOffD > 0)) {
            int  new_width, new_height, offx, offy;

            new_width = (int)((double)(m_decoder->image->clap.widthN)  / (m_decoder->image->clap.widthD) + 0.5);
            if (new_width > result.width()) {
                new_width = result.width();
            }

            new_height = (int)((double)(m_decoder->image->clap.heightN) / (m_decoder->image->clap.heightD) + 0.5);
            if (new_height > result.height()) {
                new_height = result.height();
            }

            if (new_width > 0 && new_height > 0) {

                offx = ((double)((int32_t) m_decoder->image->clap.horizOffN)) / (m_decoder->image->clap.horizOffD) +
                       (result.width() - new_width) / 2.0 + 0.5;
                if (offx < 0) {
                    offx = 0;
                } else if (offx > (result.width() - new_width)) {
                    offx = result.width() - new_width;
                }

                offy = ((double)((int32_t) m_decoder->image->clap.vertOffN)) / (m_decoder->image->clap.vertOffD) +
                       (result.height() - new_height) / 2.0 + 0.5;
                if (offy < 0) {
                    offy = 0;
                } else if (offy > (result.height() - new_height)) {
                    offy = result.height() - new_height;
                }

                result = result.copy(offx, offy, new_width, new_height);
            }
        }

        else { //Zero values, we need to avoid 0 divide.
            qWarning("ERROR: Wrong values in avifCleanApertureBox\n");
        }
    }

    if (m_decoder->image->transformFlags & AVIF_TRANSFORM_IROT) {
        QTransform transform;
        switch (m_decoder->image->irot.angle) {
        case 1:
            transform.rotate(-90);
            result = result.transformed(transform);
            break;
        case 2:
            transform.rotate(180);
            result = result.transformed(transform);
            break;
        case 3:
            transform.rotate(90);
            result = result.transformed(transform);
            break;
        }
    }

    if (m_decoder->image->transformFlags & AVIF_TRANSFORM_IMIR) {
        switch (m_decoder->image->imir.axis) {
        case 0: //vertical
            result = result.mirrored(false, true);
            break;
        case 1: //horizontal
            result = result.mirrored(true, false);
            break;
        }
    }

    if (resultformat == result.format()) {
        m_current_image = result;
    } else {
        m_current_image = result.convertToFormat(resultformat);
    }

    m_must_jump_to_next_image = false;
    return true;
}

bool QAVIFHandler::read(QImage *image)
{
    if (!ensureParsed()) {
        return false;
    }

    if (m_must_jump_to_next_image) {
        jumpToNextImage();
    }

    *image = m_current_image;
    if (imageCount() >= 2) {
        m_must_jump_to_next_image = true;
    }
    return true;
}

bool QAVIFHandler::write(const QImage &image)
{
    if (image.format() == QImage::Format_Invalid) {
        qWarning("No image data to save");
        return false;
    }

    if ((image.width() > 32768) || (image.height() > 32768)) {
        qWarning("Image is too large");
        return false;
    }

    int maxQuantizer = AVIF_QUANTIZER_WORST_QUALITY * (100 - qBound(0, m_quality, 100)) / 100;
    int minQuantizer = 0;
    int maxQuantizerAlpha = 0;
    avifResult res;

    bool save_grayscale; //true - monochrome, false - colors
    int save_depth; //8 or 10bit per channel
    QImage::Format tmpformat; //format for temporary image

    avifImage *avif = NULL;

    //grayscale detection
    switch (image.format()) {
    case QImage::Format_Mono:
    case QImage::Format_MonoLSB:
    case QImage::Format_Grayscale8:
#if (QT_VERSION >= QT_VERSION_CHECK(5, 13, 0))
    case QImage::Format_Grayscale16:
#endif
        save_grayscale = true;
        break;
    case QImage::Format_Indexed8:
        save_grayscale = image.isGrayscale();
        break;
    default:
        save_grayscale = false;
        break;
    }

    //depth detection
    switch (image.format()) {
    case QImage::Format_BGR30:
    case QImage::Format_A2BGR30_Premultiplied:
    case QImage::Format_RGB30:
    case QImage::Format_A2RGB30_Premultiplied:
#if (QT_VERSION >= QT_VERSION_CHECK(5, 13, 0))
    case QImage::Format_Grayscale16:
#endif
    case QImage::Format_RGBX64:
    case QImage::Format_RGBA64:
    case QImage::Format_RGBA64_Premultiplied:
        save_depth = 10;
        break;
    default:
        if (image.depth() > 32) {
            save_depth = 10;
        } else {
            save_depth = 8;
        }
        break;
    }

    //quality settings
    if (maxQuantizer > 20) {
        minQuantizer = maxQuantizer - 20;
        if (maxQuantizer > 40) { //we decrease quality of alpha channel here
            maxQuantizerAlpha = maxQuantizer - 40;
        }
    }

    if (save_grayscale && !image.hasAlphaChannel()) { //we are going to save grayscale image without alpha channel
#if (QT_VERSION >= QT_VERSION_CHECK(5, 13, 0))
        if (save_depth > 8) {
            tmpformat = QImage::Format_Grayscale16;
        } else {
            tmpformat = QImage::Format_Grayscale8;
        }
#else
        tmpformat = QImage::Format_Grayscale8;
        save_depth = 8;
#endif
        QImage tmpgrayimage = image.convertToFormat(tmpformat);

        avif = avifImageCreate(tmpgrayimage.width(), tmpgrayimage.height(), save_depth, AVIF_PIXEL_FORMAT_YUV400);
        avifImageAllocatePlanes(avif, AVIF_PLANES_YUV);

#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
        if (tmpgrayimage.colorSpace().isValid()) {
            avif->colorPrimaries = (avifColorPrimaries)1;
            avif->matrixCoefficients = (avifMatrixCoefficients)1;

            switch (tmpgrayimage.colorSpace().transferFunction()) {
            case QColorSpace::TransferFunction::Linear:
                /* AVIF_TRANSFER_CHARACTERISTICS_LINEAR */
                avif->transferCharacteristics = (avifTransferCharacteristics)8;
                break;
            case QColorSpace::TransferFunction::SRgb:
                /* AVIF_TRANSFER_CHARACTERISTICS_SRGB */
                avif->transferCharacteristics = (avifTransferCharacteristics)13;
                break;
            default:
                /* AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED */
                break;
            }

        }
#endif

        if (save_depth > 8) { // QImage::Format_Grayscale16
            for (int y = 0; y < tmpgrayimage.height(); y++) {
                const uint16_t *src16bit = (const uint16_t *) tmpgrayimage.constScanLine(y);
                uint16_t *dest16bit = (uint16_t *)(avif->yuvPlanes[0] + y * avif->yuvRowBytes[0]);
                for (int x = 0; x < tmpgrayimage.width(); x++) {
                    int tmp_pixelval = (int)(((float)(*src16bit) / 65535.0f) * 1023.0f + 0.5f); //downgrade to 10 bits
                    *dest16bit = qBound(0, tmp_pixelval, 1023);
                    dest16bit++;
                    src16bit++;
                }
            }
        } else { // QImage::Format_Grayscale8
            for (int y = 0; y < tmpgrayimage.height(); y++) {
                const uchar *src8bit = tmpgrayimage.constScanLine(y);
                uint8_t *dest8bit = avif->yuvPlanes[0] + y * avif->yuvRowBytes[0];
                for (int x = 0; x < tmpgrayimage.width(); x++) {
                    *dest8bit = *src8bit;
                    dest8bit++;
                    src8bit++;
                }
            }
        }

    } else { //we are going to save color image
        if (save_depth > 8) {
            if (image.hasAlphaChannel()) {
                tmpformat = QImage::Format_RGBA64;
            } else {
                tmpformat = QImage::Format_RGBX64;
            }
        } else { //8bit depth
            if (image.hasAlphaChannel()) {
                tmpformat = QImage::Format_RGBA8888;
            } else {
                tmpformat = QImage::Format_RGB888;
            }
        }

        QImage tmpcolorimage = image.convertToFormat(tmpformat);

        avifPixelFormat pixel_format = AVIF_PIXEL_FORMAT_YUV420;
        if (maxQuantizer < 20) {
            if (maxQuantizer < 10) {
                pixel_format = AVIF_PIXEL_FORMAT_YUV444; //best quality
            } else {
                pixel_format = AVIF_PIXEL_FORMAT_YUV422; //high quality
            }
        }

        avifMatrixCoefficients matrix_to_save = (avifMatrixCoefficients)1; //default for Qt 5.12 and 5.13;

#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))

        avifColorPrimaries primaries_to_save = (avifColorPrimaries)2;
        avifTransferCharacteristics transfer_to_save = (avifTransferCharacteristics)2;

        if (tmpcolorimage.colorSpace().isValid()) {
            switch (tmpcolorimage.colorSpace().primaries()) {
            case QColorSpace::Primaries::SRgb:
                /* AVIF_COLOR_PRIMARIES_BT709 */
                primaries_to_save = (avifColorPrimaries)1;
                /* AVIF_MATRIX_COEFFICIENTS_BT709 */
                matrix_to_save = (avifMatrixCoefficients)1;
                break;
            case QColorSpace::Primaries::DciP3D65:
                /* AVIF_NCLX_COLOUR_PRIMARIES_P3, AVIF_NCLX_COLOUR_PRIMARIES_SMPTE432 */
                primaries_to_save = (avifColorPrimaries)12;
                /* AVIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_NCL */
                matrix_to_save = (avifMatrixCoefficients)12;
                break;
            default:
                /* AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED */
                primaries_to_save = (avifColorPrimaries)2;
                /* AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED */
                matrix_to_save = (avifMatrixCoefficients)2;
                break;
            }

            switch (tmpcolorimage.colorSpace().transferFunction()) {
            case QColorSpace::TransferFunction::Linear:
                /* AVIF_TRANSFER_CHARACTERISTICS_LINEAR */
                transfer_to_save = (avifTransferCharacteristics)8;
                break;
            case QColorSpace::TransferFunction::Gamma:
                if (qAbs(tmpcolorimage.colorSpace().gamma() - 2.2f) < 0.1f) {
                    /* AVIF_TRANSFER_CHARACTERISTICS_BT470M */
                    transfer_to_save = (avifTransferCharacteristics)4;
                } else if (qAbs(tmpcolorimage.colorSpace().gamma() - 2.8f) < 0.1f) {
                    /* AVIF_TRANSFER_CHARACTERISTICS_BT470BG */
                    transfer_to_save = (avifTransferCharacteristics)5;
                } else {
                    /* AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED */
                    transfer_to_save = (avifTransferCharacteristics)2;
                }
                break;
            case QColorSpace::TransferFunction::SRgb:
                /* AVIF_TRANSFER_CHARACTERISTICS_SRGB */
                transfer_to_save = (avifTransferCharacteristics)13;
                break;
            default:
                /* AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED */
                transfer_to_save = (avifTransferCharacteristics)2;
                break;
            }

            //in case primaries or trc were not identified
            if ((primaries_to_save == 2) ||
                    (transfer_to_save == 2)) {

                //upgrade image to higher bit depth
                if (save_depth == 8) {
                    save_depth = 10;
                    if (tmpcolorimage.hasAlphaChannel()) {
                        tmpcolorimage = tmpcolorimage.convertToFormat(QImage::Format_RGBA64);
                    } else {
                        tmpcolorimage = tmpcolorimage.convertToFormat(QImage::Format_RGBX64);
                    }
                }

                if ((primaries_to_save == 2) &&
                        (transfer_to_save != 2)) {    //other primaries but known trc
                    primaries_to_save = (avifColorPrimaries)1; // AVIF_COLOR_PRIMARIES_BT709
                    matrix_to_save = (avifMatrixCoefficients)1; // AVIF_MATRIX_COEFFICIENTS_BT709

                    switch (transfer_to_save) {
                    case 8: // AVIF_TRANSFER_CHARACTERISTICS_LINEAR
                        tmpcolorimage.convertToColorSpace(QColorSpace(QColorSpace::Primaries::SRgb, QColorSpace::TransferFunction::Linear));
                        break;
                    case 4: // AVIF_TRANSFER_CHARACTERISTICS_BT470M
                        tmpcolorimage.convertToColorSpace(QColorSpace(QColorSpace::Primaries::SRgb, 2.2f));
                        break;
                    case 5: // AVIF_TRANSFER_CHARACTERISTICS_BT470BG
                        tmpcolorimage.convertToColorSpace(QColorSpace(QColorSpace::Primaries::SRgb, 2.8f));
                        break;
                    default: // AVIF_TRANSFER_CHARACTERISTICS_SRGB + any other
                        tmpcolorimage.convertToColorSpace(QColorSpace(QColorSpace::Primaries::SRgb, QColorSpace::TransferFunction::SRgb));
                        transfer_to_save = (avifTransferCharacteristics)13;
                        break;
                    }
                } else if ((primaries_to_save != 2) &&
                           (transfer_to_save == 2)) {    //recognized primaries but other trc
                    transfer_to_save = (avifTransferCharacteristics)13;
                    tmpcolorimage.convertToColorSpace(tmpcolorimage.colorSpace().withTransferFunction(QColorSpace::TransferFunction::SRgb));
                } else { //unrecognized profile
                    primaries_to_save = (avifColorPrimaries)1; // AVIF_COLOR_PRIMARIES_BT709
                    transfer_to_save = (avifTransferCharacteristics)13;
                    matrix_to_save = (avifMatrixCoefficients)1; // AVIF_MATRIX_COEFFICIENTS_BT709
                    tmpcolorimage.convertToColorSpace(QColorSpace(QColorSpace::Primaries::SRgb, QColorSpace::TransferFunction::SRgb));
                }
            }
        }
#endif
        avif = avifImageCreate(tmpcolorimage.width(), tmpcolorimage.height(), save_depth, pixel_format);
        avif->matrixCoefficients = matrix_to_save;

#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
        avif->colorPrimaries = primaries_to_save;
        avif->transferCharacteristics = transfer_to_save;
#endif

        avifRGBImage rgb;
        avifRGBImageSetDefaults(&rgb, avif);
        rgb.rowBytes = tmpcolorimage.bytesPerLine();
        rgb.pixels = (uint8_t *)tmpcolorimage.constBits();

        if (save_depth > 8) { //10bit depth
            rgb.depth = 16;

            if (tmpcolorimage.hasAlphaChannel()) {
                avif->alphaRange = AVIF_RANGE_FULL;
            }

            rgb.format = AVIF_RGB_FORMAT_RGBA;
        } else { //8bit depth
            rgb.depth = 8;

            if (tmpcolorimage.hasAlphaChannel()) {
                rgb.format = AVIF_RGB_FORMAT_RGBA;
                avif->alphaRange = AVIF_RANGE_FULL;
            } else {
                rgb.format = AVIF_RGB_FORMAT_RGB;
            }
        }

        res = avifImageRGBToYUV(avif, &rgb);
        if (res != AVIF_RESULT_OK) {
            qWarning("ERROR in avifImageRGBToYUV: %s\n", avifResultToString(res));
            return false;
        }

        if ((save_depth > 8) && !tmpcolorimage.hasAlphaChannel()) {
            avifImageFreePlanes(avif, AVIF_PLANES_A);
        }
    }

    avifRWData raw = AVIF_DATA_EMPTY;
    avifEncoder *encoder = avifEncoderCreate();
    encoder->maxThreads = qBound(1, QThread::idealThreadCount(), 64);
    encoder->minQuantizer = minQuantizer;
    encoder->maxQuantizer = maxQuantizer;

    if (image.hasAlphaChannel()) {
        encoder->minQuantizerAlpha = AVIF_QUANTIZER_LOSSLESS;
        encoder->maxQuantizerAlpha = maxQuantizerAlpha;
    }

    encoder->speed = 8;

    res = avifEncoderWrite(encoder, avif, &raw);
    avifEncoderDestroy(encoder);
    avifImageDestroy(avif);

    if (res == AVIF_RESULT_OK) {
        qint64 status = device()->write((const char *)raw.data, raw.size);
        avifRWDataFree(&raw);

        if (status > 0) {
            return true;
        } else if (status == -1) {
            qWarning("Write error: %s\n", qUtf8Printable(device()->errorString()));
            return false;
        }
    } else {
        qWarning("ERROR: Failed to encode: %s\n", avifResultToString(res));
    }

    return false;
}


QVariant QAVIFHandler::option(ImageOption option) const
{
    if (!supportsOption(option) || !ensureParsed()) {
        return QVariant();
    }

    switch (option) {
    case Quality:
        return m_quality;
    case Size:
        return m_current_image.size();
    case Animation:
        if (imageCount() >= 2) {
            return true;
        } else {
            return false;
        }
    default:
        return QVariant();
    }
}

void QAVIFHandler::setOption(ImageOption option, const QVariant &value)
{
    switch (option) {
    case Quality:
        m_quality = value.toInt();
        if (m_quality > 100) {
            m_quality = 100;
        } else if (m_quality < 0) {
            m_quality = 52;
        }
        return;
    default:
        break;
    }
    QImageIOHandler::setOption(option, value);
}

bool QAVIFHandler::supportsOption(ImageOption option) const
{
    return option == Quality
           || option == Size
           || option == Animation;
}

int QAVIFHandler::imageCount() const
{
    if (!ensureParsed()) {
        return 0;
    }

    if (m_decoder->imageCount >= 1) {
        return m_decoder->imageCount;
    }
    return 0;
}

int QAVIFHandler::currentImageNumber() const
{
    if (m_parseState == ParseAvifNotParsed) {
        return -1;
    }

    if (m_parseState == ParseAvifError || !m_decoder) {
        return 0;
    }

    return m_decoder->imageIndex;
}

bool QAVIFHandler::jumpToNextImage()
{
    if (!ensureParsed()) {
        return false;
    }

    if (m_decoder->imageCount < 2) {
        return true;
    }

    if (m_decoder->imageIndex >= m_decoder->imageCount - 1) {   //start from begining
        avifDecoderReset(m_decoder);
    }

    avifResult decodeResult = avifDecoderNextImage(m_decoder);

    if (decodeResult != AVIF_RESULT_OK) {
        qWarning("ERROR: Failed to decode Next image in sequence: %s\n", avifResultToString(decodeResult));
        m_parseState = ParseAvifError;
        return false;
    }

    if ((m_container_width != m_decoder->image->width) ||
            (m_container_height != m_decoder->image->height)) {
        qWarning("Decoded image sequence size (%dx%d) do not match first image size (%dx%d)!\n",
                 m_decoder->image->width, m_decoder->image->height,
                 m_container_width, m_container_height);

        m_parseState = ParseAvifError;
        return false;
    }

    if (decode_one_frame()) {
        return true;
    } else {
        m_parseState = ParseAvifError;
        return false;
    }

}

bool QAVIFHandler::jumpToImage(int imageNumber)
{
    if (!ensureParsed()) {
        return false;
    }

    if (m_decoder->imageCount < 2) {   //not an animation
        if (imageNumber == 0) {
            return true;
        } else {
            return false;
        }
    }

    if (imageNumber < 0 || imageNumber >= m_decoder->imageCount) {   //wrong index
        return false;
    }

    if (imageNumber == m_decoder->imageCount) {   // we are here already
        return true;
    }

    avifResult decodeResult = avifDecoderNthImage(m_decoder, imageNumber);

    if (decodeResult != AVIF_RESULT_OK) {
        qWarning("ERROR: Failed to decode %d th Image in sequence: %s\n", imageNumber, avifResultToString(decodeResult));
        m_parseState = ParseAvifError;
        return false;
    }

    if ((m_container_width != m_decoder->image->width) ||
            (m_container_height != m_decoder->image->height)) {
        qWarning("Decoded image sequence size (%dx%d) do not match declared container size (%dx%d)!\n",
                 m_decoder->image->width, m_decoder->image->height,
                 m_container_width, m_container_height);

        m_parseState = ParseAvifError;
        return false;
    }

    if (decode_one_frame()) {
        return true;
    } else {
        m_parseState = ParseAvifError;
        return false;
    }
}

int QAVIFHandler::nextImageDelay() const
{
    if (!ensureParsed()) {
        return 0;
    }

    if (m_decoder->imageCount < 2) {
        return 0;
    }

    int delay_ms = 1000.0 * m_decoder->imageTiming.duration;
    if (delay_ms < 1) {
        delay_ms = 1;
    }
    return delay_ms;
}

int QAVIFHandler::loopCount() const
{
    if (!ensureParsed()) {
        return 0;
    }

    if (m_decoder->imageCount < 2) {
        return 0;
    }

    return 1;
}


