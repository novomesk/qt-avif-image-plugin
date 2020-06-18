#ifndef QAVIFHANDLER_P_H
#define QAVIFHANDLER_P_H

#include <QImage>
#include <QVariant>
#include <qimageiohandler.h>
#include <QByteArray>
#include <avif/avif.h>

class QAVIFHandler : public QImageIOHandler
{
public:
  QAVIFHandler();
  ~QAVIFHandler();

  bool canRead() const override;
  bool read ( QImage *image ) override;
  bool write ( const QImage &image ) override;

  static bool canRead ( QIODevice *device );

  QVariant option ( ImageOption option ) const override;
  void setOption ( ImageOption option, const QVariant &value ) override;
  bool supportsOption ( ImageOption option ) const override;

  int imageCount() const override;
  int currentImageNumber() const override;
  bool jumpToNextImage() override;
  bool jumpToImage ( int imageNumber ) override;

  int nextImageDelay() const override;

  int loopCount() const override;
private:
  bool ensureParsed() const;
  bool ensureDecoder();
  bool decode_one_frame();

  enum ParseAvifState
  {
    ParseAvifError = -1,
    ParseAvifNotParsed = 0,
    ParseAvifSuccess = 1
  };

  ParseAvifState m_parseState;
  int m_quality;

  uint32_t m_container_width;
  uint32_t m_container_height;

  QByteArray m_rawData;
  avifROData m_rawAvifData;

  avifDecoder *m_decoder;
  QImage        m_current_image;

  bool m_must_jump_to_next_image;
};

#endif // QAVIFHANDLER_P_H
