#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    unsigned char* Raw_Image_Data = nullptr;
    unsigned char* Processed_Image_Data = nullptr;
    int img_width, img_height;

    explicit MainWindow(QWidget *parent = nullptr);

    /*
     * @breif: Function to Read 24bit RGB Bitmap
     *
     * @param: filename: Relative string path to image file
     *         img_width: Image width is retured to this pointer
     *         img_height: Image height is returned to this pointer
     *
     * @return: Pointer to image data as 1D Array
     *
     * @knownbugs: Image Width and Height needs to be multiple of 4
    */
    unsigned char* readBMP(const char* filename , int* img_width , int* img_height);

    /*
     * @breif: Function to Read 24bit RGB Bitmap
     *
     * @param: filename: Relative string path to image file
     *         img_width: Image width is retured to this pointer
     *         img_height: Image height is returned to this pointer
     *
     * @return: Pointer to image data as 1D Array and 1Byte Per Pixel
     *
     * @knownbugs: Image Width and Height needs to be multiple of 4
    */
    unsigned char* readBMP1Ch(const char* filename , int* img_width , int* img_height);

    /*
     * @breif: Function to Convert RGB Image to Gray
     *
     * @param: data: Pointer to Data
     *         Width: Image width
     *         Height: Image height
     *
     * @return: Converted Image is Returned on the Source Pointer
    */
    void RGBtoGray(unsigned char* data, int width, int height);

    /*
     * @brief: Function to Write Image to File
     *
     * @param: stream: File Name as String
     *         data*:  Pointer to Image Data
     *         width: Image Width
     *         height: Image Height
     *
    */
    void writeImagePPM (std::ostream& stream, const uint8_t* data, int width, int height);

    /*
     * @brief: Function to Write Image to File
     *
     * @param: stream: File Name as String
     *         data*:  Pointer to Image Data
     *         width: Image Width
     *         height: Image Height
     *
    */
    void writeImagePGM (std::ostream& stream, const uint8_t* data, int width, int height);

    ~MainWindow();

private slots:
    void on_LowThres_Slider_valueChanged(int value);

    void on_HighThres_Slider_valueChanged(int value);

    void on_LoadImage_pushButton_clicked();

    void on_SaveImage_pushButton_2_clicked();

    void on_ProcessImage_pushButton_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
