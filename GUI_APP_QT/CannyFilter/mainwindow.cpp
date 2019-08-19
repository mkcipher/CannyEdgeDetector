#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <string.h>
#include <qstring.h>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <fstream>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <qgraphicsview.h>
#include <qgraphicsscene.h>
#include <qlabel.h>
#include <qfile.h>

#include<CannyEdgeDetector.h>

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    /* Fix Size of Main Window */
    this->statusBar()->setSizeGripEnabled(false);
    setFixedSize(width(), height());

    /* SetUp Title of Window */
    QWidget::setWindowTitle ( "Canny Filter" );
    /* SetUp Available Platforms */
    ui->Platform_comboBox->addItem("CPU Core");

    /* Set Default Values for Threshold */
    //ui->LowThres_Slider->setValue(10);
    //ui->HighThres_Slider->setValue(30);

}

MainWindow::~MainWindow()
{
    delete ui;
    /* Delete Temp Files */
    QFile OldInFile(QDir::currentPath()+"/Temp_scaledimage.bmp");
    OldInFile.remove();
    QFile OldOutFile(QDir::currentPath()+"/Out.ppm");
    OldOutFile.remove();
}

/* UI CONTROL Functions For Threshold Sliders */
void MainWindow::on_LowThres_Slider_valueChanged(int value)
{
    /* Check for Value not being higher than the High Threshold */
    if(value>ui->HighThres_Slider->value()){
        ui->LowThres_Slider->setValue(ui->HighThres_Slider->value()-1);
    }

}
void MainWindow::on_HighThres_Slider_valueChanged(int value)
{
    /* Check for Value not being lower than the Low Threshold */
    if(value<ui->LowThres_Slider->value()){
        ui->HighThres_Slider->setValue(ui->LowThres_Slider->value()+1);
    }
}

/* UI CONTROL Functions for Load and Save Buttons */
void MainWindow::on_LoadImage_pushButton_clicked()
{
    /* Open Windows File Browser */
    //Limit Selection to Only BitMap Files
    QString FilePath = QFileDialog::getOpenFileName(this,"Load File",QDir::homePath(),"Images(*.bmp)");
    QMessageBox::information(this,"Selection",FilePath);
    /* Show File on Image Box Label */
    if(!FilePath.isEmpty())
    {
        QImage image(FilePath);

        if(image.isNull())
        {
            /* Error Condition */
            QMessageBox::information(this,"Image Viewer","Error Displaying image");
        }
        else
        {
            /* Show Scaled Image on the Image Box */
            QPixmap DisplayImage(FilePath);
            ui->ImageWindow->setPixmap(DisplayImage.scaled(ui->ImageWindow->width(),ui->ImageWindow->height(),Qt::KeepAspectRatio));
            /* Temporarily store the Read Image as a scaled down version for later processing */
            /* Right now Fixed to 640x480 Resolution */
            QImage image_scaled = image.scaled(640, 480,Qt::IgnoreAspectRatio);
            //QImage image_scaled = image.scaled(1080, 624,Qt::IgnoreAspectRatio);
            image_scaled.save ("Temp_scaledimage.bmp", "bmp", 100);

            /* Temporary DEBUG */
            int width,height;
            unsigned char* test=readBMP1Ch("Temp_scaledimage.bmp",&width,&height);
            ofstream test1 ("test.pgm", std::ios_base::binary);
            writeImagePGM(test1,test,width,height);

        }
    }
}
void MainWindow::on_SaveImage_pushButton_2_clicked()
{
    /* Only enter into the code if a processed image exsists */
    QFile OldOutFile(QDir::currentPath()+"/Out.ppm");
    if(OldOutFile.exists()){
        /* Open Windows File Browser */
        //Limit Saving Selection to only PPM Files
        QString FilePath = QFileDialog::getSaveFileName(this,"Save File",QDir::homePath(),"Images(*.ppm)");
        /* Make a copy of the output Temporary file to out requested location */
        OldOutFile.copy(QDir::currentPath()+"/Out.ppm",FilePath);
        QMessageBox::information(this,"!","File Saved");
    }
    else {
        /* Error Condition */
        QMessageBox::information(this,"Error","No Processed Image Present");
    }
}
void MainWindow::on_ProcessImage_pushButton_clicked()
{
    /* Only Start if Scaled Image is Present */
    QFile OldInFile(QDir::currentPath()+"/Temp_scaledimage.bmp");
    if(OldInFile.exists()){
        /* Read the Scaled Image */
        this->Raw_Image_Data=readBMP("Temp_scaledimage.bmp", &this->img_width, &this->img_height);
        /* Query the the Platform Combobox to find the requested Platform to run the algorithm on */
        if(ui->Platform_comboBox->currentText()=="CPU Core"){
            /* Run the Canny Filter CPU Implementation */
            CannyFilter canny;
            this->Processed_Image_Data=canny.Detector(this->Raw_Image_Data,static_cast<unsigned int>(this->img_width),
                                                      static_cast<unsigned int>(this->img_height),1.0f,
                                                      static_cast<uint8_t>(ui->LowThres_Slider->value()),
                                                      static_cast<uint8_t>(ui->HighThres_Slider->value()));
            //uint8_t* Filter_Result = canny.Detector(this->Raw_Image_Data,(uint) img_width,(uint) img_height,1.0f,10,30);

            /* Show the Processed Image on a new Window */
            /* Store On a Temp File */
            QFile OldOutFile(QDir::currentPath()+"/Out.ppm");
            OldOutFile.remove();
            ofstream outtemp ("Out.ppm", std::ios_base::binary);
            writeImagePPM(outtemp,this->Processed_Image_Data,this->img_width,this->img_height);
            /* Open the stored Image */
            //QPixmap image(QDir::currentPath()+"/Out.ppm");
            //QLabel* imageLabel = new QLabel;
            //imageLabel->setPixmap(image);
            //imageLabel->show();
            /* Show the Processed Image on the Image Window Label */
            QPixmap DisplayImage(QDir::currentPath()+"/Out.ppm");
            ui->ImageWindow->setPixmap(DisplayImage.scaled(ui->ImageWindow->width(),ui->ImageWindow->height(),Qt::KeepAspectRatio));
        }
        /* Delete the Buffer for the Input Image */
        delete[] this->Raw_Image_Data;
    }
    else {
        /* Error Condition */
        QMessageBox::information(this,"Error","No Image Loaded");
    }
}



unsigned char* MainWindow::readBMP(const char* filename , int* img_width , int* img_height)
{
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width, height;
    memcpy(&width, info + 18, sizeof(int));
    memcpy(&height, info + 22, sizeof(int));

    /* Check for height signedness */
    int heightSign = 1;
    if (height < 0){
        heightSign = -1;
    }

    /* Read Image Data */
    unsigned long long size = static_cast<unsigned long long>(3 * width * abs(height));
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);

    unsigned char* data2 = new unsigned char[size];
    /* Invert Image */
    if(heightSign == 1){
        long int index1=0;
        long int index2=(3*width)*(height-2);
        long int index3=0;
        while(index2>=0){
            *(data2+index2)=*(data+index1);
            index3=index1+1;
            for(long int loop=index2+1;loop<index2+(width*3);loop++){
                *(data2+loop)=*(data+index3);
                index3++;
            }
            index1+=3*width;
            index2-=3*width;
        }
    }
    else {
        data2=data;
    }

    /* Delete Unneeded Data */
    delete [] data;

    /* Return values */
    *img_width=width;
    *img_height=height;
    return data2;
}

unsigned char* MainWindow::readBMP1Ch(const char* filename , int* img_width , int* img_height)
{
    /* Read the BMP File */
    unsigned char* File = this->readBMP(filename,img_width,img_height);

    /* Convert RGB Data to GrayScale */
    RGBtoGray(File,*img_width,*img_height);

    /* Create New Data Buffer */
    int width=*img_width;
    int height=*img_height;
    unsigned char* data = new unsigned char[width*height];

    /* Read the First Byte of All Pixels */
    long int index = 0;
    while(index<(width*height)){
        *(data+index)=*(File+(index*3));
        index++;
    }

    /* Delete Unneeded Data */
    delete [] File;

    return data;
}

/*
 * @brief: Function to Write Image to File
 *
 * @param: stream: File Name as String
 *         data*:  Pointer to Image Data
 *         width: Image Width
 *         height: Image Height
 *
*/
void MainWindow::writeImagePPM (std::ostream& stream, const uint8_t* data, int width, int height) {
  stream << "P6\n" << width << " " << height << "\n255\n";
  stream.write ((const char*) data, width * height * 3);
}

/*
 * @brief: Function to Write Image to File
 *
 * @param: stream: File Name as String
 *         data*:  Pointer to Image Data
 *         width: Image Width
 *         height: Image Height
 *
*/
void MainWindow::writeImagePGM (std::ostream& stream, const uint8_t* data, int width, int height) {
  stream << "P5\n" << width << " " << height << "\n255\n";
  stream.write ((const char*) data, width * height);
}

/*
 * @breif: Function to Convert RGB Image to Gray
 *
 * @param: data: Pointer to Data
 *         Width: Image width
 *         Height: Image height
 *
 * @return: Converted Image is Returned on the Source Pointer
*/
void MainWindow::RGBtoGray(unsigned char* data, int width, int height)
{
    unsigned long i;
    float gray, blue, green, red;

    for (int x = 0; x < height; x++)
    {
        for (int y = 0; y < width; y++)
        {

            // Calculate position of a pixel in bitmap table with (x,y)
            // Bitmap table is arranged as follows
            // R1,G1, B1, R2, G2, B2, ...
            i = (unsigned long) (x * 3 * width + 3 * y);

            // Assigning values of BGR
            blue  = *(data + i);
            green = *(data + i + 1);
            red   = *(data + i + 2);

            // Convert values from RGB to gray
            gray = (uint8_t) (0.299 * red + 0.587 * green + 0.114 * blue);

            // Ultimately making picture grayscale.
            *(data + i) =
                *(data + i + 1) =
                *(data + i + 2) = gray;
        }
    }
}
