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
#include <math.h>
#include <qthread.h>
#include <CL/cl.hpp>
#include <cannyfilter_opencl.h>
#include <chrono>
#include <stdlib.h>


using namespace std;

void GaussianBlur( 	std::vector<unsigned char>& h_input,
                            std::vector<unsigned char>& h_outputCpu,
                            std::size_t countX,
                            std::size_t countY);
unsigned char* GaussianBlurNew ( 	unsigned char* h_input,
                            unsigned long countX,
                            unsigned long  countY);


void SobelFilter(  	std::vector<unsigned char>& h_input,
                            std::vector<unsigned char>& h_outputCpu,
                            std::vector<unsigned char>& theta,
                            std::size_t countX,
                            std::size_t countY);

void NonMaxSuppression(  	std::vector<unsigned char>& h_input,
                                std::vector<unsigned char>& h_outputCpu,
                                std::vector<unsigned char>& theta,
                                std::size_t countX,
                                std::size_t countY);

void Hysterisis(   std::vector<unsigned char>& h_input,
                        std::vector<unsigned char>& h_outputCpu,
                        std::vector<unsigned char>& theta,
                        std::size_t countX,
                        std::size_t countY,
                        unsigned char lowThresh,
                        unsigned char highThresh);

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
    /* Read the OpenCL Class to check for available Platforms */
    string AvailablePlat[10];
    int NumofAvailPlat;
    CannyFilter_OpenCL Canny;
    Canny.GetPlatforms(AvailablePlat,&NumofAvailPlat);
    /* if num is greater than zero then add them to the ComboBox */
    if(NumofAvailPlat>0){
        QString Plat;
        for(uint8_t loop=0;loop<NumofAvailPlat;loop++){
           ui->Platform_comboBox->addItem(QString::fromStdString(AvailablePlat[loop]));
        }
    }

}

MainWindow::~MainWindow()
{
    delete ui;
    /* Delete Temp Files */
    QFile OldInFile(QDir::currentPath()+"/Temp_scaledimage.bmp");
    OldInFile.remove();
    QFile OldOutFile(QDir::currentPath()+"/Out.pgm");
    OldOutFile.remove();

    QFile OldGaussianMaskFile(QDir::currentPath()+"/GaussianMask.pgm");
    OldGaussianMaskFile.remove();
    QFile OldSobelFilterFile(QDir::currentPath()+"/SobelFilter.pgm");
    OldSobelFilterFile.remove();
    QFile OldThetaFile(QDir::currentPath()+"/Theta.pgm");
    OldThetaFile.remove();
    QFile OldMaxSupFile(QDir::currentPath()+"/MaxSup.pgm");
    OldMaxSupFile.remove();
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
            //int width,height;
            //unsigned char* test=readBMP1Ch("Temp_scaledimage.bmp",&width,&height);
            //ofstream test1 ("test.pgm", std::ios_base::binary);
            //writeImagePGM(test1,test,width,height);

        }
    }
}
void MainWindow::on_SaveImage_pushButton_2_clicked()
{
    /* Only enter into the code if a processed image exsists */
    QFile OldOutFile(QDir::currentPath()+"/Out.pgm");
    if(OldOutFile.exists()){
        /* Open Windows File Browser */
        //Limit Saving Selection to only PGM Files
        QString FilePath = QFileDialog::getSaveFileName(this,"Save File",QDir::homePath(),"Images(*.pgm)");
        /* Make a copy of the output Temporary file to out requested location */
        OldOutFile.copy(QDir::currentPath()+"/Out.pgm",FilePath);
        QMessageBox::information(this,"!","File Saved");

        /* Save Intermediate Results */
        FilePath.replace(".pgm","_Gauss.pgm");
        QFile OldGaussFile(QDir::currentPath()+"/GaussianMask.pgm");
        OldGaussFile.copy(QDir::currentPath()+"/GaussianMask.pgm",FilePath);

        FilePath.replace("_Gauss.pgm","_Sobel.pgm");
        QFile OldSobelFile(QDir::currentPath()+"/SobelFilter.pgm");
        OldSobelFile.copy(QDir::currentPath()+"/SobelFilter.pgm",FilePath);

        FilePath.replace("_Sobel.pgm","_Theta.pgm");
        QFile OldThetaFile(QDir::currentPath()+"/Theta.pgm");
        OldThetaFile.copy(QDir::currentPath()+"/Theta.pgm",FilePath);

        FilePath.replace("_Theta.pgm","_MaxSup.pgm");
        QFile OldMaxFile(QDir::currentPath()+"/MaxSup.pgm");
        OldMaxFile.copy(QDir::currentPath()+"/MaxSup.pgm",FilePath);
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
        unsigned char* Raw_Image_Data;
        int img_width,img_height;
        Raw_Image_Data=readBMP1Ch("Temp_scaledimage.bmp",&img_width, &img_height);
        /* Query the the Platform Combobox to find the requested Platform to run the algorithm on */
        if(ui->Platform_comboBox->currentText()=="CPU Core"){
            /* Run the Canny Filter */
            vector <unsigned char> inputvector(Raw_Image_Data,(Raw_Image_Data+(img_width*img_height)));
            vector<unsigned char> GaussianMask(img_width*img_height);
            vector<unsigned char> SobelFiltervector(img_width*img_height);
            vector<unsigned char> Theta(img_width*img_height);
            vector<unsigned char> MaxSup(img_width*img_height);
            vector<unsigned char> Hysterisisvector(img_width*img_height);
            /* Time Stamp */
            auto start = std::chrono::system_clock::now();
            /* Run the Algorithm */
            GaussianBlur(inputvector, GaussianMask, img_width, img_height);
            SobelFilter(GaussianMask, SobelFiltervector, Theta, img_width, img_height);
            NonMaxSuppression(SobelFiltervector, MaxSup, Theta, img_width, img_height);
            Hysterisis(MaxSup, Hysterisisvector, Theta, img_width, img_height, ui->LowThres_Slider->value(),
                       (ui->HighThres_Slider->value()));
            /* Time Stamp */
            auto end = std::chrono::system_clock::now();
            /* Update GUI for the Run Time */
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::string elapsedtime_mS = "Run Time: "+ std::to_string(elapsed.count())+" mS";
            ui->RunTime_label->setText(QString::fromStdString(elapsedtime_mS));

            /* Show the Processed Image on a new Window */
            /* Store On a Temp File */
            QFile OldOutFile(QDir::currentPath()+"/Out.pgm");
            OldOutFile.remove();
            ofstream outtemp ("Out.pgm", std::ios_base::binary);
            writeImagePGM(outtemp,Hysterisisvector.data(),img_width,img_height);
            outtemp.close();

            /* Store Intermediate Results */
            QFile OldGaussianMaskFile(QDir::currentPath()+"/GaussianMask.pgm");
            OldGaussianMaskFile.remove();
            ofstream GaussianMasktemp ("GaussianMask.pgm", std::ios_base::binary);
            writeImagePGM(GaussianMasktemp,GaussianMask.data(),img_width,img_height);
            GaussianMasktemp.close();

            QFile OldSobelFilterFile(QDir::currentPath()+"/SobelFilter.pgm");
            OldSobelFilterFile.remove();
            ofstream SobelFiltertemp ("SobelFilter.pgm", std::ios_base::binary);
            writeImagePGM(SobelFiltertemp,SobelFiltervector.data(),img_width,img_height);
            SobelFiltertemp.close();

            QFile OldThetaFile(QDir::currentPath()+"/Theta.pgm");
            OldThetaFile.remove();
            ofstream Thetatemp ("Theta.pgm", std::ios_base::binary);
            writeImagePGM(Thetatemp,Theta.data(),img_width,img_height);
            Thetatemp.close();

            QFile OldMaxSupFile(QDir::currentPath()+"/MaxSup.pgm");
            OldMaxSupFile.remove();
            ofstream MaxSuptemp ("MaxSup.pgm", std::ios_base::binary);
            writeImagePGM(MaxSuptemp,MaxSup.data(),img_width,img_height);
            MaxSuptemp.close();
            /*--*/

            /* Show the Processed Image on the Image Window Label */      
            if(ui->checkBox->isChecked()){
                QString display=QDir::currentPath()+"/SobelFilter.pgm";
                QPixmap DisplayImage(display);
                ui->ImageWindow->setPixmap(DisplayImage.scaled(ui->ImageWindow->width(),ui->ImageWindow->height(),Qt::KeepAspectRatio));
            }
            else{
                QString display=QDir::currentPath()+"/Out.pgm";
                QPixmap DisplayImage(display);
                ui->ImageWindow->setPixmap(DisplayImage.scaled(ui->ImageWindow->width(),ui->ImageWindow->height(),Qt::KeepAspectRatio));
            }
        }
        else{
            CannyFilter_OpenCL Canny;
            long int elapsed_time;

            unsigned char* Result = Canny.Detector(ui->Platform_comboBox->currentText().toStdString(),
                                                    Raw_Image_Data,img_width,img_height,
                                                    ui->LowThres_Slider->value(),ui->HighThres_Slider->value(),
                                                    elapsed_time, 0);
            /* Update GUI for Run Time */
            std::string elapsedtime_mS = "Run Time: "+ std::to_string(elapsed_time)+" mS";
            ui->RunTime_label->setText(QString::fromStdString(elapsedtime_mS));

            /* Show the Processed Image on a new Window */
            /* Store On a Temp File */
            QFile OldOutFile(QDir::currentPath()+"/Out.pgm");
            OldOutFile.remove();
            ofstream outtemp ("Out.pgm", std::ios_base::binary);
            writeImagePGM(outtemp,Result,img_width,img_height);
            outtemp.close();
            delete[] Result;

            /* Intermediate Results */
            unsigned char* GaussResult = Canny.Detector(ui->Platform_comboBox->currentText().toStdString(),
                                                    Raw_Image_Data,img_width,img_height,
                                                    ui->LowThres_Slider->value(),ui->HighThres_Slider->value(),
                                                    elapsed_time, 1);
            QFile OldGaussFile(QDir::currentPath()+"/GaussianMask.pgm");
            OldGaussFile.remove();
            ofstream Gausstemp ("GaussianMask.pgm", std::ios_base::binary);
            writeImagePGM(Gausstemp,GaussResult,img_width,img_height);
            Gausstemp.close();
            delete[] GaussResult;

            unsigned char* SobelResult = Canny.Detector(ui->Platform_comboBox->currentText().toStdString(),
                                                    Raw_Image_Data,img_width,img_height,
                                                    ui->LowThres_Slider->value(),ui->HighThres_Slider->value(),
                                                    elapsed_time, 2);
            QFile OldSobelFile(QDir::currentPath()+"/SobelFilter.pgm");
            OldSobelFile.remove();
            ofstream Sobeltemp ("SobelFilter.pgm", std::ios_base::binary);
            writeImagePGM(Sobeltemp,SobelResult,img_width,img_height);
            Sobeltemp.close();
            delete[] SobelResult;

            unsigned char* ThetaResult = Canny.Detector(ui->Platform_comboBox->currentText().toStdString(),
                                                    Raw_Image_Data,img_width,img_height,
                                                    ui->LowThres_Slider->value(),ui->HighThres_Slider->value(),
                                                    elapsed_time, 3);
            QFile OldThetaFile(QDir::currentPath()+"/Theta.pgm");
            OldThetaFile.remove();
            ofstream Thetatemp ("Theta.pgm", std::ios_base::binary);
            writeImagePGM(Thetatemp,ThetaResult,img_width,img_height);
            Thetatemp.close();
            delete[] ThetaResult;

            unsigned char* MaxSupResult = Canny.Detector(ui->Platform_comboBox->currentText().toStdString(),
                                                    Raw_Image_Data,img_width,img_height,
                                                    ui->LowThres_Slider->value(),ui->HighThres_Slider->value(),
                                                    elapsed_time, 4);
            QFile OldMaxSupFile(QDir::currentPath()+"/MaxSup.pgm");
            OldMaxSupFile.remove();
            ofstream MaxSuptemp ("MaxSup.pgm", std::ios_base::binary);
            writeImagePGM(MaxSuptemp,MaxSupResult,img_width,img_height);
            MaxSuptemp.close();
            delete[] MaxSupResult;

            /* Show the Processed Image on the Image Window Label */
            if(ui->checkBox->isChecked()){
                QString display=QDir::currentPath()+"/SobelFilter.pgm";
                QPixmap DisplayImage(display);
                ui->ImageWindow->setPixmap(DisplayImage.scaled(ui->ImageWindow->width(),ui->ImageWindow->height(),Qt::KeepAspectRatio));
            }
            else{
                QString display=QDir::currentPath()+"/Out.pgm";
                QPixmap DisplayImage(display);
                ui->ImageWindow->setPixmap(DisplayImage.scaled(ui->ImageWindow->width(),ui->ImageWindow->height(),Qt::KeepAspectRatio));
            }
        }
        /* Delete the Buffer for the Input Image */
        delete[] Raw_Image_Data;
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

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
int getIndexGlobal(std::size_t countX, int i, int j) {
    return j * countX + i;
}
unsigned long getIndexGlobalNew(unsigned long countX, int i, int j) {
    return (static_cast<unsigned long>(j)  * countX) + static_cast<unsigned long>(i);
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
    if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
        return 0;
    else
        return a[getIndexGlobal(countX, i, j)];

}
// params: input, output, width, height, col, row
int getValueGlobal_int(const std::vector<unsigned char>& a, std::size_t countX, std::size_t countY, int i, int j) {
    if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
        return 0;
    else
        return a[getIndexGlobal(countX, i, j)];

}

int getValueGlobal_intNew(unsigned char* a, unsigned long countX, unsigned long countY, int i, int j) {
    if (i < 0 ||static_cast<unsigned long> (i) >= countX || j < 0 || static_cast<unsigned long> (j) >= countY)
        return 0;
    else
        return *(a+(getIndexGlobalNew(countX, i, j)));

}


////////////////////====================================////////////////////////////////
const float GaussianMask[3][3] = {  {0.0625, 0.125, 0.0625},
                                    {0.1250, 0.250, 0.1250},
                                    {0.0625, 0.125, 0.0625} };

// Define Sobel Filter Mask

// vertical
const int SobelMask_Gx[3][3] = { {-1, 0, 1},
                                 {-2, 0, 2},
                                 {-1, 0, 1} };

// horizontal
const int SobelMask_Gy[3][3] = {  {-1,-2,-1},
                                  { 0, 0, 0},
                                  { 1, 2, 1} };

const float PI = 3.14159265;

void GaussianBlur( 	std::vector<unsigned char>& h_input,
                            std::vector<unsigned char>& h_outputCpu,
                            std::size_t countX, // width
                            std::size_t countY) // height
{
    int img_height = (int) countY;
    int img_width  = (int) countX;
    for (int row = 1; row < img_height; row++)
    {
            for (int col = 1; col < img_width; col++)
            {
                int sum = 0;

                for(int i=0; i<3; i++) // row
                {
                    for(int j=0; j<3; j++) // column
                        {
                            sum += GaussianMask[i][j] * getValueGlobal_int(h_input, countX, countY, col+j-1, row+i-1);
                        }
                }
                h_outputCpu[getIndexGlobal(countX, col, row)] = min(255, max(0,sum));

            }
    }
}

unsigned char* GaussianBlurNew ( 	unsigned char* h_input,
                            unsigned long countX,
                        unsigned long  countY){

    unsigned char* h_outputCpu =new unsigned char[countX*countY];

    int img_height = static_cast<int>(countY) ;
    int img_width  = static_cast<int>(countX);
    for (int row = 1; row < img_height; row++)
    {
            for (int col = 1; col < img_width; col++)
            {
                int sum = 0;

                for(int i=0; i<3; i++) // row
                {
                    for(int j=0; j<3; j++) // column
                        {
                            sum += GaussianMask[i][j] * getValueGlobal_intNew(h_input, countX, countY, col+j-1, row+i-1);
                        }
                }
                *(h_outputCpu+getIndexGlobalNew(countX, col, row)) = static_cast<unsigned char>(min(255, max(0,sum)));

            }
    }

    return h_outputCpu;

}


void SobelFilter( 	std::vector<unsigned char>& h_input,
                            std::vector<unsigned char>& h_outputCpu,
                            std::vector<unsigned char>& theta,
                            std::size_t countX,
                            std::size_t countY)
{
    int img_height = (int) countY;
    int img_width  = (int) countX;

    for (int row = 1; row < img_height; row++)
        {
                for (int col = 1; col < img_width; col++)
                {
                    float Gx = 0, Gy = 0, angle = 0;

                    for(int i=0; i<3; i++) // row
                    {
                        for(int j=0; j<3; j++) // column
                            {
                                Gx += SobelMask_Gx[i][j] * getValueGlobal_int(h_input, countX, countY, col+j-1, row+i-1);
                                Gy += SobelMask_Gy[i][j] * getValueGlobal_int(h_input, countX, countY, col+j-1, row+i-1);
                            }
                    }
                    h_outputCpu[getIndexGlobal(countX, col, row)] = min(255, max(0,(int)hypot(Gx, Gy)));

                    angle = atan2(Gy, Gx);

                    if (angle < 0)
                    {
                        angle = fmod((angle + 2 * PI), (2 * PI));
                    }

                    angle = (180/PI)*(angle * (PI/8) + PI/8-0.0001);

                    theta[getIndexGlobal(countX, col, row)]= (((unsigned char)    (angle/ 45)) * 45 ) % 180;

                }
        }

}

void NonMaxSuppression( 	std::vector<unsigned char>& h_input,
                                std::vector<unsigned char>& h_outputCpu,
                                std::vector<unsigned char>& theta,
                                std::size_t countX,
                                std::size_t countY)
{
    int img_height = (int) countY;
    int img_width  = (int) countX;

    for (int row = 1; row < img_height-1; row++)
    {
            for (int col = 1; col < img_width-1; col++)
            {
                switch(theta[getIndexGlobal(countX, col, row)])
                {
                case 0:
                    if( (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col-1, row)]) ||
                        (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col+1, row)]) )
                    {
                            h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
                    }
                    else
                    {
                            h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
                    }
                    break;

                case 45:
                    if( (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col+1, row-1)]) ||
                        (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col-1, row+1)]) )
                    {
                            h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
                    }
                    else
                    {
                            h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
                    }
                    break;

                case 90:
                    if( (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col, row-1)]) ||
                        (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col, row+1)]) )
                    {
                            h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
                    }
                    else
                    {
                            h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
                    }
                    break;

                case 135:
                    if( (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col-1, row-1)]) ||
                        (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col+1, row+1)]) )
                    {
                            h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
                    }
                    else
                    {
                            h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
                    }
                    break;

                default:
                    h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
                    break;

                }
            }

    }

}

void Hysterisis(  std::vector<unsigned char>& h_input,
                        std::vector<unsigned char>& h_outputCpu,
                        std::vector<unsigned char>& theta,
                        std::size_t countX,
                        std::size_t countY,
                        unsigned char lowThresh,
                        unsigned char highThresh)
{
    int img_height = (int) countY;
    int img_width  = (int) countX;
    const unsigned char WHITE = 255;

    for (int row = 1; row < img_height; row++)
    {
            for (int col = 1; col < img_width; col++)
            {
                if(h_input[getIndexGlobal(countX, col, row)] <= lowThresh)
                    h_outputCpu[getIndexGlobal(countX, col, row)] = 0;

                else if(h_input[getIndexGlobal(countX, col, row)] >= highThresh)
                    h_outputCpu[getIndexGlobal(countX, col, row)] = WHITE;

                else if(	(h_outputCpu[getIndexGlobal(countX, col-1, row-1)] == WHITE) ||
                            (h_outputCpu[getIndexGlobal(countX, col  , row-1)] == WHITE) ||
                            (h_outputCpu[getIndexGlobal(countX, col+1, row-1)] == WHITE) ||
                            (h_outputCpu[getIndexGlobal(countX, col-1, row  )] == WHITE) ||
                            (h_outputCpu[getIndexGlobal(countX, col+1, row  )] == WHITE) ||
                            (h_outputCpu[getIndexGlobal(countX, col-1, row+1)] == WHITE) ||
                            (h_outputCpu[getIndexGlobal(countX, col  , row+1)] == WHITE) ||
                            (h_outputCpu[getIndexGlobal(countX, col+1, row+1)] == WHITE) 	)
                    h_outputCpu[getIndexGlobal(countX, col, row)] = WHITE;

                else
                    h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
            }
    }


}

