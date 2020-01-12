/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.13.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QLabel *ImageWindow;
    QComboBox *Platform_comboBox;
    QLabel *Platfroms_Title;
    QPushButton *LoadImage_pushButton;
    QPushButton *SaveImage_pushButton_2;
    QPushButton *ProcessImage_pushButton;
    QLabel *RunTime_label;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_2;
    QLabel *LowThres_Title;
    QLabel *HighThres_Title;
    QVBoxLayout *verticalLayout;
    QSlider *LowThres_Slider;
    QSlider *HighThres_Slider;
    QCheckBox *checkBox;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(758, 486);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        ImageWindow = new QLabel(centralWidget);
        ImageWindow->setObjectName(QString::fromUtf8("ImageWindow"));
        ImageWindow->setGeometry(QRect(30, 30, 541, 311));
        ImageWindow->setAutoFillBackground(false);
        ImageWindow->setFrameShape(QFrame::Box);
        ImageWindow->setFrameShadow(QFrame::Sunken);
        Platform_comboBox = new QComboBox(centralWidget);
        Platform_comboBox->setObjectName(QString::fromUtf8("Platform_comboBox"));
        Platform_comboBox->setGeometry(QRect(580, 90, 161, 22));
        Platfroms_Title = new QLabel(centralWidget);
        Platfroms_Title->setObjectName(QString::fromUtf8("Platfroms_Title"));
        Platfroms_Title->setGeometry(QRect(600, 60, 131, 21));
        QFont font;
        font.setPointSize(10);
        font.setBold(true);
        font.setWeight(75);
        Platfroms_Title->setFont(font);
        LoadImage_pushButton = new QPushButton(centralWidget);
        LoadImage_pushButton->setObjectName(QString::fromUtf8("LoadImage_pushButton"));
        LoadImage_pushButton->setGeometry(QRect(600, 170, 121, 41));
        SaveImage_pushButton_2 = new QPushButton(centralWidget);
        SaveImage_pushButton_2->setObjectName(QString::fromUtf8("SaveImage_pushButton_2"));
        SaveImage_pushButton_2->setGeometry(QRect(600, 230, 121, 41));
        ProcessImage_pushButton = new QPushButton(centralWidget);
        ProcessImage_pushButton->setObjectName(QString::fromUtf8("ProcessImage_pushButton"));
        ProcessImage_pushButton->setGeometry(QRect(570, 360, 171, 71));
        RunTime_label = new QLabel(centralWidget);
        RunTime_label->setObjectName(QString::fromUtf8("RunTime_label"));
        RunTime_label->setGeometry(QRect(630, 450, 91, 16));
        layoutWidget = new QWidget(centralWidget);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(110, 370, 401, 54));
        horizontalLayout = new QHBoxLayout(layoutWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        LowThres_Title = new QLabel(layoutWidget);
        LowThres_Title->setObjectName(QString::fromUtf8("LowThres_Title"));
        QFont font1;
        font1.setBold(true);
        font1.setWeight(75);
        LowThres_Title->setFont(font1);

        verticalLayout_2->addWidget(LowThres_Title);

        HighThres_Title = new QLabel(layoutWidget);
        HighThres_Title->setObjectName(QString::fromUtf8("HighThres_Title"));
        HighThres_Title->setFont(font1);

        verticalLayout_2->addWidget(HighThres_Title);


        horizontalLayout->addLayout(verticalLayout_2);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        LowThres_Slider = new QSlider(layoutWidget);
        LowThres_Slider->setObjectName(QString::fromUtf8("LowThres_Slider"));
        LowThres_Slider->setMinimum(1);
        LowThres_Slider->setMaximum(254);
        LowThres_Slider->setValue(10);
        LowThres_Slider->setOrientation(Qt::Horizontal);
        LowThres_Slider->setInvertedAppearance(false);
        LowThres_Slider->setInvertedControls(false);

        verticalLayout->addWidget(LowThres_Slider);

        HighThres_Slider = new QSlider(layoutWidget);
        HighThres_Slider->setObjectName(QString::fromUtf8("HighThres_Slider"));
        HighThres_Slider->setMinimum(1);
        HighThres_Slider->setMaximum(254);
        HighThres_Slider->setValue(30);
        HighThres_Slider->setOrientation(Qt::Horizontal);

        verticalLayout->addWidget(HighThres_Slider);


        horizontalLayout->addLayout(verticalLayout);

        checkBox = new QCheckBox(centralWidget);
        checkBox->setObjectName(QString::fromUtf8("checkBox"));
        checkBox->setGeometry(QRect(610, 130, 101, 19));
        MainWindow->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);
        QObject::connect(LowThres_Slider, SIGNAL(valueChanged(int)), ProcessImage_pushButton, SLOT(click()));
        QObject::connect(HighThres_Slider, SIGNAL(valueChanged(int)), ProcessImage_pushButton, SLOT(click()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        ImageWindow->setText(QString());
        Platfroms_Title->setText(QCoreApplication::translate("MainWindow", "Available Platforms", nullptr));
        LoadImage_pushButton->setText(QCoreApplication::translate("MainWindow", "Load Image", nullptr));
        SaveImage_pushButton_2->setText(QCoreApplication::translate("MainWindow", "Save Image", nullptr));
        ProcessImage_pushButton->setText(QCoreApplication::translate("MainWindow", "Run Canny Filter", nullptr));
        RunTime_label->setText(QString());
        LowThres_Title->setText(QCoreApplication::translate("MainWindow", "Low Threshold", nullptr));
        HighThres_Title->setText(QCoreApplication::translate("MainWindow", "High Threshold", nullptr));
        checkBox->setText(QCoreApplication::translate("MainWindow", "Switch to Sobel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
