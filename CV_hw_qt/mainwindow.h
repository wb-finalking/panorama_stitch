#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QMainWindow>
#include <QActionGroup>
#include <QGridLayout>
#include <QFileDialog>
#include <QPushButton>
#include <QDebug>
#include "ui_mainwindow.h"
#include "Paint.h"
#include "Stitch.h"
#include "FileReading.h"

class mainwindow : public QMainWindow
{
	Q_OBJECT

public:
	mainwindow(QWidget *parent = 0);
	~mainwindow();

private:
	vector<cv::Mat> res;
	int index;

private:
	Ui::mainwindowClass ui;

	private slots:
	void openImages();
	void getFileDirectory();
	void save();
	void nextRes();
	void preRes();

private:
	void createActions();
	void createMenus();
	cv::Mat QImageToMat(QImage image);
	QImage MatToQImage(const cv::Mat& mat);

	QMenu * fileMenu;

	QAction * openImagesAct;
	QAction * getDirectory;
	QAction * saveDirectory;

	Paint* paint;
	QWidget* cw;

	QHBoxLayout *HorizontalLayout;
	QVBoxLayout *VerticalLayout;
	QPushButton *next;
	QPushButton *pre;

};

#endif // MAINWINDOW_H
