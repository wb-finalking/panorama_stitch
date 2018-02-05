#include "mainwindow.h"

mainwindow::mainwindow(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	index = 0;
	paint = 0;
	if (!paint)
		paint = new Paint(this);

	createActions();
	createMenus();

	next = new QPushButton("Next");
	pre = new QPushButton("Pre");

	cw = new QWidget(this);
	setCentralWidget(cw);
	VerticalLayout = new QVBoxLayout(cw);
	HorizontalLayout = new QHBoxLayout;
	VerticalLayout->addWidget(paint);
	VerticalLayout->addLayout(HorizontalLayout);
	
	HorizontalLayout->addWidget(pre);
	HorizontalLayout->addWidget(next);

	connect(pre, SIGNAL(clicked()), this, SLOT(preRes()));
	connect(next, SIGNAL(clicked()), this, SLOT(nextRes()));
}

mainwindow::~mainwindow()
{

}

void mainwindow::createActions(){

	// File
	openImagesAct = new QAction(tr("&Open Images"), this);
	openImagesAct->setShortcut(QKeySequence::Open);
	openImagesAct->setStatusTip(tr("Open source/target images"));
	connect(openImagesAct, SIGNAL(triggered()), this, SLOT(openImages()));

	getDirectory = new QAction(tr("&File Path"), this);
	//getDirectory->setShortcut(QKeySequence::Open);
	getDirectory->setStatusTip(tr("Get File path"));
	connect(getDirectory, SIGNAL(triggered()), this, SLOT(getFileDirectory()));

	saveDirectory = new QAction(tr("&Save"), this);
	//getDirectory->setShortcut(QKeySequence::Open);
	saveDirectory->setStatusTip(tr("Get Saving path"));
	connect(saveDirectory, SIGNAL(triggered()), this, SLOT(save()));


}

void mainwindow::createMenus(){

	fileMenu = new QMenu(tr("&File"), this);
	//fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(openImagesAct);
	fileMenu->addAction(getDirectory);
	fileMenu->addAction(saveDirectory);
	menuBar()->addMenu(fileMenu);
}

void mainwindow::openImages(){

	QString source = QFileDialog::getOpenFileName(this,
		tr("Open Source Image"), QDir::currentPath(), tr("Image Files (*.png *.jpg *.bmp)"));
	QImage image(source);

	//if (!paint)
	//	paint = new Paint(this);

	/*cw = new QWidget(this);
	TopRightLayout = new QHBoxLayout(cw);
	TopRightLayout->addWidget(paint);
	setCentralWidget(cw);*/
	paint->setImage(image);
	paint->setFixedSize(image.width(), image.height());
	resize(image.width(), image.height());
	paint->show();
}

void mainwindow::getFileDirectory()
{
	QString file_path = QFileDialog::getExistingDirectory(this, "请选择路径...", "./");
	string file_folder = file_path.toStdString();
	//string file_folder("dataset1\\");

	while (file_folder.find("/")!=string::npos)
		file_folder.replace(file_folder.find("/"),1,"\\");
	file_folder = file_folder + "\\";

	vector<string> image_files;

	getAllFiles(file_folder, image_files);

	vector<cv::Mat> src_imgs(image_files.size());

	if (image_files.size() == 0)
		return;

	index = 0;
	for (int i = 0; i < image_files.size(); i++) {
		src_imgs[i] = cv::imread(image_files[i]);
	}

	Stitch stitch;
	//cv::Mat res = stitch.stitching(src_imgs);
	res = stitch.stitcher(src_imgs);

	preRes();

	//for (int i = 0; i < res.size(); i++)
	//{
	//	string name;
	//	stringstream ss;
	//	ss << "res"<< i ;
	//	ss >> name;
	//	cv::imshow(name, res[i]);
	//}

	//cv::waitKey(0);
}

void mainwindow::save()
{
	QString file_path = QFileDialog::getExistingDirectory(this, "请选择路径...", "./");
	string file_folder = file_path.toStdString();
	//string file_folder("dataset1\\");

	while (file_folder.find("/") != string::npos)
		file_folder.replace(file_folder.find("/"), 1, "\\");
	file_folder = file_folder + "\\";

	for (int i = 0; i < res.size(); i++)
	{
		string name;
		stringstream ss;
		ss << "res" << i<<".jpg";
		ss >> name;
		cv::imwrite(file_folder+name, res[i]);
	}

}

void mainwindow::preRes()
{
	if (index > 0)
		index = index - 1;

	if (index <res.size())
	{		
		QImage image = MatToQImage(res[index]);

		int width, height;
		width = image.width();
		height = image.height();

		float resize_factor;
		if (width < height) {
			resize_factor = 1500.0 / height;
		}
		else {
			resize_factor = 1500.0 / width;
		}
		if (resize_factor >= 1) {
			resize_factor = 1;
		}
		else {
			image = image.scaled(width*resize_factor, height*resize_factor, Qt::KeepAspectRatio);
		}

		paint->setImage(image);
		paint->setFixedSize(image.width(), image.height());
		resize(image.width(), image.height());
		paint->show();

		
	}
}

void mainwindow::nextRes()
{
	if (index < res.size()-1)
		index = index + 1;

	QImage image = MatToQImage(res[index]);

	int width, height;
	width = image.width();
	height = image.height();

	float resize_factor;
	if (width < height) {
		resize_factor = 1500.0 / height;
	}
	else {
		resize_factor = 1500.0 / width;
	}
	if (resize_factor >= 1) {
		resize_factor = 1;
	}
	else {
		image = image.scaled(width*resize_factor, height*resize_factor, Qt::KeepAspectRatio);
	}

	paint->setImage(image);
	paint->setFixedSize(image.width(), image.height());
	resize(image.width(), image.height());
	paint->show();
}

cv::Mat mainwindow::QImageToMat(QImage image)
{
	cv::Mat mat;
	switch (image.format())
	{
	case QImage::Format_ARGB32:
	case QImage::Format_RGB32:
	case QImage::Format_ARGB32_Premultiplied:
		mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
		break;
	case QImage::Format_RGB888:
		mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
		cv::cvtColor(mat, mat, CV_BGR2RGB);
		break;
	case QImage::Format_Indexed8:
		mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
		break;
	}
	return mat;
}

QImage mainwindow::MatToQImage(const cv::Mat& mat)
{
	// 8-bits unsigned, NO. OF CHANNELS = 1    
	if (mat.type() == CV_8UC1)
	{
		QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
		// Set the color table (used to translate colour indexes to qRgb values)    
		image.setColorCount(256);
		for (int i = 0; i < 256; i++)
		{
			image.setColor(i, qRgb(i, i, i));
		}
		// Copy input Mat    
		uchar *pSrc = mat.data;
		for (int row = 0; row < mat.rows; row++)
		{
			uchar *pDest = image.scanLine(row);
			memcpy(pDest, pSrc, mat.cols);
			pSrc += mat.step;
		}
		return image;
	}
	// 8-bits unsigned, NO. OF CHANNELS = 3    
	else if (mat.type() == CV_8UC3)
	{
		// Copy input Mat    
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat    
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	else if (mat.type() == CV_8UC4)
	{
		qDebug() << "CV_8UC4";
		// Copy input Mat    
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat    
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
		return image.copy();
	}
	else
	{
		qDebug() << "ERROR: Mat could not be converted to QImage.";
		return QImage();
	}
}