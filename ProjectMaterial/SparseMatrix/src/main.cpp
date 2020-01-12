#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "header.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif


using namespace std;
using namespace cv;

#define ROUND_UINT(d) ( (unsigned int) ((d) + ((d) > 0 ? 0.5 : -0.5)) )

void motionMat(std::vector<Mat>& motionVec, size_t image_count, size_t rfactor, bool clockwise)
{

	size_t quotient, remainder;

	if(clockwise)
	{
			for( size_t i = 0; i < image_count; i ++)
			{
				Mat motionvec =  Mat::zeros(3,3,CV_32F);
				motionvec.at<float>(0,0) = 1;
				motionvec.at<float>(0,1) = 0;
				motionvec.at<float>(1,0) = 0;
				motionvec.at<float>(1,1) = 1;
				motionvec.at<float>(2,0) = 0;
				motionvec.at<float>(2,1) = 0;
				motionvec.at<float>(2,2) = 1;

				quotient = floor(i/1.0/rfactor);
				remainder = i%rfactor;

				if(quotient%2 == 0)
					motionvec.at<float>(0,2) = remainder/1.0/rfactor;

				else
					motionvec.at<float>(0,2) = (rfactor - remainder -1)/1.0/rfactor;

				motionvec.at<float>(1,2) = quotient/1.0/rfactor;

				motionVec.push_back(motionvec);

				std::cout<<"image i = "<<i<<", x motion = "<<motionvec.at<float>(0,2)<<", y motion = "<<motionvec.at<float>(1,2)<<std::endl;
			}
	}
	else
	{
			for( size_t i = 0; i < image_count; i ++)
			{
				Mat motionvec = Mat::zeros(3,3,CV_32F);
				motionvec.at<float>(0,0) = 1;
				motionvec.at<float>(0,1) = 0;
				motionvec.at<float>(1,0) = 0;
				motionvec.at<float>(1,1) = 1;
				motionvec.at<float>(2,0) = 0;
				motionvec.at<float>(2,1) = 0;
				motionvec.at<float>(2,2) = 1;

				quotient = floor(i/1.0/rfactor);
				remainder = i%rfactor;
				if(quotient%2 == 0)
					motionvec.at<float>(1,2) = remainder/1.0/rfactor;

				else
					motionvec.at<float>(1,2) = (rfactor - remainder -1)/1.0/rfactor;

				motionvec.at<float>(0,2) = quotient/1.0/rfactor;

				motionVec.push_back(motionvec);

			}
	}

}


Eigen::SparseMatrix<float, Eigen::RowMajor,int> Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor)
{
	int dim_srcvec = Src.rows * Src.cols;
    int dim_dstvec = Dest.rows * Dest.cols;

	Eigen::SparseMatrix<float,Eigen::RowMajor, int> _Dmatrix(dim_srcvec, dim_dstvec);
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j< Src.cols; j++)
		{
			int LRindex = i*Src.cols + j;
			for (int m = rfactor*i; m < (i+1)*rfactor; m++)
			{
				for (int n = rfactor*j; n < (j+1)*rfactor; n++)
				{
					int HRindex = m*Dest.cols + n;
					_Dmatrix.coeffRef(LRindex,HRindex) = 1.0/rfactor/rfactor;
					//std::cout<<"_Dmatrix.coeffRef(LRindex,HRindex) = "<<1.0/rfactor/rfactor<<", rfactor = "<<rfactor<<std::endl;
				}
			}
		}
	}

	return _Dmatrix;
}

Eigen::SparseMatrix<float, Eigen::RowMajor,int> Hmatrix(cv::Mat & Dest, const cv::Mat& kernel)
{

    int dim_dstvec = Dest.rows * Dest.cols;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Hmatrix(dim_dstvec, dim_dstvec);


	for (int i = 0; i < Dest.rows; i++)
	{
		for (int j = 0; j< Dest.cols; j++)
		{
			int index = i*Dest.cols + j;

			int UL = (i-1)*Dest.cols + (j-1);
			if (i-1 >= 0 && j-1 >= 0 && UL < dim_dstvec)
				_Hmatrix.coeffRef(index, UL) = kernel.at<float>(0,0);
			int UM = (i-1)*Dest.cols + j;
			if (i-1 >= 0 && UM < dim_dstvec)
				_Hmatrix.coeffRef(index, UM) = kernel.at<float>(0,1);
			int UR = (i-1)*Dest.cols + (j+1);
			if (i-1 >= 0 && j+1 < Dest.cols && UR < dim_dstvec)
				_Hmatrix.coeffRef(index, UR) = kernel.at<float>(0,2);
			int ML = i*Dest.cols + (j-1);
			if (j-1 >= 0 && ML < dim_dstvec)
				_Hmatrix.coeffRef(index, ML) = kernel.at<float>(1,0);
			int MR = i*Dest.cols + (j+1);
			if (j+1 < Dest.cols && MR < dim_dstvec)
				_Hmatrix.coeffRef(index, MR) = kernel.at<float>(1,2);
			int BL = (i+1)*Dest.cols + (j-1);
			if (j-1 >= 0 && i+1 < Dest.rows && BL < dim_dstvec)
				_Hmatrix.coeffRef(index, BL) = kernel.at<float>(2,0);
			int BM = (i+1)*Dest.cols + j;
			if (i+1 < Dest.rows && BM < dim_dstvec)
				_Hmatrix.coeffRef(index, BM) = kernel.at<float>(2,1);
			int BR = (i+1)*Dest.cols + (j+1);
			if (i+1 < Dest.rows && j+1 < Dest.cols && BR < dim_dstvec)
				_Hmatrix.coeffRef(index, BR) = kernel.at<float>(2,2);

			_Hmatrix.coeffRef(index,index) = kernel.at<float>(1,1);
		}
	}

	return _Hmatrix;
}

Eigen::SparseMatrix<float, Eigen::RowMajor,int> Mmatrix(cv::Mat &Dest, float deltaX, float deltaY)
{
	int dim_dstvec = Dest.rows * Dest.cols;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Mmatrix(dim_dstvec, dim_dstvec);

	for (int i = 0; i < Dest.rows; i++)
	{
		for(int j = 0; j < Dest.cols; j++)
		{
			if(i < (Dest.rows-std::floor(deltaY)) && j< (Dest.cols-std::floor(deltaX)) && (i+std::floor(deltaY) >= 0) && (j+std::floor(deltaX) >= 0))
			{
				int index = i*Dest.cols + j;
				int neighborUL = (i+std::floor(deltaY))*Dest.cols + (j+std::floor(deltaX));
				int neighborUR = (i+std::floor(deltaY))*Dest.cols + (j+std::floor(deltaX)+1);
				int neighborBR = (i+std::floor(deltaY)+1)*Dest.cols + (j+std::floor(deltaX)+1);
				int neighborBL = (i+std::floor(deltaY)+1)*Dest.cols + (j+std::floor(deltaX));

				if(neighborUL >= 0 && neighborUL < dim_dstvec)
					_Mmatrix.coeffRef(index, neighborUL) = (i+std::floor(deltaY)+1-(i+deltaY))*(j+std::floor(deltaX)+1-(j+deltaX));
				if(neighborUR >= 0 && neighborUR < dim_dstvec)
					_Mmatrix.coeffRef(index, neighborUR) = (i+std::floor(deltaY)+1-(i+deltaY))*(j+deltaX-(j+std::floor(deltaX)));
				if(neighborBR >= 0 && neighborBR < dim_dstvec)
					_Mmatrix.coeffRef(index, neighborBR) = (i+deltaY-(i+std::floor(deltaY)))*(j+deltaX-(j+std::floor(deltaX)));
				if(neighborBL >= 0 && neighborBL < dim_dstvec)
					_Mmatrix.coeffRef(index, neighborBL) = (i+deltaY-(i+std::floor(deltaY)))*(j+std::floor(deltaX)+1-(j+deltaX));
			}

		}
	}

	return _Mmatrix;
}


Eigen::SparseMatrix<float,Eigen::RowMajor, int> ComposeSystemMatrix(cv::Mat& Src, cv::Mat& Dest, const cv::Point2f delta, int rfactor, const cv::Mat& kernel, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& DMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &HMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &MMatrix)
{

	int dim_srcvec = Src.rows * Src.cols;
    int dim_dstvec = Dest.rows * Dest.cols;

    //float maxPsfRadius = 3 * rfactor * psfWidth;

    Eigen::SparseMatrix<float,Eigen::RowMajor, int> _DHF(dim_srcvec, dim_dstvec);

    DMatrix = Dmatrix(Src, Dest, rfactor);
    HMatrix = Hmatrix(Dest, kernel);
    MMatrix = Mmatrix(Dest, delta.x, delta.y);

    _DHF = DMatrix * (HMatrix * MMatrix);

    _DHF.makeCompressed();

    return _DHF;
}


void Normalization(Eigen::SparseMatrix<float, Eigen::RowMajor, int>& src, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& dst)
{
	for(Eigen::Index c = 0; c < src.rows(); ++c)
	{
		float colsum = 0.0;
		for(typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator itL(src, c); itL; ++itL)
			 colsum += itL.value();

		for(typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator itl(src, c); itl; ++itl)
			dst.coeffRef(itl.row(), itl.col()) = src.coeffRef(itl.row(), itl.col())/colsum;
	}
}

void Gaussiankernel(cv::Mat& dst)
{
	int klim = int((dst.rows-1)/2);

	for(int i = -klim; i <= klim; i++)
	{
		for (int j = -klim; j <= klim; j++)
		{
			float dist = i*i+j*j;
			dst.at<float>(i+klim, j+klim) = 1/(2*M_PI)*exp(-dist/2);
		}
	}

	float normF = cv::sum(dst)[0];
	dst = dst/normF;
}


Eigen::SparseMatrix<float, Eigen::RowMajor, int> sparseMatSq(Eigen::SparseMatrix<float, Eigen::RowMajor, int>& src)
{

	Eigen::SparseMatrix<float,Eigen::RowMajor, int> A2(src.rows(), src.cols());

	for (int k = 0; k < src.outerSize(); ++k)
	{
	   	for (typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator innerit(src,k); innerit; ++innerit)
	   	{
	   		//A2.insert(innerit.row(), innerit.col()) = innerit.value() * innerit.value();
	   		A2.insert(k, innerit.index()) = innerit.value() * innerit.value();
	   		//A2.insert(innerit.row(), innerit.col()) = 0;
	   	}
	}
	A2.makeCompressed();
	return A2;

}


void GenerateAT(cv::Mat& Src, cv::Mat& Dest, int imgindex, std::vector<Mat>& motionVec, cv::Mat &kernel, size_t rfactor, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& DMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &HMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &MMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& A, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& AT, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& A2, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& AT2, std::vector<viennacl::compressed_matrix<float> >& DHF, std::vector<viennacl::compressed_matrix<float> >& DHFT, std::vector<viennacl::compressed_matrix<float> > &DHF2, std::vector<viennacl::compressed_matrix<float> > &DHFT2)
{

	Gaussiankernel(kernel);

	cv::Point2f Shifts;
	Shifts.x = motionVec[imgindex].at<float>(0,2)*rfactor;
	Shifts.y = motionVec[imgindex].at<float>(1,2)*rfactor;

	A = ComposeSystemMatrix(Src, Dest, Shifts, rfactor, kernel, DMatrix, HMatrix, MMatrix);

	Normalization(A, A);

	A2 = sparseMatSq(A);

	AT = A.transpose();

	AT2 = A2.transpose();

	viennacl::compressed_matrix<float>tmp_vcl(A.rows(), A.cols(), A.nonZeros());
	viennacl::compressed_matrix<float>tmp_vclT(AT.rows(), AT.cols(), AT.nonZeros());

	viennacl::copy(A, tmp_vcl);
	viennacl::copy(AT, tmp_vclT);

	DHF.push_back(tmp_vcl);
	DHFT.push_back(tmp_vclT);

	viennacl::copy(A2, tmp_vcl);
	viennacl::copy(AT2, tmp_vclT);

	DHF2.push_back(tmp_vcl);
	DHFT2.push_back(tmp_vclT);


}


int main(int argc, char** argv)
{

    size_t image_count = 4;// M
    int rfactor = 2;//magnification factor
    float psfWidth = 3;


    std::vector<cv::Mat> Src(image_count);
    cv::Mat dest;
    cv::Mat kernel = cv::Mat::zeros(cv::Size(psfWidth, psfWidth), CV_32F);


    std::vector<viennacl::compressed_matrix<float> > DHF; 
    std::vector<viennacl::compressed_matrix<float> > DHFT; 
    std::vector<viennacl::compressed_matrix<float> > DHF2; 
    std::vector<viennacl::compressed_matrix<float> > DHFT2; 
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> DMatrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> HMatrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> MMatrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> AT;  // transpose of matrix A_i
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> A;  // matrix A_i
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> AT2; //transpose of matrix B_i
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> A2; // matrix B_i


    /***** Generate motion parameters ******/

    std::vector<cv::Mat> motionvec;
    motionMat(motionvec, image_count, rfactor, true);

    for (size_t i = 0;i < image_count;i++)
    {
        Src[i] = cv::imread("../Images/Cameraman/LR"+ boost::lexical_cast<std::string> (i+1) + ".tif", CV_LOAD_IMAGE_ANYDEPTH);
        
	if(! Src[i].data)
            std::cerr<<"No files can be found!"<<std::endl;

        Src[i].convertTo(Src[i], CV_32F);


	dest = cv::Mat(Src[0].rows * rfactor, Src[0].cols * rfactor, CV_16UC1);
	cv::resize(Src[0], dest, dest.size(), 0, 0, INTER_CUBIC);



    /***** Generate Matrices A = DHF, inverse A = DHFT and B = DHF2, invere B = DHFT2 ******/
	GenerateAT(Src[i], dest, i, motionvec, kernel, rfactor, DMatrix, HMatrix, MMatrix, A, AT, A2, AT2, DHF, DHFT, DHF2, DHFT2);

	std::cout<<"Matrices of image "<<(i+1)<<" done."<<std::endl;
    }

    std::cout<<"CPU calculation is done."<<std::endl;

    return 0;
}

