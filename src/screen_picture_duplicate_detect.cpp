/************************************************************************/
/* duplicate imgs detection  */
/************************************************************************/



#include "screen_picture_duplicate_detect.h"












/************************************************************************/
/* resize the image, will release old image if success resized          */
/************************************************************************/
IplImage* resize_image(IplImage* img,int max_size){
	int org_width = img->width;
	int org_height = img->height;
	double scale = org_width*1.0 / org_height;
	
	if(org_width> max_size||org_height>max_size){
		CvSize sz;
		if( scale > 1) sz = cvSize(max_size,max_size/scale);
		else sz = cvSize(scale*max_size,max_size);
		IplImage* image_o = cvCreateImage(sz,img->depth,img->nChannels);
		cvResize(img,image_o,CV_INTER_AREA);
		cvReleaseImage(&img);
		return image_o;
	}
	return img;
}

/************************************************************************/
/* resize the image with the scale = 0.75, and max_size;                */
/************************************************************************/
IplImage* resize_image_scale(IplImage* img,int nResize)
{
  int org_width = img->width;
  int org_height = img->height;
  double scale = 0.75;
  
  CvSize sz;
  if(org_height > org_width) 
    sz = cvSize(nResize*scale,nResize);
  else 
    sz = cvSize(nResize,nResize*scale);
  IplImage* image_o = cvCreateImage(sz,img->depth,img->nChannels);
  cvResize(img,image_o,CV_INTER_AREA);
  cvReleaseImage(&img);

  return image_o;
}

/************************************************************************/
/*                       extract feature function                       */ 
/************************************************************************/
void feat_extract(std::string img_name, std::vector<float> &vecfeat,int &cHeight, int &cWidth,int &featDim)
{
    // load img;
    const char* imgPath = img_name.c_str();
    IplImage* img = cvLoadImage(imgPath,CV_LOAD_IMAGE_COLOR);
    if (!img)
    { 
      cout << imgPath << " load failed!"<< endl;
      return;
    }

    IplImage* grayImg = convert_to_gray32(img);//convert to gray_32 img, float* data;
    int    imgWidth = grayImg->width;
    int    imgHeight = grayImg->height;

    // resize gray image if need;
    if(imgWidth > maxImSize || imgHeight >maxImSize)
    {
       //grayImg = resize_image(grayImg,maxImSize);
      grayImg = resize_image_scale(grayImg,maxImSize);
       imgWidth = grayImg->width;
       imgHeight = grayImg->height;
    }

    float* im= (float*)grayImg->imageData ; 

    // extract lbp feat;
    VlLbp * lbp = vl_lbp_new(VlLbpUniform,false);
    vl_size lbpdm = vl_lbp_get_dimension(lbp);
    vl_size cellSize = CELL_SIZE; // 3*3;  

    int nSize = floor(imgWidth/cellSize) * floor(imgHeight/cellSize) * lbpdm;

    float *lbpfeat = (float*)malloc(nSize * sizeof(float));

    vl_lbp_process(lbp, lbpfeat, im , imgWidth, imgHeight, cellSize) ;

    cWidth = imgWidth / cellSize;
    cHeight = imgHeight / cellSize ;
    featDim = lbpdm;
    
    for (int i = 0; i < cHeight*cWidth*featDim; i++)
    {
      vecfeat.push_back(lbpfeat[i]);
    }

  // delete the object lbp;
    vl_lbp_delete(lbp);
    free(lbpfeat);

    cvReleaseImage(&img);
    cvReleaseImage(&grayImg);
      
}


int  dup_img_detect(std::string img_path1, std::string img_path2)
{
     int       nRet = 0;

     int       cHeight = 0; 
     int       cWidth = 0;
     int       featDim = 0;
     float     fvisualSimi = 0.0;

     float     fthreshold = 10.0;

     std::vector<float> vecfeat1;
     std::vector<float> vecfeat2;

     vecfeat1.clear();
     vecfeat2.clear();

     feat_extract(img_path1.c_str(),vecfeat1,cHeight,cWidth,featDim);
     feat_extract(img_path2.c_str(),vecfeat2,cHeight,cWidth,featDim);

     // cal the visual similarity;
     for (int i = 0; i < cHeight*cWidth*featDim; i++)
     {
       fvisualSimi += sqrtf((vecfeat1[i]-vecfeat2[i])*(vecfeat1[i]-vecfeat2[i]));
     }
     //cout << fvisualSimi << endl;

     if(fthreshold <= fvisualSimi)
        nRet = -1;
     else
        nRet = 1;


     return nRet;
}



/*****************************************************************/
/*      Converts an image to 32-bit grayscale                    */

/*  @param img a 3-channel 8-bit color (BGR) or 8-bit gray image */

/*  @return Returns a 32-bit grayscale image                     */
/*****************************************************************/
IplImage* convert_to_gray32(IplImage* img)
{
  IplImage* gray8, * gray32;

  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
  gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
  cvCvtColor( img, gray8, CV_BGR2GRAY );
  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

  cvReleaseImage( &gray8 );
  return gray32;
}
