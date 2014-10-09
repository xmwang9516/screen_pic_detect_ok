/************************************************************************/
/* 
    _resize img with maxImgSIze ;
     maxImgSize = 100 in the code;
*/
/************************************************************************/

#include "screen_picture_img_resize.h"






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
/* resize the image with the scale = 0.75;                              */
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
/* input original img, output resized img         */
/************************************************************************/
void img_resize(std::string img_ori_path, std::string img_dst_path, int nSize)
{
  const char* img_path_ori = NULL;
  const char* img_path_save = NULL;

  if(img_ori_path != "")
    img_path_ori = img_ori_path.c_str();
  else
    cout << "input img path is null!" << endl;

  if(img_dst_path != "")
    img_path_save = img_dst_path.c_str();
  else
    cout << "resize img path is null!" << endl;

  IplImage* img_dst = NULL;

  IplImage* img_ori = cvLoadImage(img_path_ori,CV_LOAD_IMAGE_COLOR);
  if (!img_ori)
  { 
    cout << img_path_ori << " load failed!"<< endl;
    return;
  }
  //img_dst =  resize_image(img_ori,nSize);
  img_dst =  resize_image_scale(img_ori,nSize);

  cvSaveImage(img_path_save,img_dst);

  cvReleaseImage(&img_dst);
}