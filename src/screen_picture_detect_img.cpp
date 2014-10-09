/************************************************************************/
/*           detect each img whether is a screen picture or not         */
/************************************************************************/

#include "screen_picture_detect_img.h"




/************************************************************************/

int     print_null(const char *s,...) {return 0;}
static  int (*info)(const char *fmt,...) = &printf;
struct  feature_node *x;
int     max_nr_attr = 64;
struct  model* model_;
int     flag_predict_probability = 0;

int     maxImSize = MAX_IMG_SIZE;  // image resize size(height=maxImSize or width=maxImSize);

/************************************************************************/








/************************************************************************/
/* predict codes */
/************************************************************************/



void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void do_predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int nr_class=get_nr_class(model_);
	double *prob_estimates=NULL;
	int j, n;
	int nr_feature=get_nr_feature(model_);
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	if(flag_predict_probability)
	{
		int *labels;

		if(!check_probability_model(model_))
		{
			fprintf(stderr, "probability output is only supported for logistic regression\n");
			exit(1);
		}

		labels=(int *) malloc(nr_class*sizeof(int));
		get_labels(model_,labels);
		prob_estimates = (double *) malloc(nr_class*sizeof(double));
		fprintf(output,"labels");
		for(j=0;j<nr_class;j++)
			fprintf(output," %d",labels[j]);
		fprintf(output,"\n");
		free(labels);
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);
			// feature indices larger than those in training are not used
			if(x[i].index <= nr_feature)
				++i;
		}
		if(model_->bias>=0)
		{
			x[i].index = n;
			x[i].value = model_->bias;
			i++;
		}
		x[i].index = -1;

		if(flag_predict_probability)
		{
			int j;
			predict_label = predict_probability(model_,x,prob_estimates); // return the predict label;
			fprintf(output,"%g",predict_label);
			for(j=0;j<model_->nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = predict(model_,x);
			fprintf(output,"%g\n",predict_label);
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if(model_->param.solver_type==L2R_L2LOSS_SVR ||
	   model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
	   model_->param.solver_type==L2R_L2LOSS_SVR_DUAL)
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
	if(flag_predict_probability)
		free(prob_estimates);
}

void exit_with_help()
{
	printf(
	"Usage: predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}



int imgs_predict(char *testFile,char *resultFile)
{
	FILE *input, *output;

    flag_predict_probability = 0;  // set 0 as default;

	input = fopen(testFile,"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",testFile);
		exit(1);
	}

	output = fopen(resultFile,"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",resultFile);
		exit(1);
	}

	if((model_=load_model(MODEL_FILE))==0)
	{
		fprintf(stderr,"can't open model file %s\n",MODEL_FILE);
		exit(1);
	}

	x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
	do_predict(input, output);
	free_and_destroy_model(&model_);
	free(line);
	free(x);
	fclose(input);
	fclose(output);

	return 0;
}



/************************************************************************/
/* resize the image, will release old image if success resized          */
/************************************************************************/
IplImage* resize_image(IplImage* img,int max_size)
{
	int org_width = img->width;
	int org_height = img->height;
	double scale = org_width*1.0 / org_height;
	
	if(org_width> max_size||org_height>max_size)
	{
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
/*                         predict one img                              */
/************************************************************************/
int img_predict(char* imgName)
{
	int  nlabel = 0;

    flag_predict_probability = 0;  // set 0 as default;  

    // load model;
	if((model_=load_model(MODEL_FILE))==0)
	{
		fprintf(stderr,"can't open model file %s\n",MODEL_FILE);
		exit(1);
	}
	//extract the img feature;
    int       cHeight = 0; 
    int       cWidth = 0;
    int       featDim = 0;

    vector<float>   vecfeat;
    feat_extract(imgName, vecfeat, cHeight, cWidth,featDim);
    if(vecfeat.size() == 0)
    {
    	return nlabel;
    }


    int nSize = cWidth * cHeight * featDim;
 
    char      *eachfeat  = (char*)malloc(nSize * nSize); 
    memset(eachfeat,0,sizeof(eachfeat));

    sprintf(eachfeat,"%s","+1 "); 
    for (int i = 0; i < cHeight*cWidth*featDim; i++)
    {
    	if(vecfeat[i] != 0)
       		sprintf(eachfeat+strlen(eachfeat),"%d:%f ",(i+1),vecfeat[i]);
    }

	x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
	
	nlabel = do_each_predict(eachfeat);

	free_and_destroy_model(&model_);
	free(x);
	free(eachfeat);
	free(line);

	return nlabel;
}


int do_each_predict(char *eachImgFeat)
{
	int nRetLabel = 0;  

	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int nr_class=get_nr_class(model_);
	double *prob_estimates=NULL;
	int j, n;
	int nr_feature=get_nr_feature(model_);
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	if(eachImgFeat != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok(eachImgFeat," \t\n"); 
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			// feature indices larger than those in training are not used
			if(x[i].index <= nr_feature)
				++i;
		}

		if(model_->bias>=0)
		{
			x[i].index = n;
			x[i].value = model_->bias;
			i++;
		}
		x[i].index = -1;

		if(flag_predict_probability)
		{
			int j;
			predict_label = predict_probability(model_,x,prob_estimates);
		}
		else
		{
			predict_label = predict(model_,x);
			nRetLabel = predict_label;
		}
	}

    return nRetLabel;
}

void feat_extract(string img_name, vector<float> &vecfeat,int &cHeight, int &cWidth,int &featDim)
{
    // load img;
    const char* imgPath = img_name.c_str();
    IplImage* img = cvLoadImage(imgPath,CV_LOAD_IMAGE_COLOR);
    if (!img)
    { 
      cout << imgPath << " load failed!"<< endl;
      return ;
    }

    IplImage* grayImg = convert_to_gray32(img);
    int    imgWidth = grayImg->width;
    int    imgHeight = grayImg->height;

    // resize gray image if need;
    if(imgWidth > maxImSize || imgHeight > maxImSize)
    {
       grayImg = resize_image_scale(grayImg,maxImSize);
       imgWidth = grayImg->width;
       imgHeight = grayImg->height;
    }
    float* im= (float*)grayImg->imageData ;

    // extract lbp feat;
    VlLbp * lbp = vl_lbp_new(VlLbpUniform,false);
    vl_size lbpdm = vl_lbp_get_dimension(lbp);
    vl_size cellSize = CELL_SIZE; // 3*3;  

    int   nSize = floor(imgWidth/cellSize) * floor(imgHeight/cellSize) * lbpdm;
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

/*****************************************************************/
/*      Converts an image to 32-bit grayscale                    */
/*  @param img a 3-channel 8-bit color (BGR) or 8-bit gray image */
/*  @return Returns a 32-bit grayscale image                     */
/*****************************************************************/
IplImage* convert_to_gray32( IplImage* img )
{
  IplImage* gray8  = NULL;
  IplImage* gray32 = NULL;

  gray32  =  cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
  gray8   =  cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
  
  cvCvtColor( img, gray8, CV_BGR2GRAY );
  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

  cvReleaseImage( &gray8 );
  return gray32;
}

