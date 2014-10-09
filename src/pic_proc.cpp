
#include "pic_proc.h"


int     print_null(const char *s,...) {return 0;}
static  int (*info)(const char *fmt,...) = &printf;
struct  feature_node *x;
int     max_nr_attr = 64;
struct  model* model_;
int     flag_predict_probability = 0;

struct  feature_node *x_space;
struct  parameter param;
struct  problem prob;
int     flag_cross_validation;
int     nr_fold;
double  bias;
static  char *line = NULL;
static  int max_line_len;

int     maxImSize = MAX_IMG_SIZE;  // image resize size(height=maxImSize or width=maxImSize);


/************************************************************************/
/*                input original img, output resized img                */
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
	//free(line);

	return nlabel;
}

void  train_model(std::string img_pos_file,std::string img_neg_file)
{
	int nRet = 0;
    std::string feat_save_file = FEAT__SAVE_FILE;
    // extract feature of pos and neg imgs and save into feature file;
    feat_extract(img_pos_file,img_neg_file,FEAT__SAVE_FILE);

    char **params = NULL;
    params = new char *[6];
    for(int i=0;i<6;i++)
    {
      params[i] = new char[100];
    }

    memset(params[0],0,sizeof(params[0]));
    sprintf(params[0],"%s","train");

    memset(params[1],0,sizeof(params[1]));
    sprintf(params[1],"%s","-B");

    memset(params[2],0,sizeof(params[2]));
    sprintf(params[2],"%s","1");

    memset(params[3],0,sizeof(params[3]));
    sprintf(params[3],"%s","-s");

    memset(params[4],0,sizeof(params[4]));
    sprintf(params[4],"%s","2");

    memset(params[5],0,sizeof(params[5]));
    sprintf(params[5],"%s",feat_save_file.c_str());

    nRet = train_model(6, params);   

    delete[] params;

    if(nRet == 1)
    	cout << "Success to train model !" << endl;
    else
    	cout << "Failed to train model !" << endl;

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

int train_model(int nNum, char **pParams)
{
	char   input_file_name[1024];
	char   model_file_name[1024];
	const  char *error_msg;

	parse_command_line(nNum, pParams, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = check_parameter(&prob,&param);

	//cout << "input_file_name = " << input_file_name << endl;
	//cout << "model_file_name = " << model_file_name << endl;

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	if(flag_cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		model_=train(&prob, &param);
		if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}
		free_and_destroy_model(&model_);
	}
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 1;
}



/************************************************************************/
/*                extract  feature function                             */
/************************************************************************/
void  feat_extract(std::string img_pos_file,std::string img_neg_file,std::string feat_save_file)
{
   ofstream feat_file(feat_save_file.c_str()); 

   for(int iFile=1;iFile<3;iFile++)
   {
     std::vector<string>   img_path;
     img_path.clear();
     string           img_file = "";
     string           strLabel = "";
     if(iFile == 1)
     {
          img_file = img_pos_file;
          strLabel = "+1";
     }
     if(iFile == 2)
     {
          img_file = img_neg_file;
          strLabel = "-1";
     }
    cout << img_file.c_str() << " is processing..." << endl;
    // get file list;
    get_file_list(img_file.c_str(),1,img_path);
   
    /*string strline = "";
    ifstream img_name(img_file.c_str()); 
    while(getline(img_name,strline))
    {
      if(strline != "")
      {
         string strpath = "./imgs/";
 	     strpath = strpath + strline;
         img_path.push_back(strpath);
      }
    }   
    img_name.close();*/

   // extract feat and save;
   for(int i=0;i<img_path.size();i++)
   {
	// load img;
   	//const char* imgPath = "./imgs/0a0ee8ec-0f54-464a-addd-715b3eae18fc.jpg";
    const char* imgPath = img_path[i].c_str();

   	IplImage* img = cvLoadImage(imgPath,CV_LOAD_IMAGE_COLOR);
   	if (!img)
   	{ 
        cout << imgPath << " load failed!"<< endl;
        //return ;
    }

    IplImage* grayImg = convert_to_gray32(img);//convert to gray_32 img, float* data;
	int    imgWidth = grayImg->width;
	int    imgHeight = grayImg->height;

	// resize gray image if need;
	if(imgWidth > maxImSize || imgHeight >maxImSize)
	{
       grayImg = resize_image(grayImg,maxImSize);
       imgWidth = grayImg->width;
       imgHeight = grayImg->height;
    }

   	float* im= (float*)grayImg->imageData ;	

	// extract lbp feat;
	// define lbp object;
   	VlLbp * lbp = vl_lbp_new(VlLbpUniform,false);
    vl_size lbpdm = vl_lbp_get_dimension(lbp);
   	vl_size cellSize = CELL_SIZE; // 3*3;  

   	int nSize = floor(imgWidth/cellSize) * floor(imgHeight/cellSize) * lbpdm;

   	float *lbpfeat = (float*)malloc(nSize * sizeof(float));
   	memset(lbpfeat,0,sizeof(lbpfeat));

   	vl_lbp_process(lbp, lbpfeat, im , imgWidth, imgHeight, cellSize) ;

   	int  cWidth = imgWidth / cellSize;
   	int  cHeight = imgHeight / cellSize ;
    int  featDim = lbpdm;
 	// save feat into file;
    feat_file << strLabel.c_str() << " ";
    for (int ii = 0; ii < cHeight*cWidth*featDim; ii++)
    {
      feat_file << (ii + 1) << ":"<< lbpfeat[ii] << " ";
    }
    feat_file << "\n";
 
   
	// delete the object lbp;
   	vl_lbp_delete(lbp);
   	free(lbpfeat);

   	cvReleaseImage(&img);
   	cvReleaseImage(&grayImg);
  }
}
  feat_file.close();
}


void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			  );
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}

	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	flag_cross_validation = 0;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}

void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"  for regression\n"
	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n"
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
	"		where f is the primal function and pos/neg are # of\n"
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
	"	-s 1, 3, 4, and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

IplImage* convert_to_gray32( IplImage* img )
{
  IplImage* gray8  = NULL;
  IplImage* gray32 = NULL;
  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
  gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
  cvCvtColor( img, gray8, CV_BGR2GRAY );
  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

  cvReleaseImage( &gray8 );
  return gray32;
}



void get_file_list(std::string file_path, int depth, std::vector<string> &vecfilelist)
{
    DIR *dir; //声明一个句柄;
    struct dirent *file; 
    struct stat sb;    
    
    if(!(dir = opendir(file_path.c_str())))
    {
        cout << "error open path " << file_path << endl;
        return ;
    }

    while((file = readdir(dir)) != NULL)
    {
        if(strncmp(file->d_name, ".", 1) == 0)
            continue;
        string strpath = file_path;
        strpath.append(file->d_name);
        vecfilelist.push_back(strpath);
        if(stat(file->d_name, &sb) >= 0 && S_ISDIR(sb.st_mode) && depth <= 3)
        {
            get_file_list(file->d_name, depth + 1,vecfilelist);
        }
    }
    closedir(dir);
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
	// if(model_->param.solver_type==L2R_L2LOSS_SVR ||
	//    model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
	//    model_->param.solver_type==L2R_L2LOSS_SVR_DUAL)
	// {
	// 	info("Mean squared error = %g (regression)\n",error/total);
	// 	info("Squared correlation coefficient = %g (regression)\n",
	// 		((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
	// 		((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
	// 		);
	// }
	// else
	// 	info("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
	if(flag_predict_probability)
		free(prob_estimates);
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

void feat_extract(std::string img_name, std::vector<float> &vecfeat,int &cHeight, int &cWidth,int &featDim)
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


void lbp_extract(string img_pos_file,string img_neg_file,string feat_save_file)
{

   
  ofstream feat_file(feat_save_file.c_str()); 

   for(int iFile=1;iFile<3;iFile++)
   {
     vector<string>   img_path;
     img_path.clear();
     string           img_file = "";
     string           strLabel = "";
     if(iFile == 1){
          img_file = img_pos_file;
          strLabel = "+1";
     }
     if(iFile == 2){
          img_file = img_neg_file;
          strLabel = "-1";
     }
    cout << "iFile = " << iFile << endl;
    cout << img_file.c_str() << " is processing..." << endl;
   
    string strline = "";
    
    ifstream img_name(img_file.c_str()); 
    while(getline(img_name,strline))
    {
      if(strline != "")
      {
         string strpath = "./imgs/";
 	      strpath = strpath + strline;
	 
         img_path.push_back(strpath);
      }
        
    }   
    img_name.close();
    cout << img_path.size() << endl;
 
   // ext feat and save;
  
   
   for(int i=0;i<img_path.size();i++)
   {
	// define lbp object;
   	VlLbp * lbp = vl_lbp_new(VlLbpUniform,false);
 
	// load img;
   	//const char* imgPath = "./imgs/0a0ee8ec-0f54-464a-addd-715b3eae18fc.jpg";
    const char* imgPath = img_path[i].c_str();
    cout << "imgpath = [" << imgPath << "]" << endl;


   	IplImage* img = cvLoadImage(imgPath,CV_LOAD_IMAGE_COLOR);
   	if (!img){ 
             cout << imgPath << " load failed!"<< endl;
             //return ;
        }

      IplImage* grayImg = convert_to_gray32(img);//convert to gray_32 img, float* data;
	  int    imgWidth = grayImg->width;
	  int    imgHeight = grayImg->height;

	  // resize gray image if need;
	  if(imgWidth > maxImSize || imgHeight >maxImSize)
	  {
     	   grayImg = resize_image(grayImg,maxImSize);
    	   imgWidth = grayImg->width;
    	   imgHeight = grayImg->height;
    }

   	float* im= (float*)grayImg->imageData ;	

	// extract lbp feat;
 	vl_size lbpdm = vl_lbp_get_dimension(lbp);
   	vl_size cellSize = CELL_SIZE; // 3*3;  

   	int nSize = floor(imgWidth/cellSize) * floor(imgHeight/cellSize) * lbpdm;

   	float *lbpfeat = (float*)malloc(nSize * sizeof(float));

   	vl_lbp_process(lbp, lbpfeat, im , imgWidth, imgHeight, cellSize) ;

   	int  cWidth = imgWidth / cellSize;
   	int  cHeight = imgHeight / cellSize ;
    int  featDim = lbpdm;
 	  // save feat into file;
    //ofstream feat_file(feat_save_file.c_str(),ios::app);  // for save the neg feature;

    feat_file << strLabel.c_str() << " ";
    for (int ii = 0; ii < cHeight*cWidth*featDim; ii++)
    {
      feat_file << (ii + 1) << ":"<< lbpfeat[ii] << " ";
    }
    feat_file << "\n";
 
   
	// delete the object lbp;
   	vl_lbp_delete(lbp);
   	free(lbpfeat);

   	cvReleaseImage(&img);
   	cvReleaseImage(&grayImg);
  }
}
feat_file.close();

   
}