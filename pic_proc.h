#ifndef pic_proc_h__
#define pic_proc_h__



#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include "linear.h"


extern "C"{
   #include <vl/generic.h>
   #include <vl/lbp.h>
}


using namespace std;

#define  Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define  INF HUGE_VAL

#define  MODEL_FILE        "./train_feat.model"
 
#define  FEAT__SAVE_FILE   "./train_feat"

#define  CELL_SIZE         9

#define  MAX_IMG_SIZE      100


// resize img;
IplImage*       resize_image(IplImage* img,int max_size);
// resize img with sacel=0.75;
IplImage*       resize_image_scale(IplImage* img,int nResize);

void            img_resize(std::string img_ori_path, std::string img_dst_path, int nSize);
// predict function;
int             imgs_predict(char *testFile,char *resultFile);
// return the predict result of one img;
int             img_predict(char* imgName);
// extract feature ;
void            feat_extract(std::string img_name, std::vector<float> &vecfeat,int &cHeight, int &cWidth,int &featDim);
// extract feature ; 
void         	feat_extract(std::string img_pos_file,std::string img_neg_file,std::string feat_save_file);
// convert img to gray float;
IplImage*       convert_to_gray32( IplImage* img );
// do each img predict;
int             do_each_predict(char *feat_line);
// img predict;
void            do_predict(FILE *input, FILE *output);
// print null;
void            print_null(const char *s) {}
// show help;
void         	exit_with_help();
// read line of file;
static char* 	readline(FILE *input);
// parse params;
void         	parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
// show read problem;
void         	read_problem(const char *filename);
// corss validation;
void         	do_cross_validation();
// show input error;
void         	exit_input_error(int line_num);
// get imgs list of input path;
void         	get_file_list(std::string file_path, int depth, std::vector<string> &vecfilelist);
// train model with params;
int          	train_model(int nNum, char **pParams);
// train model main func;
void         	train_model(std::string img_pos_file,std::string img_neg_file);
// duplicate img detection function , input is img_name1 and img_name2;
int             dup_img_detect(std::string img_path1, std::string img_path2);
// extract feature all;
void            lbp_extract(string img_pos_file,string img_neg_file,string feat_save_file);












#endif