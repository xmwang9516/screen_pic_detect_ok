
#include "pic_proc.h"


int main(int argc,char * argv[])
{

  int i = 0;
  int nIndex = 0;

  string optionInput = argv[1];
  cout << "<" << optionInput.c_str() << ">" << endl;

  nIndex = 2;

  
  if (optionInput == "resize")
  {
    // used for resize img call test;
    string input = "";
    string output = "";
    int    nSize = 0;

    for(i=nIndex;i<argc;i++)
    {
      if(argv[i][0] != '-') break;
          ++i;
      switch(argv[i-1][1])
      {
        case 'i':
          input = argv[i];
          break;
        case 's':
          nSize = atoi(argv[i]);
          break;
        case 'o':
          output = argv[i];
          break;
      }
    }

    img_resize(input, output,nSize);
  }
  
  if (optionInput == "train")
  {
     // used for model train test;
    string pos_path = "";
    string neg_path = "";

    for(i=nIndex;i<argc;i++)
    {
      if(argv[i][0] != '-') break;
          ++i;
      switch(argv[i-1][1])
      {
        case 'p':
            pos_path = argv[i];
            break;
        case 'n':
            neg_path = argv[i];
            break;
      }
    }

      train_model(pos_path,neg_path);
  }

  if (optionInput == "detect")
  {
    // usef for detect img;

     char    szImgName[100] = "";

    for(i=nIndex;i<argc;i++)
    {
      if(argv[i][0] != '-') break;
        ++i;
      switch(argv[i-1][1])
      {
          case 'i':
              sprintf(szImgName,"%s",argv[i]);
              break;
      }
    }


  cout << "input img is \"" << szImgName << "\"" << endl;


  clock_t start  = clock();

  int dRet = img_predict(szImgName);

  cout << "dRet = " << dRet << endl;
  if (dRet == 1)
    cout << "This image is not a screen picture." << endl;
  else if(dRet == -1)
    cout << "This image is a screen picture." << endl;
  else
  {
    cout << "Error. Maybe the input params are not right!" << endl;
    cout << "param format : " << "./screen_picture_detect_img -i ./testimgs/test1.jpg" << endl;
  }
    
 
  clock_t end = clock();
  double time_used = ((double)(end-start)) / CLOCKS_PER_SEC;
  cout << "Time used " << time_used << " seconds." << endl;
  }

   if(optionInput == "dup_detect")
   {
    string imgpath1 = "";
    string imgpath2 = "";

    imgpath1 = argv[nIndex];
    imgpath2 = argv[nIndex+1];

    clock_t start  = clock();

    int nRet = dup_img_detect(imgpath1, imgpath2);

    if(nRet == 1)
      cout << "dup result = " << "similar" << endl;
    else if(nRet == -1)
      cout << "dup result = " << "dissimilar" << endl;
    else
      cout << "detect result = " << "unknown" << endl;

    clock_t end = clock();
    double time_used = ((double)(end-start)) / CLOCKS_PER_SEC;
    cout << "Time used " << time_used << " seconds." << endl;
   }


  return 1;
}
