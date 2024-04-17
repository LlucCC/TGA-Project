#include <stdio.h>

__global__ void KernelFilter_BlackAndWhite(unsigned char *img, const int width, const int height, const int pixelSize)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * width + j;
  idx = idx * pixelSize;


  if (i < height && j < width) {
    char mean = (img[idx] + img[idx+1] + img[idx+2]) / 3;
    img[idx]= mean;
    img[idx+1]=mean;
    img[idx+2]=mean;
  }
}



int SequentialFilter_BlackAndWhite(unsigned char *img, const int& width, const int& height, const int& pixelSize)
{
	for(int i=0;i<width*height*3;i=i+3)
	{
		char mean = (img[i] + img[i+1] + img[i+2]) / 3;
		img[i]= mean;
		img[i+1]=mean;
		img[i+2]=mean;
	}
	return 0;
}
