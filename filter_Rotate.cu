#include <stdio.h>

__global__ void KernelFilter_Rotate(unsigned char *img, const int width, const int height, const int pixelSize)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * width + j;
  
  int i_new = height - i - 1;
  int j_new = width - j - 1;

  int idx_new = (i_new*width + j_new) * pixelSize;
  idx = idx*pixelSize;

  if (i < height/2 && j < width) {

    int auxR = img[idx_new];
    int auxG = img[idx_new + 1];
    int auxB = img[idx_new+ 2];

    img[idx_new] = img[idx];
    img[idx_new + 1] = img[idx + 1];
    img[idx_new + 2] = img[idx + 2];

    img[idx] = auxR;
    img[idx + 1] = auxG;
    img[idx + 2] = auxB;
  }
}



int SequentialFilter_Rotate(unsigned char *img, const int& width, const int& height, const int& pixelSize)
{
    for(int i = 0; i < height/2; ++i)
	{
        int i_new = height - i - 1;
        for (int j = 0; j < width; ++j) {
            int j_new = width - j - 1;
            int idx = (i*width + j) * pixelSize;
            int idx_new = (i_new*width + j_new) * pixelSize;

            
            int auxR = img[idx_new];
            int auxG = img[idx_new + 1];
            int auxB = img[idx_new+ 2];

            img[idx_new] = img[idx];
            img[idx_new + 1] = img[idx + 1];
            img[idx_new + 2] = img[idx + 2];

            img[idx] = auxR;
            img[idx + 1] = auxG;
            img[idx + 2] = auxB;
        } 
	}
	return 0;
}
