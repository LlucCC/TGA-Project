#include <stdio.h>
#include <math.h>

__global__ void KernelFilter_Sobel(unsigned char *aux, unsigned char *img, const int width, const int height, const int pixelSize)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = y * width + x;

  idx = idx*pixelSize;

  if (y < height - 1 && x < width - 1 && y > 0 && x > 0) {
    int idx = (x + y*width)*pixelSize;

    int idx1 = ((x - 1) + y*width) * pixelSize;
    int idx2 = ((x - 1) + (y - 1)*width) * pixelSize;
    int idx3 = ((x - 1) + (y + 1)*width) * pixelSize;

    int idx4 = ((x + 1) + y*width) * pixelSize;
    int idx5 = ((x + 1) + (y + 1)*width) * pixelSize;
    int idx6 = ((x + 1) + (y - 1)*width) * pixelSize;

    int idx7 = (x + (y + 1)*width) * pixelSize;
    int idx8 = (x + (y - 1)*width) * pixelSize;
            
    short int redX = 0, greenX = 0, blueX = 0, redY = 0, greenY = 0, blueY = 0;
            
    redX = aux[idx5] + aux[idx6] + aux[idx4]*2 - aux[idx2] - aux[idx3] - 2*aux[idx1];
    greenX = aux[idx5 + 1] + aux[idx6 + 1] + aux[idx4 + 1]*2 - aux[idx2 + 1] - aux[idx3 + 1] - 2*aux[idx1 + 1];
    blueX = aux[idx5 + 2] + aux[idx6 + 2] + aux[idx4 + 2]*2 - aux[idx2 + 2] - aux[idx3 + 2] - 2*aux[idx1 + 2];

    redY = aux[idx5] + aux[idx6] + aux[idx7]*2 - aux[idx2] - aux[idx3] - 2*aux[idx8];
    greenY = aux[idx5 + 1] + aux[idx6 + 1] + aux[idx7 + 1]*2 - aux[idx2 + 1] - aux[idx3 + 1] - 2*aux[idx8 + 1];
    blueY = aux[idx5 + 2] + aux[idx6 + 2] + aux[idx7 + 2]*2 - aux[idx2 + 2] - aux[idx3 + 2] - 2*aux[idx8 + 2];

    float red,green, blue;
    red = sqrtf(redX*redX + redY*redY);
    green = sqrtf(greenX*greenX + greenY*greenY);
    blue = sqrtf(blueX*blueX + blueY*blueY);

    if (red > 255) red = 255;
    if (green > 255) green = 255;
    if (blue > 255) blue = 255;

    img[idx] = 255 - red;
    img[idx + 1] = 255 - green;
    img[idx + 2] = 255 - blue;
  }
}



int SequentialFilter_Sobel(unsigned char *img, const int& width, const int& height, const int& pixelSize)
{
    unsigned char *aux = (unsigned char*) malloc(width * height * pixelSize);
    memcpy(aux, img, width * height * pixelSize);
    for (int x = 1; x < width - 1; ++x) {
        for (int y = 1; y < height - 1; ++y)
        {
          int idx = (x + y*width)*pixelSize;

          int idx1 = ((x - 1) + y*width) * pixelSize;
          int idx2 = ((x - 1) + (y - 1)*width) * pixelSize;
          int idx3 = ((x - 1) + (y + 1)*width) * pixelSize;

          int idx4 = ((x + 1) + y*width) * pixelSize;
          int idx5 = ((x + 1) + (y + 1)*width) * pixelSize;
          int idx6 = ((x + 1) + (y - 1)*width) * pixelSize;

          int idx7 = (x + (y + 1)*width) * pixelSize;
          int idx8 = (x + (y - 1)*width) * pixelSize;
            
          short int redX = 0, greenX = 0, blueX = 0, redY = 0, greenY = 0, blueY = 0;
            
          redX = aux[idx5] + aux[idx6] + aux[idx4]*2 - aux[idx2] - aux[idx3] - 2*aux[idx1];
          greenX = aux[idx5 + 1] + aux[idx6 + 1] + aux[idx4 + 1]*2 - aux[idx2 + 1] - aux[idx3 + 1] - 2*aux[idx1 + 1];
          blueX = aux[idx5 + 2] + aux[idx6 + 2] + aux[idx4 + 2]*2 - aux[idx2 + 2] - aux[idx3 + 2] - 2*aux[idx1 + 2];

          redY = aux[idx5] + aux[idx6] + aux[idx7]*2 - aux[idx2] - aux[idx3] - 2*aux[idx8];
          greenY = aux[idx5 + 1] + aux[idx6 + 1] + aux[idx7 + 1]*2 - aux[idx2 + 1] - aux[idx3 + 1] - 2*aux[idx8 + 1];
          blueY = aux[idx5 + 2] + aux[idx6 + 2] + aux[idx7 + 2]*2 - aux[idx2 + 2] - aux[idx3 + 2] - 2*aux[idx8 + 2];

          float red,green, blue;
          red = sqrt(redX*redX + redY*redY);
          green = sqrt(greenX*greenX + greenY*greenY);
          blue = sqrt(blueX*blueX + blueY*blueY);

          if (red > 255) red = 255;
          if (green > 255) green = 255;
          if (blue > 255) blue = 255;

          img[idx] = 255 - red;
          img[idx + 1] = 255 - green;
          img[idx + 2] = 255 - blue;
        }
    }
    return 0;
}
