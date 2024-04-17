#include <stdio.h>

__global__ void KernelFilter_Mean(unsigned char *imgAux, unsigned char *img, const int width, const int height, const int pixelSize)
{
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;

 if (0 < x && x < height - 1 && 0 < y && y < width - 1)
 {
 	unsigned short int red = 0, green = 0, blue = 0;
	 for (int i = -1; i <= 1; ++i)
	{
		for (int j = -1; j <=1; ++j)
		{
			int idx = ((x + i) * width + (y + j)) * pixelSize;
			red += imgAux[idx];
			green += imgAux[idx + 1];
			blue += imgAux[idx + 2];
			
		}
	}

	int idx = (x * width + y) * pixelSize;
	img[idx] = red / 9;
	img[idx + 1] = green / 9;
	img[idx + 2] = blue / 9;
  }
}



int SequentialFilter_Mean(unsigned char *img, const int& width, const int& height, const int& pixelSize)
{
	unsigned char *aux = (unsigned char*) malloc(width * height * pixelSize);
	memcpy(aux, img, width * height * pixelSize);
	for (int x = 1; x < height - 1; ++x)
	{
		for (int y = 1; y < width - 1; ++y)
		{
			unsigned short int red = 0, green = 0, blue = 0;
			for (int i = -1; i <= 1; ++i)
			{
				for (int j = -1; j <= 1; ++j)
				{
					int idx = ((x + i) * width + (y + j)) * pixelSize;
					red += aux[idx];
					green += aux[idx + 1];
					blue += aux[idx + 2];
				}
			}
			int idx = (x * width + y) * pixelSize;
			img[idx] = red / 9;
			img[idx + 1] = green / 9;
			img[idx + 2] = blue / 9;
		}
	}
	return 0;
}
