#include <stdio.h>

__device__ unsigned short int weights [5][5] = 
{
	{2, 4,  5,  4,  2},
	{4, 9,  12, 9,  4},
	{5, 12, 15, 12, 5},
	{4, 9,  12, 9,  4},
	{2, 4,  5,  4,  2}
};


unsigned short int hweights [5][5] = 
{
	{2, 4,  5,  4,  2},
	{4, 9,  12, 9,  4},
	{5, 12, 15, 12, 5},
	{4, 9,  12, 9,  4},
	{2, 4,  5,  4,  2}
};
__global__ void KernelFilter_Gaussiano(unsigned char *imgAux,unsigned char *img, const int width, const int height, const int pixelSize)
{
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = x * width + y;
  idx = idx * pixelSize;


  if (1 < x && x < height - 2 && 1 < y && y < width - 2) {
    unsigned short int red = 0, green = 0, blue = 0;
    for (int i = -2; i <= 2; ++i)
    {
    	for (int j = -2; j <= 2; ++j)
	{
		
		int idx = ((x + i) * width + (y + j)) * pixelSize;
		unsigned short int weight = weights[i + 2][j + 2];
		red += imgAux[idx] * weight;
		green += imgAux[idx + 1] * weight;
		blue += imgAux[idx + 2] * weight;

	}
    }
    int idx = (x * width + y) * pixelSize;
    img[idx] = red / 159;
    img[idx + 1] = green / 159;
    img[idx + 2] = blue / 159;
  }
}



int SequentialFilter_Gaussiano(unsigned char *img, const int& width, const int& height, const int& pixelSize)
{
	unsigned char *aux = (unsigned char*) malloc(width * height * pixelSize);
	memcpy(aux, img, width * height * pixelSize);
	for (int x = 2; x < height - 2; ++x)
	{
		for (int y = 2; y < width - 2; ++y)
		{
			unsigned int red = 0, green = 0, blue = 0;
			for (int i = -2; i <= 2; ++i)
			{
				for (int j = -2; j <= 2; ++j)
				{
					int idx = ((x + i) * width + (y + j)) * pixelSize;
					unsigned short int weight = hweights[i + 2][j + 2];
					red += aux[idx] * weight;
					green += aux[idx + 1] * weight;
					blue += aux[idx + 2] * weight;
				}
			}
			int idx = (x * width + y) * pixelSize;
			img[idx] = red / 159;
			img[idx + 1] = green /159;
			img[idx + 2] = blue / 159;
		}
	}
	return 0;
}
