#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <sys/times.h>
#include <sys/resource.h>
#include <bits/stdc++.h>

#include "filter_BlackAndWhite.cu"
#include "filter_Rotate.cu"
#include "filter_Sobel.cu"
#include "filter_Mean.cu"
#include "filter_Gaussiano.cu"

#define PINNED 1
#define ITER 1

void CheckCudaError(char sms[], int line);
float GetTime(void); 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char** argv)
{
//////////////////////////////////////////////////////////////
///////////////////DATOS//////////////////////////////////////
//////////////////////////////////////////////////////////////

  char *fileIN;	//Input file 
  char *fileOutSeq, *fileOutKernel; 	//Output file
  char *filter;

  unsigned char *img; //original image
  unsigned char *hImg;
  unsigned char *dImg;
  unsigned char *dAux;
  unsigned char *sImg;
  
  int width, height, pixelWidth; //meta info de la imagen
  //Time data
  float t0, t1;
  cudaEvent_t E1, E2, E0, E3;
  cudaEventCreate(&E1); cudaEventCreate(&E2); cudaEventCreate(&E0); cudaEventCreate(&E3);
  float TiempoGlobal[ITER], TiempoKernel[ITER], TiempoSequencial[ITER];
  float mediaGlobal, mediaKernel, mediaSequencial;

  //Sizes
  unsigned int numBytes;
  unsigned int nBlocksX, nThreadsX, nBlocksY, nThreadsY;

  //Cuda stuff
  int gpu, count;
  
//////////////////////////////////////////////////////////////
///////////////////INICIALIZACION/////////////////////////////
//////////////////////////////////////////////////////////////
  if (argc == 5) { fileIN = argv[1]; fileOutSeq = argv[2]; fileOutKernel = argv[3]; filter = argv[4];}
  else { printf("Usage: ./exe fileIN fileOUTSEQ fileOUTKERNEL\n"); exit(0); }

  printf("Reading image...\n");
  img = stbi_load(fileIN, &width, &height, &pixelWidth, 0);
  if (!img) {
    fprintf(stderr, "Couldn't load image.\n");
     return (-1);
  }
  printf("width: %i, height: %i\n", width, height);
  
  //Set device
  cudaGetDeviceCount(&count);
  srand(time(NULL));
  gpu = rand();
  cudaSetDevice((gpu>>3) % count);
  
  numBytes = width * height * pixelWidth * sizeof(unsigned char);
  
  nThreadsX = 32;
  nThreadsY = 32;
  
  nBlocksX = (width + nThreadsX - 1)/nThreadsX;
  nBlocksY = (height + nThreadsY - 1)/nThreadsY;
  
  dim3 dimGrid(nBlocksX, nBlocksY, 1);
  dim3 dimBlock(nThreadsX, nThreadsY, 1);
//////////////////////////////////////////////////////////////
///////////////////RESERVA DE MEMORIA/////////////////////////
//////////////////////////////////////////////////////////////
  if (PINNED) cudaMallocHost((unsigned char**)&hImg, numBytes); 
  else hImg = (unsigned char*) malloc(numBytes); 
  
  cudaMalloc((unsigned char**)&dImg, numBytes);
  
  
  sImg = (unsigned char*) malloc(numBytes);
  
  for(int i = 0; i < width*height*pixelWidth; ++i) {
  	sImg[i] = img[i];
	  hImg[i] = img[i];
  }
  
//////////////////////////////////////////////////////////////
///////////////////PROCESADO//////////////////////////////////
//////////////////////////////////////////////////////////////
  for (int i = 0; i < 1; ++i) {
    printf("Filtrando\n");

    cudaEventRecord(E0,0);
    cudaEventSynchronize(E0);
    cudaMemcpy(dImg,hImg,numBytes,cudaMemcpyHostToDevice);

    switch (filter[0])
    {
    case 'B':
      cudaEventRecord(E1,0);
      cudaEventSynchronize(E1);
      KernelFilter_BlackAndWhite<<<dimGrid,dimBlock>>>(dImg, width, height, pixelWidth);
      //gpuErrchk( cudaPeekAtLastError() );
      //gpuErrchk( cudaDeviceSynchronize() );
      cudaEventRecord(E2,0);
      cudaEventSynchronize(E2);

      cudaMemcpy(hImg,dImg,numBytes,cudaMemcpyDeviceToHost);
      cudaEventRecord(E3,0);
      cudaEventSynchronize(E3);

      t0 = GetTime();
      SequentialFilter_BlackAndWhite(sImg, width, height, pixelWidth);
      t1 = GetTime(); 
      break;
    case 'R':
      cudaEventRecord(E1,0);
      cudaEventSynchronize(E1);
      KernelFilter_Rotate<<<dimGrid,dimBlock>>>(dImg, width, height, pixelWidth);
      //gpuErrchk( cudaPeekAtLastError() );
      //gpuErrchk( cudaDeviceSynchronize() );
      cudaEventRecord(E2,0);
      cudaEventSynchronize(E2);

      cudaMemcpy(hImg,dImg,numBytes,cudaMemcpyDeviceToHost);
      cudaEventRecord(E3,0);
      cudaEventSynchronize(E3);

      t0 = GetTime();
      SequentialFilter_Rotate(sImg, width, height, pixelWidth);
      t1 = GetTime(); 
      break;
    case 'M':
      
      cudaMalloc((unsigned char**)&dAux, numBytes);
      cudaMemcpy(dAux,hImg,numBytes,cudaMemcpyHostToDevice);

      cudaEventRecord(E1,0);
      cudaEventSynchronize(E1);
      KernelFilter_Mean<<<dimGrid,dimBlock>>>(dAux, dImg, width, height, pixelWidth);
      cudaEventRecord(E2,0);
      cudaEventSynchronize(E2);

      cudaMemcpy(hImg,dImg,numBytes,cudaMemcpyDeviceToHost);
      cudaEventRecord(E3,0);
      cudaEventSynchronize(E3);
    
      t0 = GetTime();
      SequentialFilter_Mean(sImg, width, height, pixelWidth);
      t1 = GetTime(); 
      break;
    case 'G':
      cudaMalloc((unsigned char**)&dAux, numBytes);
      cudaMemcpy(dAux,hImg,numBytes,cudaMemcpyHostToDevice);
      
      cudaEventRecord(E1,0);
      cudaEventSynchronize(E1);
      KernelFilter_Gaussiano<<<dimGrid,dimBlock>>>(dAux, dImg, width, height, pixelWidth);
      cudaEventRecord(E2,0);
      cudaEventSynchronize(E2);

      cudaMemcpy(hImg,dImg,numBytes,cudaMemcpyDeviceToHost);
      cudaEventRecord(E3,0);
      cudaEventSynchronize(E3);
    
      t0 = GetTime();
      SequentialFilter_Gaussiano(sImg, width, height, pixelWidth);
      t1 = GetTime(); 
      break;
    case 'S':
      cudaMalloc((unsigned char**)&dAux, numBytes);
      cudaMemcpy(dAux,hImg,numBytes,cudaMemcpyHostToDevice);

      cudaEventRecord(E1,0);
      cudaEventSynchronize(E1);
      KernelFilter_Sobel<<<dimGrid,dimBlock>>>(dAux, dImg, width, height, pixelWidth);
      cudaEventRecord(E2,0);
      cudaEventSynchronize(E2);

      cudaMemcpy(hImg,dImg,numBytes,cudaMemcpyDeviceToHost);
      cudaEventRecord(E3,0);
      cudaEventSynchronize(E3);
    
      t0 = GetTime();
      SequentialFilter_Sobel(sImg, width, height, pixelWidth);
      t1 = GetTime();
      break;  

    default:
      break;
    }
    

    TiempoSequencial[i] = t1 - t0;
    cudaError_t cd1 = cudaEventElapsedTime(&TiempoKernel[i], E1, E2);
    cudaError_t cd2 =  cudaEventElapsedTime(&TiempoGlobal[i], E0, E3);
  }

  float maxS = TiempoSequencial[0];
  float minS = TiempoSequencial[0];
  float maxG = TiempoGlobal[0]; 
  float minG = TiempoGlobal[0];
  float maxK = TiempoKernel[0];
  float minK = TiempoKernel[0];
  mediaSequencial = maxS;
  mediaGlobal = maxG;
  mediaKernel = maxK;
  for (int i = 1; i < ITER; ++i) {
    float S = TiempoSequencial[i];
    float G = TiempoGlobal[i];
    float K = TiempoKernel[i];
    
    mediaSequencial += S;
    mediaGlobal += G;
    mediaKernel += K;

    if (S > maxS) maxS = S;
    else if (S < minS) minS =  S;

    if (G > maxG) maxG = G;
    else if (G < minG) minG =  G;

    if (K > maxK) maxK = K;
    else if (K < minK) minK =  K;
  }

  mediaSequencial = TiempoSequencial[0];
  mediaGlobal = TiempoGlobal[0];
  mediaKernel = TiempoKernel[0];

  mediaSequencial /= (ITER - 2);
  mediaGlobal /= (ITER - 2);
  mediaKernel /= (ITER - 2);
 

  printf("\nKERNEL %s\n", filter);
  printf("Sequential Time: %f ms\n", mediaSequencial);
  printf("GPU utilizada: %d\n", gpu);
  printf("Dimensiones: %dx%d \n", width, height);
  printf("nThreads: %dx%d (%d)\n", nThreadsX, nThreadsY, nThreadsX * nThreadsY);
  printf("nBlocks: %dx%d (%d)\n", nBlocksX, nBlocksY, nBlocksX*nBlocksY);
  if (PINNED) printf("Usando Pinned Memory\n");
  else printf("NO usa Pinned Memory\n");
  printf("Tiempo Global: %4.6f milseg\n", mediaGlobal);
  printf("Tiempo Kernel: %4.6f milseg\n", mediaKernel);
  printf("Speedup Global: %4.6f\n", mediaSequencial/mediaGlobal);
  printf("Speedup Kernel: %4.6f\n", mediaSequencial/mediaKernel);
  //printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoTotal));
  //printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoKernel));

//////////////////////////////////////////////////////////////
///////////////////TEST///////////////////////////////////////
//////////////////////////////////////////////////////////////

  int correct = 1;
  int i = 0;
  while (correct && i < width*height*pixelWidth) {
  	correct = sImg[i] == hImg[i] && sImg[i + 1] == hImg[i + 1] && sImg[i + 2] == hImg[i + 2];
	  i += 3;
  }
  if (correct) printf("TEST PASS \n");
  else {
    printf("TEST FAIL, PIXEL: %i \n", i);
    printf("SEQ: 0: %u ,1: %u , 2: %u \n", sImg[i - 3], sImg[i - 2], sImg[i - 1]); 
    printf("KERNEL: 0: %u ,1: %u , 2: %u \n", hImg[i - 3], hImg[i - 2], hImg[i - 1]);  
  }

  stbi_write_png(fileOutSeq, width, height, pixelWidth, sImg, 0);
  stbi_write_png(fileOutKernel, width, height, pixelWidth, hImg, 0); 
//////////////////////////////////////////////////////////////
///////////////////LIMPIEZA///////////////////////////////////
//////////////////////////////////////////////////////////////

  cudaEventDestroy(E1);
  cudaEventDestroy(E2);

  if (PINNED) cudaFreeHost(hImg);
  else free(hImg);
  cudaFree(dImg);
  cudaFree(dAux);  
  free(sImg);

}

void CheckCudaError(char sms[], int line) {
  cudaError_t error;
 
    /* Descomentar si se quiere debugar el kernel. Cuidado! eso provoca la ejecucion sincrona del kernel */
    // #define DEBUG 1
#ifndef DEBUG
    #define DEBUG 0
#endif
  if (DEBUG) cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }
  //else printf("(OK) %s \n", sms);
}


float GetTime(void)        
{
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((float)tim.tv_sec + (float)tim.tv_usec / 1000000.0)*1000.0;
}
