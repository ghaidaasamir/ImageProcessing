#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <math.h>

__global__ void HistogramEqualizationKernel(unsigned char *input, int *hist, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size){
        hist[input[x]] += 1;
    }
}

__global__ void calculate_cdf(int *hist, float *cdf, int numBins) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < numBins) {
        for (int i = 0; i <= x; i++) {
            cdf[x] += hist[i];
        }
    }
}

__global__ void map_pixels(unsigned char *input, unsigned char *output, float *cdf, int size, float minCdf, float maxCdf) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < size) {
        output[x] = (unsigned char)((cdf[input[x]] - minCdf) / (maxCdf - minCdf) * 255);
    }
}

void HistogramEqualization(){

    cv::Mat image = cv::imread("lanes.jpeg");

    cv::Mat grayImage;

    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    std::cout << "Width: " << grayImage.cols << std::endl;   
    std::cout << "Height: " << grayImage.rows << std::endl; 

    int inputHeight = grayImage.rows;
    int inputWidth = grayImage.cols;

    std::vector<unsigned char> src(inputWidth * inputHeight);  
    std::vector<unsigned char> dst(inputWidth * inputHeight,0);

    unsigned char* d_input;
    unsigned char* d_output;
    int *hist;
    float *cdf;

    int size = inputWidth * inputHeight;

    cudaMalloc(&d_input, inputHeight * inputWidth * sizeof(unsigned char));
    cudaMalloc(&d_output, inputHeight * inputWidth * sizeof(unsigned char));
    cudaMalloc(&hist, 256 * sizeof(int));
    cudaMalloc(&cdf, 256 * sizeof(float));

    memcpy(src.data(), grayImage.data, inputHeight * inputWidth);

    cudaMemcpy(d_input, src.data(), inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(hist, 0, 256 * sizeof(int));

    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    HistogramEqualizationKernel<<<numBlocks, threadsPerBlock>>>(d_input, hist, size);
    calculate_cdf<<<numBlocks, 256, 256*sizeof(int)>>>(hist, cdf, 256);

    float minCdf, maxCdf;
    cudaMemcpy(&minCdf, cdf, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxCdf, &cdf[256 - 1], sizeof(float), cudaMemcpyDeviceToHost);

    map_pixels<<<numBlocks, threadsPerBlock>>>(d_input, d_output, cdf, size, minCdf, maxCdf);

    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data(), d_output, inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat outputImage(image.rows, image.cols, CV_8UC1, dst.data());
    cv::imshow("Image", outputImage);
    cv::waitKey(0);

    cv::imwrite("output.jpeg", outputImage);

    cudaFree(d_input);
    cudaFree(d_output);

}

int main(){
    HistogramEqualization();
    return 0;
}