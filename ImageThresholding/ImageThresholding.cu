#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <math.h>

__global__ void ImageThresholdingKernel(unsigned char *input, unsigned char *output, int inputHeight, int inputWidth, int threshold)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < inputHeight && y < inputWidth) {        

            int index = (x * inputWidth + y);
            
            if((int)input[index]>threshold){
                output[index] = 255;
            }
            else{
                output[index] = 0;
            }
        
    }
}

void ImageThresholding(){

    cv::Mat image = cv::imread("lanes.jpeg");

    cv::Mat grayImage;

    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    std::cout << "Width: " << grayImage.cols << std::endl;   
    std::cout << "Height: " << grayImage.rows << std::endl; 

    int inputHeight = grayImage.rows;
    int inputWidth = grayImage.cols;
    int threshold = 120;

    std::vector<unsigned char> src(inputWidth * inputHeight);  
    std::vector<unsigned char> dst(inputWidth * inputHeight,0);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, inputHeight * inputWidth * sizeof(unsigned char));
    cudaMalloc(&d_output, inputHeight * inputWidth * sizeof(unsigned char));

    memcpy(src.data(), grayImage.data, inputHeight * inputWidth);

    cudaMemcpy(d_input, src.data(), inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((inputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (inputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    ImageThresholdingKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, inputHeight, inputWidth, threshold);

    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data(), d_output, inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat outputImage(grayImage.rows, grayImage.cols, CV_8UC1, dst.data());
    cv::imshow("Image", outputImage);
    cv::waitKey(0);

    cv::imwrite("output.jpeg", outputImage);

    cudaFree(d_input);
    cudaFree(d_output);

}

int main(){
    ImageThresholding();
    return 0;
}