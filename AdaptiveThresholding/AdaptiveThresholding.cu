#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <math.h>

__global__ void AdaptiveThresholdingKernelShared(unsigned char *input, unsigned char *output, int inputHeight, int inputWidth, int block_size, int C)
{
    extern __shared__ unsigned char sharedInput[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sharedWidth = blockDim.x + block_size - 1;
    int sharedHeight = blockDim.y + block_size - 1;

    int inputX = x - block_size / 2;
    int inputY = y - block_size / 2;

    for (int i = ty; i < sharedHeight; i += blockDim.y) {
        for (int j = tx; j < sharedWidth; j += blockDim.x) {
            int sharedIdx = i * sharedWidth + j;
            int globalX = inputX + j;
            int globalY = inputY + i;
            if (globalX >= 0 && globalX < inputWidth && globalY >= 0 && globalY < inputHeight) {
                sharedInput[sharedIdx] = input[globalY * inputWidth + globalX];
            } else {
                sharedInput[sharedIdx] = 0;
            }
        }
    }

    __syncthreads();

    if (x < inputWidth && y < inputHeight) {
        float sum = 0;
        int count = 0;
        int half_block = block_size / 2;
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                int ni = ty + i;
                int nj = tx + j;
                if (ni < sharedHeight && nj < sharedWidth) {
                    sum += sharedInput[ni * sharedWidth + nj];
                    count++;
                }
            }
        }
        float avg = sum / count;
        int threshold = avg - C;
        int out_idx = y * inputWidth + x;
        if (input[out_idx] > threshold) {
            output[out_idx] = 255;
        } else {
            output[out_idx] = 0;
        }
    }
}


__global__ void AdaptiveThresholdingKernel(unsigned char *input, unsigned char *output, int inputHeight, int inputWidth, int block_size, int C)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < inputWidth && y < inputHeight) {        
        int count = 0;
        float sum = 0;
        int half_block = block_size / 2;
        for(int i =-half_block; i <= half_block;i++){
            for(int j=-half_block; j <= half_block;j++){
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < inputWidth && ny >= 0 && ny < inputHeight) {
                    int idx = ny * inputWidth + nx;
                    sum += input[idx];
                    count++;
                }
            }
        }
        float avg = sum / count;
        int threshold = avg - C;
        int out_idx = y * inputWidth + x;
        if(input[out_idx] > threshold){
            output[out_idx] = 255;   
        }
        else{
            output[out_idx] = 0;
       }
    }
}


void AdaptiveThresholding(){

    cv::Mat image = cv::imread("lanes.jpeg");

    cv::Mat grayImage;

    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    std::cout << "Width: " << grayImage.cols << std::endl;   
    std::cout << "Height: " << grayImage.rows << std::endl; 

    int inputHeight = grayImage.rows;
    int inputWidth = grayImage.cols;
    int block_size = 11;  
    int C = 2;            

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

    AdaptiveThresholdingKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, inputHeight, inputWidth, block_size, C);

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
    AdaptiveThresholding();
    return 0;
}
