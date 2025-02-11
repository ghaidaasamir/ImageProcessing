#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void ConvertToGrayscaleKernelShared(unsigned char *input, unsigned char *output, 
                                  int inputHeight, int inputWidth)
{

    extern __shared__ unsigned char sharedInput[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sharedWidth = blockDim.x;
    int sharedHeight = blockDim.y;
    
    int idx = (y * inputWidth + x) * 3;
    int idx_shared = (ty * sharedWidth + tx) * 3;
    sharedInput[idx_shared] = input[idx];
    sharedInput[idx_shared+1] = input[idx+1];
    sharedInput[idx_shared+2] = input[idx+2];

    __syncthreads();
    if (x < inputWidth && y < inputHeight) {
        float gray = 0.299f * sharedInput[idx_shared + 2] + 0.587f * sharedInput[idx_shared + 1] + 0.114f * sharedInput[idx_shared]; // BGR to Gray
        output[y * inputWidth + x] = static_cast<unsigned char>(gray);
    }

}

__global__ void ConvertToGrayscaleKernel(unsigned char *input, unsigned char *output, 
                                  int inputHeight, int inputWidth)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < inputWidth && y < inputHeight) {
        int idx = (y * inputWidth + x) * 3;
        float gray = 0.299f * input[idx + 2] + 0.587f * input[idx + 1] + 0.114f * input[idx]; // BGR to Gray
        output[y * inputWidth + x] = static_cast<unsigned char>(gray);
    }

}

void ConvertToGrayscale(){

    cv::Mat image = cv::imread("New2.png");

    std::cout << "Width: " << image.cols << std::endl;   
    std::cout << "Height: " << image.rows << std::endl; 

    int inputHeight = image.rows;
    int inputWidth = image.cols;

    std::vector<unsigned char> src(inputWidth * inputHeight * 3);  
    std::vector<unsigned char> dst(inputWidth * inputHeight);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, 3 * inputHeight * inputWidth * sizeof(unsigned char));
    cudaMalloc(&d_output, inputHeight * inputWidth * sizeof(unsigned char));

    memcpy(src.data(), image.data, 3 * inputHeight * inputWidth);

    cudaMemcpy(d_input, src.data(), 3 * inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((inputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (inputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    ConvertToGrayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, inputHeight, inputWidth);

    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data(), d_output, inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat outputImage(image.rows, image.cols, CV_8UC1, dst.data());
    cv::imshow("Grayscale Image", outputImage);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);

}

int main(){
    ConvertToGrayscale();
    return 0;
}