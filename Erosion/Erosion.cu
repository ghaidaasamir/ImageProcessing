#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void ErosionKernelShared(unsigned char *input, unsigned char *output, unsigned char *kernel, int outputHeight, int outputWidth, int kernelSize)
{
    extern __shared__ unsigned char sharedInput[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sharedWidth = blockDim.x + kernelSize - 1; 
    int k_center_y = kernelSize / 2;
    int k_center_x = kernelSize / 2;
    
    
    if (x >= k_center_x && x < (outputWidth-k_center_x) && y >= k_center_y && y < (outputHeight-k_center_y)) {        
        for(int i =0; i < kernelSize;i++){
            for(int j=0; j < kernelSize;j++){
                int idx = ((y + i - k_center_y) * outputWidth) + j + (x + j - k_center_x);
                int idx_shared = ((ty + i - k_center_y) * outputWidth) + j + (tx + j - k_center_x);
                sharedInput[idx_shared] = input[idx];
            }
        }
        __syncthreads();
    }
    else{
        sharedInput[ty* outputWidth + tx] = 0;
    }
    if (x >= k_center_x && x < (outputWidth-k_center_x) && y >= k_center_y && y < (outputHeight-k_center_y)) {        
        bool match = true;
        for(int i =0; i < kernelSize;i++){
            for(int j=0; j < kernelSize;j++){
                if(kernel[i * kernelSize + j] == 1){
                    int idx_shared = ((ty + i - k_center_y) * outputWidth) + j + (tx + j - k_center_x);
                    if(sharedInput[idx_shared] == 0){
                        match = false;
                        break;
                    }
                }
            }
            if(!match){
                break;
            }
        }
        if(match){
            output[y*outputWidth + x] = 1;
        }
        else {
            output[y*outputWidth + x] = 0; 
        }
    }

}

__global__ void ErosionKernel(unsigned char *input, unsigned char *output, unsigned char *kernel, int outputHeight, int outputWidth, int kernelSize)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int k_center_y = kernelSize / 2;
    int k_center_x = kernelSize / 2;
    
    if (x >= k_center_x && x < (outputWidth-k_center_x) && y >= k_center_y && y < (outputHeight-k_center_y)) {        
        bool match = true;
        for(int i =0; i < kernelSize;i++){
            for(int j=0; j < kernelSize;j++){
                if(kernel[i * kernelSize + j] == 1){
                    int idx = ((y + i - k_center_y) * outputWidth) + j + (x + j - k_center_x);
                    if(input[idx] == 0){
                        match = false;
                        break;
                    }
                }
            }
            if(!match){
                break;
            }
        }
        if(match){
            output[y*outputWidth + x] = 1;
        }
        else {
            output[y*outputWidth + x] = 0; 
        }
    }

}


void Erosion(){

    cv::Mat image = cv::imread("lanes.jpeg");

    std::cout << "Width: " << image.cols << std::endl;   
    std::cout << "Height: " << image.rows << std::endl; 

    int inputHeight = image.rows;
    int inputWidth = image.cols;
    int kernelSize = 5;
    int outputHeight = inputHeight;
    int outputWidth = inputWidth;

    std::vector<unsigned char> src(inputWidth * inputHeight * 3);  
    std::vector<unsigned char> dst(outputWidth * outputHeight * 3);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, 3 * inputHeight * inputWidth * sizeof(unsigned char));
    cudaMalloc(&d_output, 3 * outputHeight * outputWidth * sizeof(unsigned char));

    memcpy(src.data(), image.data, 3 * inputHeight * inputWidth);

    cudaMemcpy(d_input, src.data(), 3 * inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((inputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (inputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    ErosionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, outputHeight, outputWidth, kernelSize);

    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data(), d_output, 3 * outputHeight * outputWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat outputImage(image.rows, image.cols, CV_8UC3, dst.data());
    cv::imshow("Erosion Image", outputImage);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);

}

int main(){
    Erosion();
    return 0;
}