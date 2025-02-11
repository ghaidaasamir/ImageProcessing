#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <math.h>

__global__ void EdgeDetectionKernelShared(unsigned char *input, unsigned char *output, int inputHeight, int inputWidth)
{
    extern __shared__ unsigned char sharedInput[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sharedWidth = blockDim.x + 2; 
    
    if (x < inputWidth && y < inputHeight){
        for (int c = 0; c < 3; c++) {
            int index_shared = (tx * sharedWidth + ty) * 3 + c;
            int index_right_shared = index_shared + 3;
            int index_left_shared = index_shared - 3;
            int index_top_shared = index_shared - sharedWidth * 3;
            int index_bottom_shared = index_shared + sharedWidth * 3;
            
            int index = (x * inputWidth + y) * 3 + c;
            int index_right = index + 3;
            int index_left = index - 3;
            int index_top = index - inputWidth * 3;
            int index_bottom = index + inputWidth * 3;
            
            sharedInput[index_shared] = input[index];
            sharedInput[index_right_shared] = input[index_right];
            sharedInput[index_left_shared] = input[index_left];
            sharedInput[index_top_shared] = input[index_top];
            sharedInput[index_bottom_shared] = input[index_bottom];

        }
    }    
    __syncthreads();

    if (tx >= 1 && tx < (blockDim.x - 1) && ty >= 1 && ty < (blockDim.y - 1) &&
        x >= 1 && x < (inputWidth - 1) && y >= 1 && y < (inputHeight - 1)) {        
        for (int c = 0; c < 3; c++) {
            int index_shared = (tx * inputWidth + ty) * 3 + c;
            int index_right_shared = index_shared + 3;
            int index_left_shared = index_shared - 3;
            int index_top_shared = index_shared - inputWidth * 3;
            int index_bottom_shared = index_shared + inputWidth * 3;
            
            int dx = (int)sharedInput[index_right_shared] - (int)sharedInput[index_left_shared];
            int dy = (int)sharedInput[index_bottom_shared] - (int)sharedInput[index_top_shared];

            int gradient_magnitude = (int)sqrtf(dx * dx + dy * dy);

            output[index_shared] = (unsigned char)max(0, min(255, gradient_magnitude));
        }
    }
}

__global__ void EdgeDetectionKernel(unsigned char *input, unsigned char *output, int inputHeight, int inputWidth)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < (inputHeight - 1) && y >= 1 && y < (inputWidth - 1)) {        
        for (int c = 0; c < 3; c++) {
            int index = (x * inputWidth + y) * 3 + c;
            int index_right = index + 3;
            int index_left = index - 3;
            int index_top = index - inputWidth * 3;
            int index_bottom = index + inputWidth * 3;
            
            int dx = (int)input[index_right] - (int)input[index_left];
            int dy = (int)input[index_bottom] - (int)input[index_top];

            int gradient_magnitude = (int)sqrtf(dx * dx + dy * dy);

            output[index] = (unsigned char)max(0, min(255, gradient_magnitude));
        }
    }
}

void EdgeDetection(){

    cv::Mat image = cv::imread("lanes.jpeg");

    std::cout << "Width: " << image.cols << std::endl;   
    std::cout << "Height: " << image.rows << std::endl; 

    int inputHeight = image.rows;
    int inputWidth = image.cols;

    std::vector<unsigned char> src(inputWidth * inputHeight * 3);  
    std::vector<unsigned char> dst(inputWidth * inputHeight * 3,0);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, 3 * inputHeight * inputWidth * sizeof(unsigned char));
    cudaMalloc(&d_output, 3 * inputHeight * inputWidth * sizeof(unsigned char));

    memcpy(src.data(), image.data, 3 * inputHeight * inputWidth);

    cudaMemcpy(d_input, src.data(), 3 * inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((inputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (inputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    EdgeDetectionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, inputHeight, inputWidth);

    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data(), d_output, 3 * inputHeight * inputWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat outputImage(image.rows, image.cols, CV_8UC3, dst.data());
    cv::imshow("Blurred Image", outputImage);
    cv::waitKey(0);

    cv::imwrite("output.jpeg", outputImage);

    cudaFree(d_input);
    cudaFree(d_output);

}

int main(){
    EdgeDetection();
    return 0;
}