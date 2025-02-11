#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void BlurringKernelShared(unsigned char *input, unsigned char *output, int outputHeight, int outputWidth, int kernelSize)
{
    int offset = kernelSize / 2;
    extern __shared__ unsigned char sharedInput[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sharedWidth = blockDim.x + 2 * offset;
    int sharedHeight = blockDim.y + 2 * offset;

    for (int i = ty; i < sharedHeight; i += blockDim.y) {
        for (int j = tx; j < sharedWidth; j += blockDim.x) {
            int globalX = blockIdx.x * blockDim.x + j - offset;
            int globalY = blockIdx.y * blockDim.y + i - offset;
            int sharedIdx = i * sharedWidth + j;

            if (globalX >= 0 && globalX < outputWidth && globalY >= 0 && globalY < outputHeight) {
                int globalIdx = (globalY * outputWidth + globalX) * 3;
                sharedInput[sharedIdx * 3] = input[globalIdx];
                sharedInput[sharedIdx * 3 + 1] = input[globalIdx + 1];
                sharedInput[sharedIdx * 3 + 2] = input[globalIdx + 2];
            } else {
                sharedInput[sharedIdx * 3] = 0;
                sharedInput[sharedIdx * 3 + 1] = 0;
                sharedInput[sharedIdx * 3 + 2] = 0;
            }
        }
    }

    __syncthreads();

    if (x >= offset && x < (outputWidth - offset) && y >= offset && y < (outputHeight - offset)) {
        float sumR = 0, sumG = 0, sumB = 0;
        for (int i = -offset; i <= offset; i++) {
            for (int j = -offset; j <= offset; j++) {
                int nx = tx + offset + i;
                int ny = ty + offset + j;
                int idx = (ny * sharedWidth + nx) * 3;
                sumB += sharedInput[idx];
                sumG += sharedInput[idx + 1];
                sumR += sharedInput[idx + 2];
            }
        }
        float avgR = sumR / (kernelSize * kernelSize);
        float avgG = sumG / (kernelSize * kernelSize);
        float avgB = sumB / (kernelSize * kernelSize);
        int out_idx = (y * outputWidth + x) * 3;
        output[out_idx] = static_cast<unsigned char>(avgB);
        output[out_idx + 1] = static_cast<unsigned char>(avgG);
        output[out_idx + 2] = static_cast<unsigned char>(avgR);
    }
}

__global__ void BlurringKernel(unsigned char *input, unsigned char *output, int outputHeight, int outputWidth, int kernelSize)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = kernelSize / 2;

    if (x >= offset && x < (outputWidth - offset) && y >= offset && y < (outputHeight - offset)) {        
        int c = 0;
        float sumR = 0, sumG = 0, sumB = 0;
        for(int i =-offset; i <= offset;i++){
            for(int j=-offset; j <= offset;j++){
            
                int idx = ((j+y) * outputWidth + i+x)*3;
                sumB += input[idx];
                sumG += input[idx + 1];
                sumR += input[idx + 2];
                c ++;
            }
        }
        float avgR = sumR / c;
        float avgG = sumG / c;
        float avgB = sumB / c;
        int out_idx = (y * outputWidth + x) * 3;
        output[out_idx] = static_cast<unsigned char>(avgB);
        output[out_idx + 1] = static_cast<unsigned char>(avgG);
        output[out_idx + 2] = static_cast<unsigned char>(avgR);
    }
}


void Blurring(){

    cv::Mat image = cv::imread("lanes.jpeg");

    std::cout << "Width: " << image.cols << std::endl;   
    std::cout << "Height: " << image.rows << std::endl; 

    int inputHeight = image.rows;
    int inputWidth = image.cols;
    int kernelSize = 7;
    int outputHeight = inputHeight-kernelSize/2;
    int outputWidth = inputWidth-kernelSize/2;

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

    BlurringKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, outputHeight, outputWidth, kernelSize);

    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data(), d_output, 3 * outputHeight * outputWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat outputImage(image.rows, image.cols, CV_8UC3, dst.data());
    cv::imshow("Blurred Image", outputImage);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);

}

int main(){
    Blurring();
    return 0;
}