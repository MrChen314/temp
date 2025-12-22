int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("==========================================\n");
    printf("         GPU 设备信息查询\n");
    printf("==========================================\n\n");
    printf("检测到 %d 个CUDA设备\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("设备 %d: %s\n", i, prop.name);
        printf("------------------------------------------\n");
        
        // 计算能力
        printf("计算能力: %d.%d\n", prop.major, prop.minor);
        
        // SM相关
        printf("\n【SM信息】\n");
        printf("  SM数量: %d\n", prop.multiProcessorCount);
        printf("  每SM最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  每SM最大Block数: %d\n", prop.maxBlocksPerMultiProcessor);
        
        // Warp相关
        printf("\n【Warp信息】\n");
        printf("  Warp大小: %d 线程\n", prop.warpSize);
        int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        printf("  每SM最大Warp数: %d\n", maxWarpsPerSM);
        
        // Block/Thread相关
        printf("\n【Block/Thread限制】\n");
        printf("  每Block最大线程数: %d\n", prop.maxThreadsPerBlock);
        printf("  Block维度限制: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Grid维度限制: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        
        // 内存相关
        printf("\n【内存信息】\n");
        printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  每Block共享内存: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  每SM共享内存: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  每Block寄存器数: %d\n", prop.regsPerBlock);
        printf("  每SM寄存器数: %d\n", prop.regsPerMultiprocessor);
        
        // 性能特性
        printf("\n【性能特性】\n");
        printf("  GPU时钟频率: %.2f GHz\n", prop.clockRate / 1e6);
        printf("  内存时钟频率: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  内存总线宽度: %d bit\n", prop.memoryBusWidth);
        printf("  L2缓存大小: %d KB\n", prop.l2CacheSize / 1024);
        
        printf("\n");
    }
    
    return 0;
}


【SM信息】
  SM数量: 78
  每SM最大线程数: 2048
  每SM最大Block数: 32

【Warp信息】
  Warp大小: 32 线程
  每SM最大Warp数: 64

【Block/Thread限制】
  每Block最大线程数: 1024
  Block维度限制: (1024, 1024, 64)
  Grid维度限制: (2147483647, 65535, 65535)

【内存信息】
  全局内存: 95.00 GB
  每Block共享内存: 48 KB
  每SM共享内存: 228 KB
  每Block寄存器数: 65536
  每SM寄存器数: 65536

【性能特性】
  GPU时钟频率: 1.98 GHz
  内存时钟频率: 2.62 GHz
  内存总线宽度: 6144 bit
  L2缓存大小: 61440 KB
