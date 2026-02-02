#include "test_2sm_mma.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Kernel 实现 (cluster dims 在 launch 时指定)
__global__ void test_utcmma_ss_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    float* __restrict__ P_out
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);
    
    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    
    // 1. 初始化 barrier
    if (warp_idx == 0) {
        if (cute::elect_one_sync()) {
            smem.bar_mma_done.init(1);
            cutlass::arch::fence_barrier_init();
        }
    }
    
    cute::cluster_sync();  // We must add a cluster_sync() here, or operations from CTA1 may launch before barrier initialization in CTA0
    
    // 2. 分配 TMEM：512 列，供 2 个 CTA 共享
    if (warp_idx == 0) {
        if (cute::elect_one_sync()) {
            cute::TMEM::Allocator2Sm().allocate(512, smem.tmem_start_addr);
            TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);
            cute::TMEM::Allocator2Sm().release_allocation_lock();
        }
    }
    
    __syncthreads();
    cute::cluster_sync();
    
    // 3. 构造 SMEM Tensor（先构造，用于正确加载数据）
    auto sQ = cute::make_tensor(cute::make_smem_ptr(smem.q), SmemLayoutQ{});
    auto sK = cute::make_tensor(cute::make_smem_ptr(smem.k), SmemLayoutK{});
    
    // 4. 从 Global Memory 加载 Q, K 到 SMEM
    // 每个 CTA 加载各自的一半：CTA0 加载前64行，CTA1 加载后64行
    // 使用 CuTe tensor 的布局映射来正确访问 swizzled 布局
    const int q_row_offset = cta_idx * (M/2);
    const int k_row_offset = cta_idx * (N/2);
    
    // 加载 Q: [64, 256]
    // 使用 tensor 的 (row, col) 索引，让 CuTe 处理布局映射
    // 每个线程处理多行，在列维度上并行
    constexpr int Q_ELEMENTS = (M/2) * K_DIM;
    constexpr int Q_ELEMENTS_PER_THREAD = (Q_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
    for (int i = 0; i < Q_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * Q_ELEMENTS_PER_THREAD + i;
        if (linear_idx < Q_ELEMENTS) {
            int row = linear_idx / K_DIM;
            int col = linear_idx % K_DIM;
            int global_row = q_row_offset + row;
            sQ(row, col) = Q[global_row * K_DIM + col];
        }
    }
    
    // 加载 K: [64, 256]
    constexpr int K_ELEMENTS = (N/2) * K_DIM;
    constexpr int K_ELEMENTS_PER_THREAD = (K_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
    for (int i = 0; i < K_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * K_ELEMENTS_PER_THREAD + i;
        if (linear_idx < K_ELEMENTS) {
            int row = linear_idx / K_DIM;
            int col = linear_idx % K_DIM;
            int global_row = k_row_offset + row;
            sK(row, col) = K[global_row * K_DIM + col];
        }
    }
    
    __syncthreads();
    cute::cluster_sync();
    
    // 5. 初始化 TiledMMA 并分配 TMEM fragment
    TiledMMA_P tiled_mma;
    auto tP = cute::partition_fragment_C(tiled_mma, cute::Shape<cute::Int<M/2>, cute::Int<N>>{});
    tP.data().get() = smem.tmem_start_addr[0];  // 使用动态分配的 TMEM 地址
    
    // 6. 执行 utcmma_ss (只在 CTA0 的特定 warp 中执行，参考 phase1.cuh:494-540)
    if (cta_idx == 0 && warp_idx == 0 && cute::elect_one_sync()) {
        ku::tcgen05_after_thread_sync();
        ku::utcmma_ss(tiled_mma, sQ, sK, tP, true);  // clear_accum = true
        ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_mma_done, 1|2);
    }
    
    // 等待 MMA 完成
    smem.bar_mma_done.wait(0);
    ku::tcgen05_after_thread_sync();
    
    // 7. 从 TMEM 读取结果到寄存器，然后写入全局内存
    // 每个 CTA 读取一半结果：CTA0 读 P[0:64, :], CTA1 读 P[64:128, :]
    // TMEM 布局：每个 CTA 存 [M/2, N] = [64, 128] 的结果
    
    // 使用 tmem_ld 读取：每个线程读取若干元素
    // 32dp32bNx 模式：32 个数据路径，每个路径 32bit (即 1 个 float)
    
    const int p_row_offset = cta_idx * (M/2);
    
    // 每个线程处理多行
    // 总共 64 行，128 列，每个 CTA 128 线程
    // 简单方案：每个线程读取一行的一部分
    
    // TMEM 中 tP 的数据布局需要按照 MMA traits 中的 CLayout 来解析
    // CLayout: Layout<Shape<_2, Shape<Int<M/2>, Int<N>>>, Stride<Int<M/2>, Stride<_1, Int<M>>>>
    // 这意味着数据是按 (cta, (row, col)) 的顺序存储的
    
    // 对于 M=128, N=128:
    // - 每个 CTA 存储 64 行 x 128 列 = 8192 个 float
    // - TMEM 地址 = TMEM_COL_P + row + col * M
    
    // 每个线程读取并写出若干元素
    constexpr int ELEMENTS_PER_THREAD = (M/2 * N) / NUM_THREADS;  // 8192 / 128 = 64
    // 读取 TMEM 到寄存器（参考 phase1.cuh 的用法）
    // 使用 32dp32bNx 批量读取模式：
    // - 32dp: 32 个数据路径（对应 warp 内 32 个线程）
    // - 32b: 每个数据路径 32bit（1 个 float）
    // - Nx: 在列方向重复 N 次，一次读取 N 列数据
    
    // 使用 float2 数组配合批量读取（参考 phase1.cuh 第176行）
    // ELEMENTS_PER_THREAD = 64，对应 32 个 float2
    float2 local_p[ELEMENTS_PER_THREAD / 2];
    
    // 计算 TMEM 起始列地址（根据 CLayout stride: <M/2, <1, M>>）
    uint32_t tmem_col = smem.tmem_start_addr[0] + cta_idx * (M/2);
    
    // 批量读取 64 列数据（参考 phase1.cuh 第177行）
    // 64 是支持的 kNumElements 值（文档第77行）
    ku::tmem_ld_32dp32bNx<ELEMENTS_PER_THREAD>(tmem_col, local_p);
    
    // 同步原语（参考 phase1.cuh 第178-179行）
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();
    __syncthreads();
    
    // 转换为 float 指针用于后续写出
    float* local_p_float = reinterpret_cast<float*>(local_p);
    
    // 写入全局内存
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * ELEMENTS_PER_THREAD + i;
        int row = linear_idx / N;
        int col = linear_idx % N;
        int global_row = p_row_offset + row;
        P_out[global_row * N + col] = local_p_float[i];
    }

#endif
}

// C++ wrapper
void launch_test_utcmma_ss(
    const bf16* Q,
    const bf16* K,
    float* P_out,
    cudaStream_t stream
) {
    dim3 grid(2, 1, 1);  // 2 CTAs
    dim3 block(NUM_THREADS, 1, 1);
    
    // Cluster 配置
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaLaunchKernelEx(&config, test_utcmma_ss_kernel, Q, K, P_out);
}

// Python binding: 接受 torch tensor
torch::Tensor utcmma_ss_forward(torch::Tensor Q, torch::Tensor K) {
    // 检查输入
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(Q.dim() == 2 && Q.size(0) == M && Q.size(1) == K_DIM, 
                "Q shape must be [128, 256]");
    TORCH_CHECK(K.dim() == 2 && K.size(0) == N && K.size(1) == K_DIM,
                "K shape must be [128, 256]");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    
    // 创建输出 tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(Q.device());
    torch::Tensor P_out = torch::empty({M, N}, options);
    
    // 获取数据指针
    const bf16* Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>());
    const bf16* K_ptr = reinterpret_cast<const bf16*>(K.data_ptr<at::BFloat16>());
    float* P_out_ptr = P_out.data_ptr<float>();
    
    // 获取 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // 调用 kernel
    launch_test_utcmma_ss(Q_ptr, K_ptr, P_out_ptr, stream);
    
    return P_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("utcmma_ss", &utcmma_ss_forward, "Test utcmma_ss (CUDA)");
}
