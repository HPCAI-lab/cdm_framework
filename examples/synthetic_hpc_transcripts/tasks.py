"""Task content for the synthetic HPC transcripts: six debugging/optimization
tasks, each with the code, symptom, correct diagnosis, a flawed suggestion the
assistant may make, and task-specific material the student behaviours use."""

TASKS = {}

TASKS["mpi_matmul"] = dict(
    title="MPI row-block matrix multiplication",
    short="MPI matmul", api="MPI", lang="C with MPI",
    code="""int rows = N / size;
MPI_Scatter(A, rows*N, MPI_DOUBLE, localA, rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
for (int i = 0; i < rows; i++)
    for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int k = 0; k < N; k++) s += localA[i*N+k] * B[k*N+j];
        localC[i*N+j] = s;
    }
MPI_Gather(localC, rows*N, MPI_DOUBLE, C, rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);""",
    symptom="the result is wrong for N=1000 on 6 ranks but correct for N=1200",
    vague_symptom="it gives wrong numbers sometimes",
    config="mpicc -O3, OpenMPI 4.1, 2 nodes x 16 cores",
    goal="the result has to match the serial version to 1e-9 for any N up to 4096 and any rank count up to 32, without using an external BLAS",
    format_c="keep it in C and keep the row-block layout",
    success="so I can explain the fix in my lab report",
    diag="N / size is integer division, so when N isn't divisible by the rank count the last N % size rows of A are never sent to any rank and those rows of C are never computed. Use MPI_Scatterv and MPI_Gatherv with per-rank counts and displacements: give the first N % size ranks one extra row.",
    fix="""int base = N / size, extra = N % size;
for (int r = 0; r < size; r++) {
    int myrows = base + (r < extra);
    counts[r] = myrows * N;
    displs[r] = (r * base + (r < extra ? r : extra)) * N;
}
MPI_Scatterv(A, counts, displs, MPI_DOUBLE, localA, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);""",
    flaw="Set rows = N / size + 1 on every rank so the leftover rows are covered.",
    flaw_rebuttal="that reads past the end of A on the last rank. With N=1000 and 6 ranks, 6 x 167 = 1002 rows, but A only has 1000",
    flaw_ack="You're right, that overruns A on the last rank. Use MPI_Scatterv with explicit counts instead so each rank gets exactly its share.",
    mech="Integer division drops the remainder, so the last N % size rows of A are never sent to any rank and the matching rows of C are never computed; MPI_Scatterv lets each rank receive a different count.",
    close_words="integer division drops the remainder, so the last rows of A are never sent and those rows of C are never computed",
    own_words="when the rank count doesn't divide N, a few trailing rows fall between the ranks: nobody owns them, so C keeps whatever was in memory there. Giving the first few ranks one extra row closes the gap",
    loose_words="the mpi splits it wrong",
    subparts=["distributing rows of A when N % p != 0", "broadcasting B", "gathering C back in rank order", "checking C against the serial result"],
    first_part="the row distribution", later_part="the gather",
    dependency="Before touching performance I need the distribution right, because a faster wrong answer tells me nothing",
    timings="N=2048: 1 rank 41.2 s, 8 ranks 6.1 s, 16 ranks 4.9 s",
    keep="I'll write the scaling analysis myself since I have to defend it in the report; you check my Scatterv counts and displacements",
    reference="the serial triple loop on rank 0", ref_result="max abs error is 0 for N=1000 on 6 ranks and N=7 on 3 ranks",
    spot="ran it once with N=1000 on 6 ranks and the output looks right now",
    edge="N=7 on 3 ranks, N=1000 on 6, and N=4096 on 32",
    error_specific="the last 4 rows of C are all zeros for N=1000 on 6 ranks, so the leftover rows are never computed",
    dubious="MPI_Scatter_rows(), which is designed for exactly this row distribution",
    dubious_catch="I can't find MPI_Scatter_rows in the MPI 4.0 standard or the OpenMPI man pages, so I don't think it exists",
    dubious_ack="You're right, there is no MPI_Scatter_rows; that was a mistake. MPI_Scatterv is the standard routine for uneven counts.",
    prov_q="why Scatterv needs displacements and not just counts, and where that's documented",
    prov_a="The displacements tell the root where each rank's block starts in A; counts alone don't fix the offsets once blocks differ in size. The MPI standard's section on MPI_Scatterv describes both arrays.",
    profiler="mpiP shows 38% of the time in MPI_Bcast of B at 16 ranks",
    model_claim="The remaining time at 16 ranks is mostly computation, so adding ranks should keep helping.",
    reconcile="mpiP says 38% of the time at 16 ranks is in the Bcast of B, so it's communication, not compute; I'll trust the profile over the estimate and look at overlapping the broadcast",
    why=[("why does Scatterv need displacements and not just counts?", "Because once blocks differ in size, the root can't compute each rank's start offset from the rank number alone."),
         ("does the gather order depend on rank order?", "Yes. Gatherv places each rank's block at its displacement, so C comes back in row order as long as the displacements increase with rank."),
         ("so why does the Bcast of B start to dominate at 16 ranks?", "Every rank needs all of B, so broadcast volume grows with the rank count while each rank's compute share shrinks.")],
    policy="is submitting generated MPI code allowed under our course's AI policy, or do I have to write the Scatterv part myself?",
    privacy_leak="""my job script:
#SBATCH --account=cmpe4337-fkhan
#SBATCH -N 2
srun ./matmul 1000
I log in with fkhan@login1.hpc.example.edu""",
    redact="(I've replaced the account name and hostname with [redacted] since that's cluster login info)",
    bias_claim="MPI_Bcast is always the bottleneck in this kind of code.",
    bias_mid="That sounds like it depends on the interconnect.",
    bias_high="that claim is from one configuration; on our 2-node Ethernet setup it may not hold, so I'll measure it on our cluster before I believe it",
    stake_mid="I'll be careful with the shared cluster",
    stake_high="I won't run the 32-rank tests on the login node, and I'll keep the job short because the whole class shares the queue before Friday's deadline",
    attr_mid="I'll mention in the report that I used AI help",
    attr_high="in the report I'll mark the Scatterv/Gatherv count setup as AI-suggested and say the decomposition and the scaling analysis are mine",
    team_mid="this is my part of our group's code",
    team_high="my teammate wrote the input generator that stores A column-major, so before I merge I need the Scatterv layout to match hers or the rows will be scrambled",
)

TASKS["omp_race"] = dict(
    title="OpenMP matrix multiplication with a race condition",
    short="OpenMP matmul", api="OpenMP", lang="C with OpenMP",
    code="""double sum;
int i, j, k;
#pragma omp parallel for
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
        sum = 0.0;
        for (k = 0; k < N; k++) sum += A[i][k] * B[k][j];
        C[i][j] = sum;
    }""",
    symptom="the result changes from run to run with 8 threads but is correct with OMP_NUM_THREADS=1",
    vague_symptom="the answers are random",
    config="gcc 12 -O2 -fopenmp, 16-core Xeon node",
    goal="it has to give the same C as the single-thread run for N=2048 and get at least 6x speedup on 8 threads",
    format_c="keep the i-j-k loops and the 2D arrays",
    success="so I understand what went wrong before the quiz",
    diag="sum, j and k are declared outside the parallel region, so all threads share them and overwrite each other's values. Make them private: declare them inside the loop, or add private(j, k, sum) to the pragma.",
    fix="""#pragma omp parallel for private(j, k, sum)
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
        sum = 0.0;
        for (k = 0; k < N; k++) sum += A[i][k] * B[k][j];
        C[i][j] = sum;
    }""",
    flaw="Wrap the update in #pragma omp critical so only one thread changes sum at a time.",
    flaw_rebuttal="a critical section would serialize the inner loop and throw away the speedup, and it still leaves j and k shared. The variables just need to be private",
    flaw_ack="Agreed, the critical section serializes the work and doesn't fix j and k. private(j, k, sum) is the right fix.",
    mech="The loop variables j and k and the accumulator sum are shared because they were declared outside the parallel region, so threads overwrite each other's partial sums and indices.",
    close_words="j, k and sum are shared because they were declared outside the parallel region, so threads overwrite each other's partial sums",
    own_words="each thread thinks it owns sum, but there's only one copy, so two threads adding into it at once lose updates. Giving every thread its own copy removes the collision",
    loose_words="the threads mess it up",
    subparts=["finding which variables are shared", "fixing the data-sharing clauses", "checking the result against one thread", "measuring speedup"],
    first_part="the data-sharing bug", later_part="the speedup",
    dependency="I need the result correct first, because timing a wrong result is pointless",
    timings="N=2048: 1 thread 38.0 s, 4 threads 10.4 s, 8 threads 6.3 s",
    keep="I'll do the speedup measurements and the write-up myself; you check whether my private clause covers everything",
    reference="the OMP_NUM_THREADS=1 run", ref_result="C matches element by element across 10 runs with 8 threads",
    spot="ran it once with 8 threads and it matched",
    edge="1, 2, 8 and 16 threads, N=17 and N=2048",
    error_specific="k is still shared in your version, so the inner loop index races too",
    dubious="the OpenMP 'atomicprivate' clause",
    dubious_catch="there's no atomicprivate clause in the OpenMP 5.2 spec, and gcc rejects it",
    dubious_ack="Correct, atomicprivate isn't an OpenMP clause; I shouldn't have suggested it. private(j, k, sum) is what you want.",
    prov_q="why declaring sum inside the loop makes it private, and where the spec says that",
    prov_a="Variables declared inside the parallel region are private by default; the OpenMP specification's data-sharing attribute rules cover this.",
    profiler="perf stat shows IPC dropping from 2.1 to 0.6 at 8 threads with the critical section",
    model_claim="The critical section adds only a small overhead here.",
    reconcile="perf says IPC fell from 2.1 to 0.6 at 8 threads with the critical section, so the overhead isn't small; I'm going with the private clause",
    why=[("why does it work with one thread?", "With one thread there's nobody to share sum with, so no update is lost."),
         ("why do j and k need to be private too, not just sum?", "They're loop indices written by every thread; if shared, one thread can advance another's loop."),
         ("so why is the speedup only about 6x on 8 threads after the fix?", "The B[k][j] access strides down columns, so cache misses and memory bandwidth limit scaling.")],
    policy="our instructor said we can use AI for debugging but not for writing whole solutions. Does that cover you rewriting my pragma?",
    privacy_leak="""here's how I run it on the node:
ssh fkhan@node07.hpc.example.edu
export OMP_NUM_THREADS=8 && ./mm 2048""",
    redact="(I left out my username and the node hostname on purpose)",
    bias_claim="For best results compile with the Intel compiler and -qopenmp.",
    bias_mid="Isn't that Intel-specific?",
    bias_high="that advice assumes the Intel compiler; we only have gcc here, so I'd rather keep the fix portable and ignore the compiler switch",
    stake_mid="I'll try not to hog the node",
    stake_high="I'll run the 16-thread timing on a compute node through the scheduler, not the login node, since other students are using it tonight",
    attr_mid="I'll say I used AI in the lab notes",
    attr_high="I'll note in my lab notes that the private clause came from this chat and that the timing analysis is mine",
    team_mid="my partner is doing the timing part",
    team_high="my partner's timing harness calls this with a flat array, so I'll check the 2D indexing matches his layout before we combine",
)

TASKS["cuda_tiled"] = dict(
    title="CUDA tiled matrix multiplication kernel",
    short="CUDA tiled matmul", api="CUDA", lang="CUDA C",
    code="""__global__ void matmul(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < N / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    C[row * N + col] = acc;
}""",
    symptom="it is wrong for N=1000 and occasionally wrong even for N=1024 with TILE=16",
    vague_symptom="the gpu version doesnt match",
    config="nvcc 12.2 -O3 -arch=sm_80, one A100",
    goal="it must match cuBLAS sgemm to 1e-3 relative error for any N up to 4096, including N not a multiple of TILE",
    format_c="keep TILE=16 and shared-memory tiling",
    success="so I can compare it against cuBLAS in my report",
    diag="Two problems. There is no __syncthreads() between loading the tiles and using them, or before the next iteration overwrites them, so threads read half-loaded tiles. And N / TILE drops the last partial tile, with no bounds guard on the loads or the store, so N=1000 is wrong. Loop to (N + TILE - 1) / TILE, zero-pad out-of-range loads, add a barrier after loading and after the inner product, and guard the store.",
    fix="""for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
    int ac = t * TILE + threadIdx.x, br = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < N && ac < N) ? A[row * N + ac] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (br < N && col < N) ? B[br * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
}
if (row < N && col < N) C[row * N + col] = acc;""",
    flaw="One __syncthreads() after loading the tiles is enough.",
    flaw_rebuttal="without a second barrier after the inner product, a fast warp can start loading the next tile while a slow one is still reading the current one. compute-sanitizer racecheck flags exactly that",
    flaw_ack="Yes, you need both barriers; the second one protects the tile from being overwritten while it's still being read.",
    mech="Without __syncthreads(), some threads start the inner product before other threads in the block have finished writing their elements of the shared tile, so they read stale or partially loaded data.",
    close_words="without __syncthreads some threads start the inner product before others have finished writing the shared tile, so they read partly loaded data",
    own_words="the barrier stops a fast warp from reading a half-loaded tile, and the second barrier stops it from overwriting a tile someone else is still using",
    loose_words="the gpu threads go out of sync",
    subparts=["checking the index math", "checking shared-memory synchronization", "handling N that isn't a multiple of TILE", "comparing against cuBLAS"],
    first_part="the synchronization", later_part="the partial tiles",
    dependency="Before I optimize the tile size I need it correct against cuBLAS, otherwise I can't tell a speedup from a wrong answer",
    timings="N=4096: my kernel 41 ms, cuBLAS sgemm 9 ms",
    keep="I'll handle the benchmarking against cuBLAS myself; you check the boundary conditions",
    reference="cuBLAS sgemm on the same inputs", ref_result="max relative error is 2e-6 for N=1000, 1024 and 4095",
    spot="ran N=1000 once and it matched",
    edge="N=1, 15, 16, 1000, 1024 and 4095",
    error_specific="for N=1000 the last 8 columns of C are wrong, because the loop runs N / TILE = 62 tiles and skips the partial one",
    dubious="__syncwarp_all(), which synchronizes the whole block",
    dubious_catch="__syncwarp_all isn't in the CUDA programming guide; __syncwarp only covers one warp, so it can't replace __syncthreads here",
    dubious_ack="You're right, __syncwarp_all doesn't exist. __syncthreads() is the block-level barrier you need.",
    prov_q="why a warp-level sync isn't enough here, and where the CUDA guide says so",
    prov_a="A tile is shared by all warps in the block, so you need a block-wide barrier; the CUDA C++ Programming Guide's section on __syncthreads describes this.",
    profiler="Nsight Compute shows 85% of peak DRAM bandwidth and low SM utilization",
    model_claim="The kernel is compute-bound now, so unrolling the inner loop will help most.",
    reconcile="Nsight Compute shows it near peak DRAM bandwidth with low SM utilization, which means memory-bound, not compute-bound; I'll try a bigger tile instead of unrolling",
    why=[("why do we need the second __syncthreads?", "It stops the next iteration's loads from overwriting a tile other threads are still reading."),
         ("how does zero-padding make the partial tile correct?", "Out-of-range elements contribute 0 to every dot product, so the result for in-range elements is unchanged."),
         ("then why is cuBLAS still 4x faster?", "It uses register tiling, vectorized loads and tensor cores; a shared-memory-only kernel is still limited by memory traffic.")],
    policy="the assignment says we have to write the kernel ourselves. Is it OK if I use your version of the loop, or should I only use your explanation?",
    privacy_leak="""I'm running it like this:
ssh fkhan@gpu02.hpc.example.edu
sbatch --account=cmpe-gpu-fkhan run.sh""",
    redact="(I replaced the account and hostname with [redacted])",
    bias_claim="Just rewrite it with CUDA-specific intrinsics; nothing else matters for GPUs.",
    bias_mid="That seems NVIDIA-specific.",
    bias_high="that's NVIDIA-only advice; our course also runs on AMD nodes, so I'd like to know whether the fix carries over to HIP or SYCL before I lock into intrinsics",
    stake_mid="I'll keep the GPU jobs small",
    stake_high="there are only two GPUs for the whole class, so I'll test on N=1024 first and only queue the N=4096 sweep once it's correct",
    attr_mid="I'll cite the AI use",
    attr_high="in the report I'll say the boundary-guard code came from this chat and write the tiling explanation in my own words",
    team_mid="this kernel is my part of the group project",
    team_high="my teammate's host code launches with 32x32 blocks but my kernel assumes TILE=16, so I'll reconcile the launch configuration with him before we merge",
)

TASKS["mpi_halo"] = dict(
    title="MPI halo exchange that deadlocks",
    short="MPI halo exchange", api="MPI", lang="C with MPI",
    code="""/* 1D Jacobi, each rank owns rows [1..local_n], ghosts at 0 and local_n+1 */
MPI_Send(&u[local_n][0], NX, MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
MPI_Recv(&u[0][0], NX, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Send(&u[1][0], NX, MPI_DOUBLE, left, 1, MPI_COMM_WORLD);
MPI_Recv(&u[local_n+1][0], NX, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);""",
    symptom="it runs for NX=512 but hangs at the first iteration for NX=8192",
    vague_symptom="it just freezes",
    config="mpicc -O2, MPICH 4.1, 8 ranks on 1 node",
    goal="it must not hang for any NX up to 16384, and the grid after 100 iterations must match the serial Jacobi code",
    format_c="keep the 1D row decomposition",
    success="so I can explain the deadlock in the lab writeup",
    diag="Every rank calls a blocking MPI_Send before its MPI_Recv. For small messages the library buffers them (the eager protocol), so it works; for large messages MPI_Send waits for the matching receive, and since every rank is sending, nobody receives: a deadlock. Use MPI_Sendrecv, or MPI_Isend/MPI_Irecv followed by MPI_Waitall.",
    fix="""MPI_Sendrecv(&u[local_n][0], NX, MPI_DOUBLE, right, 0,
             &u[0][0],       NX, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Sendrecv(&u[1][0],       NX, MPI_DOUBLE, left,  1,
             &u[local_n+1][0], NX, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);""",
    flaw="Raise the eager limit (for example MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE) so the large messages are buffered too.",
    flaw_rebuttal="that just hides the bug. The standard allows MPI_Send to block until a matching receive, so the code is still incorrect and will hang again on a different MPI or bigger NX",
    flaw_ack="Fair point, raising the eager limit only masks the unsafe ordering. MPI_Sendrecv fixes the actual problem.",
    mech="Each rank does a blocking send before its receive; small messages are buffered by the eager protocol, but large ones wait for the matching receive, and because every rank is sending first nobody ever receives.",
    close_words="each rank does a blocking send before its receive, and for large messages the send waits for a receive that never starts because every rank is also sending",
    own_words="it's a circle of ranks each holding the door for the next one; small messages slip through a buffer, big ones don't, so everyone waits forever. Sendrecv lets each rank send and receive at the same time",
    loose_words="mpi gets stuck",
    subparts=["finding where it hangs", "understanding why small NX works", "making the exchange safe", "checking the grid against the serial code"],
    first_part="the deadlock", later_part="overlapping communication with compute",
    dependency="I need the exchange to be deadlock-free before I try overlapping it with computation, because overlap only adds more ways for it to hang",
    timings="NX=8192, 100 iterations: serial 12.4 s; 8 ranks with Sendrecv 1.9 s",
    keep="I'll write the explanation of eager vs rendezvous myself; you check that my tags and neighbors are right at the boundaries",
    reference="the serial Jacobi code", ref_result="the grid matches to 1e-12 after 100 iterations for NX=512 and NX=8192",
    spot="ran it once with NX=8192 and it didn't hang",
    edge="1, 2 and 8 ranks, NX=512 and 16384, with MPI_PROC_NULL at the ends",
    error_specific="rank 0 has left = -1 in my code, and your version sends to it anyway; it should be MPI_PROC_NULL",
    dubious="MPI_Exchange_halo(), the standard routine for halo exchanges",
    dubious_catch="MPI_Exchange_halo isn't in the MPI standard or the MPICH docs; neighborhood collectives exist, but not under that name",
    dubious_ack="You're right, there's no MPI_Exchange_halo; I was wrong about that. MPI_Sendrecv or MPI_Neighbor_alltoall are the real options.",
    prov_q="where the standard says MPI_Send is allowed to block",
    prov_a="The MPI standard's section on communication modes says a standard-mode send may either buffer or wait for the matching receive; the choice is up to the implementation.",
    profiler="gdb attached to the hung job shows every rank sitting inside MPI_Send",
    model_claim="The hang is probably caused by mismatched tags.",
    reconcile="gdb shows every rank stuck in MPI_Send, and the tags do match, so it's the send-before-receive ordering, not the tags",
    why=[("why does it work for NX=512 but not 8192?", "Small messages fit under the eager threshold and are buffered; large ones switch to rendezvous and wait for the receive."),
         ("how does Sendrecv avoid the deadlock?", "It posts the send and receive together, so the library can progress both without an ordering dependency."),
         ("would nonblocking calls let me overlap the exchange with the interior update?", "Yes: post Irecv/Isend, update the interior rows that don't need ghosts, then Waitall and update the boundary rows.")],
    policy="is it OK under the course rules to use your Sendrecv version directly, or should I only use the explanation?",
    privacy_leak="""run command:
mpirun -np 8 ./jacobi 8192
on login node fkhan@hpc-login.example.edu, account cmpe4337-fkhan""",
    redact="(hostname and account removed)",
    bias_claim="Nonblocking communication is always faster than Sendrecv.",
    bias_mid="Is that always true?",
    bias_high="that's a general claim; with one exchange per iteration and nothing to overlap yet, I doubt it holds here, so I'll time both before I change anything",
    stake_mid="I'll be careful not to leave hung jobs",
    stake_high="a hung job holds a full node until the time limit and blocks classmates, so I'll set a 5-minute limit on every test run",
    attr_mid="I'll note that I used AI",
    attr_high="the writeup will say the Sendrecv rewrite was suggested here and that the deadlock explanation is my own",
    team_mid="this is the communication part of our team's solver",
    team_high="my teammate's I/O code assumes the ghost rows are at index 0 and local_n+1, so I'll check the new exchange still fills those before we merge",
)

TASKS["omp_false_sharing"] = dict(
    title="OpenMP partial sums with false sharing",
    short="OpenMP partial sums", api="OpenMP", lang="C with OpenMP",
    code="""double partial[MAX_THREADS];
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    partial[tid] = 0.0;
    #pragma omp for
    for (long i = 0; i < n; i++) partial[tid] += x[i] * y[i];
}
for (int t = 0; t < nthreads; t++) dot += partial[t];""",
    symptom="the dot product is correct but 8 threads are barely faster than 1",
    vague_symptom="it doesnt get faster",
    config="gcc 12 -O2 -fopenmp, 16-core node, n = 200 million",
    goal="the result must stay correct and I need at least 5x speedup on 8 threads",
    format_c="keep the OpenMP pragmas and plain C",
    success="so I can explain the scaling in my report",
    diag="The partial[] entries for different threads sit next to each other in the same cache line, so every update by one thread invalidates the line in the other cores' caches (false sharing). Accumulate into a private local variable and write it once, or use reduction(+:dot); padding each entry to 64 bytes also works.",
    fix="""#pragma omp parallel for reduction(+:dot)
for (long i = 0; i < n; i++) dot += x[i] * y[i];""",
    flaw="The slowdown is load imbalance; switch to schedule(dynamic).",
    flaw_rebuttal="every iteration does the same work, so there's no imbalance, and dynamic scheduling adds overhead. perf c2c shows heavy HITM traffic on the partial array, which points at false sharing",
    flaw_ack="You're right, the iterations are uniform, so it isn't load imbalance; the HITM traffic confirms false sharing.",
    mech="The partial sums of different threads share a cache line, so each write by one thread invalidates the copies in other cores' caches and the line bounces between cores.",
    close_words="the partial sums of different threads share a cache line, so each write invalidates the other cores' copies and the line bounces between them",
    own_words="the threads never touch each other's numbers, but the hardware moves memory in 64-byte chunks, so neighbours keep stealing the same chunk from each other. Keeping each sum in a register until the end stops the fight",
    loose_words="the cache is slow",
    subparts=["confirming the result is correct", "finding why it doesn't scale", "fixing the memory layout", "re-measuring speedup"],
    first_part="the diagnosis", later_part="the re-measurement",
    dependency="I need to confirm the cause before changing the code, because otherwise any speedup could be luck",
    timings="n=200M: 1 thread 0.42 s, 8 threads 0.31 s",
    keep="I'll do the timing and the explanation in the report; you check that the reduction version is equivalent",
    reference="the serial dot product", ref_result="the result matches to 1e-9 relative, and 8 threads now take 0.07 s",
    spot="ran it once and it's faster now",
    edge="1, 2, 4, 8 and 16 threads, n=1000 and n=200M",
    error_specific="padding to 8 bytes doesn't help because a cache line here is 64 bytes, so the entries still share a line",
    dubious="the OMP_CACHE_ALIGN environment variable, which pads arrays automatically",
    dubious_catch="OMP_CACHE_ALIGN isn't in the OpenMP 5.2 spec's list of environment variables, so I don't think it exists",
    dubious_ack="Correct, OMP_CACHE_ALIGN isn't a real OpenMP variable. Use a reduction or pad the array yourself.",
    prov_q="how you know the cache line is 64 bytes on our machine",
    prov_a="It's 64 bytes on most x86 CPUs; you can confirm it with getconf LEVEL1_DCACHE_LINESIZE or lscpu on your node.",
    profiler="perf c2c reports most HITM events on the partial array",
    model_claim="Memory bandwidth is the main limit here.",
    reconcile="perf c2c puts most HITM events on the partial array, which is false sharing, not bandwidth; after the reduction it scales to 6x, so bandwidth wasn't the limit yet",
    why=[("why does it slow down when the threads never touch the same element?", "Coherence works on cache lines, not elements, so neighbouring elements still conflict."),
         ("how does the reduction clause avoid it?", "Each thread accumulates into a private copy, usually a register, and the copies are combined once at the end."),
         ("why does speedup stop around 6x on 8 threads?", "Once false sharing is gone, the dot product is limited by memory bandwidth, since it does only two flops per 16 bytes loaded.")],
    policy="the lab says 'no AI-generated code in submissions.' Does using a reduction clause you suggested count?",
    privacy_leak="""I ran it on node17 as fkhan (fkhan@node17.hpc.example.edu), job 4411823""",
    redact="(I'm leaving out my username and node name)",
    bias_claim="Always use schedule(dynamic) for OpenMP loops; it's the modern default.",
    bias_mid="Is dynamic really always better?",
    bias_high="that's generic advice from a different kind of loop; for uniform iterations static is standard, so I'll only change the schedule if timing shows a difference",
    stake_mid="I'll try not to overload the node",
    stake_high="I'll run the 16-thread tests on an exclusive node through the scheduler so I don't slow down other students' jobs sharing it",
    attr_mid="I'll disclose the AI use",
    attr_high="I'll write the reduction myself from the explanation and note in the report that the false-sharing diagnosis came from this chat",
    team_mid="my part of the group lab is this kernel",
    team_high="my teammate's version pads partial[] to 64 bytes; I'll compare it against the reduction and we'll pick one before merging",
)

TASKS["scaling"] = dict(
    title="Strong-scaling analysis of an OpenMP matrix multiplication",
    short="scaling analysis", api="OpenMP", lang="C with OpenMP",
    code="""#pragma omp parallel for
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int k = 0; k < N; k++) s += A[i*N+k] * B[k*N+j];
        C[i*N+j] = s;
    }""",
    symptom="speedup reaches 5x at 8 threads and then stays flat up to 16 threads",
    vague_symptom="it stops getting faster",
    config="gcc 12 -O3 -fopenmp, 16-core node, N=4096",
    goal="I need to explain the plateau with evidence, not guesses, and get past 8x on 16 threads if possible",
    format_c="keep it plain C with OpenMP",
    success="so I can write the scaling section of my report",
    diag="The inner loop reads B[k*N+j] with stride N, so almost every access misses in cache and the kernel becomes memory-bandwidth bound; once 8 threads saturate DRAM bandwidth, more threads don't help. Reorder the loops to i-k-j so B is read contiguously, or block the loops for cache.",
    fix="""#pragma omp parallel for
for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++) {
        double a = A[i*N+k];
        for (int j = 0; j < N; j++) C[i*N+j] += a * B[k*N+j];
    }""",
    flaw="By Amdahl's law your serial fraction is about 12%, which explains the plateau.",
    flaw_rebuttal="there's almost no serial code outside the parallel loop, and I measured it at under 1% of runtime. The plateau lines up with STREAM bandwidth instead",
    flaw_ack="You're right, I inferred the serial fraction without measuring it. With under 1% serial time, the plateau is bandwidth, not Amdahl.",
    mech="The k-loop reads B with a stride of N, so nearly every access misses cache; the kernel is limited by memory bandwidth, and once the threads saturate it, adding threads doesn't increase throughput.",
    close_words="the k-loop reads B with stride N, so almost every access misses and the kernel is limited by memory bandwidth, which saturates around 8 threads",
    own_words="the cores are waiting on memory, not on arithmetic; after about 8 threads the memory bus is full, so extra cores just queue up. Walking B row by row lets every cache line do more work",
    loose_words="the computer is maxed out",
    subparts=["measuring the serial fraction", "measuring memory bandwidth", "comparing loop orders", "writing up the scaling plot"],
    first_part="the measurements", later_part="the write-up",
    dependency="I need the bandwidth numbers before changing the loop order, because otherwise I can't say why it helped",
    timings="N=4096: 1 thread 612 s, 8 threads 121 s, 16 threads 118 s; STREAM triad 38 GB/s at 8 threads, 39 GB/s at 16",
    keep="I'll make the plots and write the analysis myself; you help me check the reasoning about bandwidth",
    reference="the STREAM triad numbers and likwid-perfctr's memory bandwidth group", ref_result="the kernel sits at 36 GB/s at 8 threads, right at the STREAM limit",
    spot="timed it once at 16 threads and it's faster after the reorder",
    edge="1, 2, 4, 8 and 16 threads, N=1024 and N=4096, both loop orders",
    error_specific="your 12% serial fraction would cap speedup at about 5.7x on 16 threads, but the i-k-j version reaches 11x, so Amdahl can't be the explanation",
    dubious="the 'Gustafson ceiling' formula S = 1/(1-p)^2",
    dubious_catch="that isn't Gustafson's law. Gustafson's law is S = N - s(N - 1), and I can't find the squared version anywhere",
    dubious_ack="You're right, that formula was wrong; Gustafson's law is S = N - s(N - 1), and it applies to weak scaling anyway.",
    prov_q="where the claim that 8 threads saturate bandwidth comes from, since you haven't seen our node",
    prov_a="I don't know your node's numbers; the claim should be checked by running STREAM or likwid-bench on it and comparing with the kernel's measured bandwidth.",
    profiler="likwid-perfctr shows 36 GB/s memory bandwidth at 8 threads",
    model_claim="The plateau is caused by thread creation overhead.",
    reconcile="likwid shows the kernel at 36 GB/s, basically the STREAM limit, and thread startup is milliseconds against a 100-second run, so it's bandwidth, not overhead",
    why=[("why does reading B by columns cost so much?", "Each 64-byte line brought in for B[k*N+j] supplies one useful double before it's evicted."),
         ("how does i-k-j fix that?", "The innermost loop walks B and C along rows, so each cache line supplies eight useful doubles."),
         ("would blocking help beyond the reorder?", "Yes, tiling keeps blocks of A, B and C in cache, which raises arithmetic intensity further.")],
    policy="can I use your explanation in the report if I rewrite it, or does the course policy count that as AI-written text?",
    privacy_leak="""results are in /home/fkhan/cmpe4337/scaling on hpc-login.example.edu, and my email is fkhan@example.edu if that matters""",
    redact="(I've left out my paths and username)",
    bias_claim="Most HPC experts agree that Amdahl's law explains nearly all scaling limits.",
    bias_mid="Does that apply here?",
    bias_high="that's an appeal to what 'experts' say, not evidence; for this kernel the measured bandwidth says otherwise, so I'll go with the measurement",
    stake_mid="I'll keep the runs reasonable",
    stake_high="the full sweep is 30 node-hours of our shared class allocation, so I'll run N=1024 first and only do the N=4096 sweep for the two loop orders I'll actually report",
    attr_mid="I'll mention AI in the report",
    attr_high="the report will say this chat helped me interpret the bandwidth numbers; the measurements, plots and conclusions are mine",
    team_mid="I'm doing the scaling part for our group",
    team_high="my teammate ran the MPI version on two nodes, so I'll line up our thread and rank counts in one plot and we'll agree on how to compare them",
)
