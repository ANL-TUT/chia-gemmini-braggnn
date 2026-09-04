#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "braggnn.h"
#include "include/gemmini_testutils.h"

#define NLB_STAGE1_STORE_SCALE (127.0f / 22805.0f)
#define NLB_STAGE2_BERT_SCALE (NLB_MATMUL_BERT_SCALE / NLB_STAGE1_STORE_SCALE)

// ------------------------------------------------------------------
// Hardware performance counters (see braggnn_tune.c for the full
// rationale -- this mirrors that file's per-kernel counter plumbing).
//
// Only 8 hardware counter slots exist, so the three families below
// (DMA/wait, EXE/control-overhead, MAIN controller-overlap) cannot be
// observed simultaneously -- each needs its own pass over the whole
// benchmark, driven by the "active pass" (g_pass_mode) below.
//
// Within a pass, instead of measuring the whole gemmini_inference()
// call as one span (which only tells us the total for the entire
// network), MEASURE_KERNEL() resets/reconfigures the counters around
// *each individual kernel call* and accumulates the deltas into a
// per-kernel table. Nothing is printed per kernel while the network
// runs; the whole per-kernel table is printed once, after the whole
// inference finishes.
// ------------------------------------------------------------------

// Counters 0/1: cycles the read/write DMA engines spend actively moving
// data between DRAM and the scratchpad/accumulator (mvin/mvout).
// Counters 2-7: cycles the systolic array's A/B/D operand feed is stalled
// waiting for data to become available in the scratchpad or accumulator,
// i.e. compute-side stalls caused by DMA/spad not being ready in time.
typedef struct {
  unsigned long long rdma;
  unsigned long long wdma;
  unsigned long long spad_a_wait;
  unsigned long long spad_b_wait;
  unsigned long long spad_d_wait;
  unsigned long long acc_a_wait;
  unsigned long long acc_b_wait;
  unsigned long long acc_d_wait;
} dma_stats_t;

static void configure_dma_counters() {
  counter_reset();
  counter_configure(0, RDMA_ACTIVE_CYCLE);
  counter_configure(1, WDMA_ACTIVE_CYCLE);
  counter_configure(2, SCRATCHPAD_A_WAIT_CYCLE);
  counter_configure(3, SCRATCHPAD_B_WAIT_CYCLE);
  counter_configure(4, SCRATCHPAD_D_WAIT_CYCLE);
  counter_configure(5, ACC_A_WAIT_CYCLE);
  counter_configure(6, ACC_B_WAIT_CYCLE);
  counter_configure(7, ACC_D_WAIT_CYCLE);
}

static dma_stats_t read_dma_counters() {
  dma_stats_t s;
  s.rdma = counter_read(0);
  s.wdma = counter_read(1);
  s.spad_a_wait = counter_read(2);
  s.spad_b_wait = counter_read(3);
  s.spad_d_wait = counter_read(4);
  s.acc_a_wait = counter_read(5);
  s.acc_b_wait = counter_read(6);
  s.acc_d_wait = counter_read(7);
  return s;
}

static void print_dma_stats(const dma_stats_t *s) {
  printf("dram<->spad DMA cycles: rdma=%llu, wdma=%llu\n", s->rdma, s->wdma);
  printf("compute stall waiting on spad: A=%llu, B=%llu, D=%llu\n",
         s->spad_a_wait, s->spad_b_wait, s->spad_d_wait);
  printf("compute stall waiting on acc:  A=%llu, B=%llu, D=%llu\n",
         s->acc_a_wait, s->acc_b_wait, s->acc_d_wait);
}

// Second set of 8 counters: cycles spent in the systolic array itself
// (EXE_ACTIVE_CYCLE) vs. execution-side overhead (weight-reload flushes,
// overlap hazards) and instruction-issue overhead (reservation station).
// LOOP_MATMUL_ACTIVE_CYCLES / LOOP_CONV_ACTIVE_CYCLES are the total time
// the CISC LOOP_WS / LOOP_CONV_WS unrollers are busy (from accepting the
// macro-instruction through draining every micro-op it generates), so
// together they show the matmul-vs-conv split of that overhead -- unlike
// exe_active, which doesn't distinguish which unroller a compute cycle
// came from. EXE_PRELOAD_HAZ_CYCLE was dropped from this set: it only
// fires in OS dataflow (BraggNN is WS-only) and always reads 0 here.
// LOOP_CONV_ACTIVE_CYCLES requires the LOOP_CONV_ACTIVE_CYCLES RTL counter
// added to Controller.scala/CounterFile.scala -- rebuild Gemmini
// (make CONFIG=GemminiRocketConfig) before using this.
typedef struct {
  unsigned long long exe_active;
  unsigned long long exe_control_q_block;
  unsigned long long exe_flush;
  unsigned long long exe_overlap_haz;
  unsigned long long rs_full;
  unsigned long long rs_active;
  unsigned long long loop_matmul_active;
  unsigned long long loop_conv_active;
} exe_stats_t;

static void configure_exe_counters() {
  counter_reset();
  counter_configure(0, EXE_ACTIVE_CYCLE);
  counter_configure(1, EXE_CONTROL_Q_BLOCK_CYCLE);
  counter_configure(2, EXE_FLUSH_CYCLE);
  counter_configure(3, EXE_OVERLAP_HAZ_CYCLE);
  counter_configure(4, RESERVATION_STATION_FULL_CYCLES);
  counter_configure(5, RESERVATION_STATION_ACTIVE_CYCLES);
  counter_configure(6, LOOP_MATMUL_ACTIVE_CYCLES);
  counter_configure(7, LOOP_CONV_ACTIVE_CYCLES);
}

static exe_stats_t read_exe_counters() {
  exe_stats_t s;
  s.exe_active = counter_read(0);
  s.exe_control_q_block = counter_read(1);
  s.exe_flush = counter_read(2);
  s.exe_overlap_haz = counter_read(3);
  s.rs_full = counter_read(4);
  s.rs_active = counter_read(5);
  s.loop_matmul_active = counter_read(6);
  s.loop_conv_active = counter_read(7);
  return s;
}

static void print_exe_stats(const exe_stats_t *s) {
  printf("exe: active=%llu, control_q_block=%llu, flush=%llu, "
         "overlap_haz=%llu\n",
         s->exe_active, s->exe_control_q_block, s->exe_flush,
         s->exe_overlap_haz);
  printf("reservation station: full=%llu, active=%llu\n", s->rs_full,
         s->rs_active);
  printf("loop_matmul active=%llu, loop_conv active=%llu\n",
         s->loop_matmul_active, s->loop_conv_active);
}

// Third set of 8 counters: the MAIN_* family partitions every cycle by
// which subset of {Load, Store, Ex} controllers is busy (mutually
// exclusive by construction). Together they directly answer whether DMA
// (Load/Store) and compute (Ex) actually overlap in this workload, and,
// by subtracting their sum from total cycles, give an unambiguous "none
// of ld/st/ex busy" idle figure -- unlike RESERVATION_STATION_FULL_CYCLES,
// which conflates genuine backpressure with simply having no new command
// to allocate. EXE_ACTIVE_CYCLE is included again here (not just in pass
// 2) so it can be compared against "any cycle where ex is busy" to isolate
// time the ex controller spends busy but not actually in its compute
// state (e.g. preload/config overhead).
typedef struct {
  unsigned long long ld_only;
  unsigned long long st_only;
  unsigned long long ex_only;
  unsigned long long ld_st;
  unsigned long long ld_ex;
  unsigned long long st_ex;
  unsigned long long ld_st_ex;
  unsigned long long exe_active;
} main_stats_t;

static void configure_main_counters() {
  counter_reset();
  counter_configure(0, MAIN_LD_CYCLES);
  counter_configure(1, MAIN_ST_CYCLES);
  counter_configure(2, MAIN_EX_CYCLES);
  counter_configure(3, MAIN_LD_ST_CYCLES);
  counter_configure(4, MAIN_LD_EX_CYCLES);
  counter_configure(5, MAIN_ST_EX_CYCLES);
  counter_configure(6, MAIN_LD_ST_EX_CYCLES);
  counter_configure(7, EXE_ACTIVE_CYCLE);
}

static main_stats_t read_main_counters() {
  main_stats_t s;
  s.ld_only = counter_read(0);
  s.st_only = counter_read(1);
  s.ex_only = counter_read(2);
  s.ld_st = counter_read(3);
  s.ld_ex = counter_read(4);
  s.st_ex = counter_read(5);
  s.ld_st_ex = counter_read(6);
  s.exe_active = counter_read(7);
  return s;
}

static void print_main_stats(const main_stats_t *s, unsigned long long total) {
  unsigned long long any_busy = s->ld_only + s->st_only + s->ex_only +
                                 s->ld_st + s->ld_ex + s->st_ex + s->ld_st_ex;
  unsigned long long none_busy = total > any_busy ? total - any_busy : 0;
  unsigned long long ex_busy_any =
      s->ex_only + s->ld_ex + s->st_ex + s->ld_st_ex;
  unsigned long long ex_busy_not_compute =
      ex_busy_any > s->exe_active ? ex_busy_any - s->exe_active : 0;

  printf("controller busy: ld_only=%llu, st_only=%llu, ex_only=%llu\n",
         s->ld_only, s->st_only, s->ex_only);
  printf("controller busy: ld+st=%llu, ld+ex=%llu, st+ex=%llu, "
         "ld+st+ex=%llu\n",
         s->ld_st, s->ld_ex, s->st_ex, s->ld_st_ex);
  printf("none of ld/st/ex busy (true idle): %llu\n", none_busy);
  printf("ex busy but not in compute state (preload/config overhead): "
         "%llu\n",
         ex_busy_not_compute);
}

// ------------------------------------------------------------------
// Per-kernel measurement plumbing
// ------------------------------------------------------------------

typedef enum { PASS_DMA, PASS_EXE, PASS_MAIN } pass_mode_t;

// The 3-stage NLB splits the attention matmul+softmax into two steps
// (see NLB_STAGE1_STORE_SCALE/NLB_STAGE2_BERT_SCALE above): a raw
// theta@phi matmul with NO_ACTIVATION (K_ATTN_LOGITS), then a separate
// identity-matmul that applies SOFTMAX to the already-int8 logits
// (K_ATTN_SOFTMAX) -- unlike braggnn_tune_pt2e.c's fused single matmul.
typedef enum {
  K_CONV1 = 0,
  K_THETA,
  K_PHI,
  K_NLB_G,
  K_ATTN_LOGITS,
  K_ATTN_SOFTMAX,
  K_ATTENDED_MATMUL,
  K_NLB_OUT_CONV,
  K_RESADD,
  K_CONV2,
  K_CONV3,
  K_FC1,
  K_FC2,
  K_FC3,
  K_FC4,
  K_OUTPUT,
  NUM_KERNELS
} kernel_id_t;

static const char *kernel_names[NUM_KERNELS] = {
  "conv1", "theta", "phi", "nlb_g", "attn_logits", "attn_softmax",
  "attended_matmul", "nlb_out_conv", "resadd", "conv2", "conv3",
  "fc1", "fc2", "fc3", "fc4", "output",
};

static pass_mode_t g_pass_mode = PASS_DMA;
static dma_stats_t g_dma_kernel[NUM_KERNELS];
static exe_stats_t g_exe_kernel[NUM_KERNELS];
static main_stats_t g_main_kernel[NUM_KERNELS];
static unsigned long long g_kernel_cycles[NUM_KERNELS];

static void reset_dma_kernel_table(dma_stats_t t[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) t[k] = (dma_stats_t){0};
}
static void reset_exe_kernel_table(exe_stats_t t[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) t[k] = (exe_stats_t){0};
}
static void reset_main_kernel_table(main_stats_t t[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) t[k] = (main_stats_t){0};
}

// Wall-clock cycles spent inside each kernel (tracked in every pass,
// regardless of which counter family is active), so that per-kernel
// none_busy/ex_busy_not_compute in Pass 3 are computed against that
// kernel's own cycle count instead of the whole inference's.
static void reset_kernel_cycles(unsigned long long t[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) t[k] = 0;
}
static void add_kernel_cycles(unsigned long long dst[NUM_KERNELS], const unsigned long long src[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) dst[k] += src[k];
}
static void div_kernel_cycles(unsigned long long t[NUM_KERNELS], unsigned long long n) {
  for (int k = 0; k < NUM_KERNELS; k++) t[k] /= n;
}
static void print_kernel_cycles(const unsigned long long t[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    printf("-- kernel: %s: %llu cycles --\n", kernel_names[k], t[k]);
  }
}

static void add_dma_kernel_table(dma_stats_t dst[NUM_KERNELS], const dma_stats_t src[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    dst[k].rdma += src[k].rdma;
    dst[k].wdma += src[k].wdma;
    dst[k].spad_a_wait += src[k].spad_a_wait;
    dst[k].spad_b_wait += src[k].spad_b_wait;
    dst[k].spad_d_wait += src[k].spad_d_wait;
    dst[k].acc_a_wait += src[k].acc_a_wait;
    dst[k].acc_b_wait += src[k].acc_b_wait;
    dst[k].acc_d_wait += src[k].acc_d_wait;
  }
}
static void add_exe_kernel_table(exe_stats_t dst[NUM_KERNELS], const exe_stats_t src[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    dst[k].exe_active += src[k].exe_active;
    dst[k].exe_control_q_block += src[k].exe_control_q_block;
    dst[k].exe_flush += src[k].exe_flush;
    dst[k].exe_overlap_haz += src[k].exe_overlap_haz;
    dst[k].rs_full += src[k].rs_full;
    dst[k].rs_active += src[k].rs_active;
    dst[k].loop_matmul_active += src[k].loop_matmul_active;
    dst[k].loop_conv_active += src[k].loop_conv_active;
  }
}
static void add_main_kernel_table(main_stats_t dst[NUM_KERNELS], const main_stats_t src[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    dst[k].ld_only += src[k].ld_only;
    dst[k].st_only += src[k].st_only;
    dst[k].ex_only += src[k].ex_only;
    dst[k].ld_st += src[k].ld_st;
    dst[k].ld_ex += src[k].ld_ex;
    dst[k].st_ex += src[k].st_ex;
    dst[k].ld_st_ex += src[k].ld_st_ex;
    dst[k].exe_active += src[k].exe_active;
  }
}

static void div_dma_kernel_table(dma_stats_t t[NUM_KERNELS], unsigned long long n) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    t[k].rdma /= n; t[k].wdma /= n;
    t[k].spad_a_wait /= n; t[k].spad_b_wait /= n; t[k].spad_d_wait /= n;
    t[k].acc_a_wait /= n; t[k].acc_b_wait /= n; t[k].acc_d_wait /= n;
  }
}
static void div_exe_kernel_table(exe_stats_t t[NUM_KERNELS], unsigned long long n) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    t[k].exe_active /= n; t[k].exe_control_q_block /= n; t[k].exe_flush /= n;
    t[k].exe_overlap_haz /= n; t[k].rs_full /= n; t[k].rs_active /= n;
    t[k].loop_matmul_active /= n; t[k].loop_conv_active /= n;
  }
}
static void div_main_kernel_table(main_stats_t t[NUM_KERNELS], unsigned long long n) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    t[k].ld_only /= n; t[k].st_only /= n; t[k].ex_only /= n;
    t[k].ld_st /= n; t[k].ld_ex /= n; t[k].st_ex /= n; t[k].ld_st_ex /= n;
    t[k].exe_active /= n;
  }
}

static void print_dma_kernel_table(const dma_stats_t t[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    printf("-- kernel: %s --\n", kernel_names[k]);
    print_dma_stats(&t[k]);
  }
}
static void print_exe_kernel_table(const exe_stats_t t[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    printf("-- kernel: %s --\n", kernel_names[k]);
    print_exe_stats(&t[k]);
  }
}
// total_per_kernel: that kernel's own cycle count (from g_kernel_cycles),
// not the whole inference's -- otherwise none_busy/ex_busy_not_compute
// would be computed against the wrong denominator.
static void print_main_kernel_table(const main_stats_t t[NUM_KERNELS], const unsigned long long total_per_kernel[NUM_KERNELS]) {
  for (int k = 0; k < NUM_KERNELS; k++) {
    printf("-- kernel: %s --\n", kernel_names[k]);
    print_main_stats(&t[k], total_per_kernel[k]);
  }
}

// Resets/reconfigures whichever counter family the active pass wants,
// right before a single kernel call.
static void measure_kernel_begin() {
  switch (g_pass_mode) {
    case PASS_DMA:  configure_dma_counters();  break;
    case PASS_EXE:  configure_exe_counters();  break;
    case PASS_MAIN: configure_main_counters(); break;
  }
}

// Reads whichever counter family the active pass wants, right after a
// single kernel call, and accumulates the deltas into that kernel's slot.
static void measure_kernel_end(kernel_id_t k) {
  switch (g_pass_mode) {
    case PASS_DMA: {
      dma_stats_t s = read_dma_counters();
      g_dma_kernel[k].rdma += s.rdma;
      g_dma_kernel[k].wdma += s.wdma;
      g_dma_kernel[k].spad_a_wait += s.spad_a_wait;
      g_dma_kernel[k].spad_b_wait += s.spad_b_wait;
      g_dma_kernel[k].spad_d_wait += s.spad_d_wait;
      g_dma_kernel[k].acc_a_wait += s.acc_a_wait;
      g_dma_kernel[k].acc_b_wait += s.acc_b_wait;
      g_dma_kernel[k].acc_d_wait += s.acc_d_wait;
      break;
    }
    case PASS_EXE: {
      exe_stats_t s = read_exe_counters();
      g_exe_kernel[k].exe_active += s.exe_active;
      g_exe_kernel[k].exe_control_q_block += s.exe_control_q_block;
      g_exe_kernel[k].exe_flush += s.exe_flush;
      g_exe_kernel[k].exe_overlap_haz += s.exe_overlap_haz;
      g_exe_kernel[k].rs_full += s.rs_full;
      g_exe_kernel[k].rs_active += s.rs_active;
      g_exe_kernel[k].loop_matmul_active += s.loop_matmul_active;
      g_exe_kernel[k].loop_conv_active += s.loop_conv_active;
      break;
    }
    case PASS_MAIN: {
      main_stats_t s = read_main_counters();
      g_main_kernel[k].ld_only += s.ld_only;
      g_main_kernel[k].st_only += s.st_only;
      g_main_kernel[k].ex_only += s.ex_only;
      g_main_kernel[k].ld_st += s.ld_st;
      g_main_kernel[k].ld_ex += s.ld_ex;
      g_main_kernel[k].st_ex += s.st_ex;
      g_main_kernel[k].ld_st_ex += s.ld_st_ex;
      g_main_kernel[k].exe_active += s.exe_active;
      break;
    }
  }
}

// Wraps a single kernel call: reset/reconfigure counters, run the
// kernel, then fold the delta into that kernel's running total. Nothing
// is printed here -- the caller prints the whole per-kernel table once,
// after the whole inference finishes.
//
// The fence before _k_end is required: Gemmini instructions are enqueued
// asynchronously, so a kernel call (e.g. tiled_conv_auto, which unlike
// tiled_matmul_outer has no internal fence) can return long before Gemmini
// has actually finished executing them. Without waiting here, _k_end (and
// the counter_read()s in measure_kernel_end) sample too early, so
// g_kernel_cycles undercounts the kernel's true hardware time -- which is
// why the wait-cycle counters could otherwise come out bigger than the
// kernel's own reported cycle count.
#define MEASURE_KERNEL(kid, call) \
  do { \
    measure_kernel_begin(); \
    unsigned long long _k_start = read_cycles(); \
    call; \
    gemmini_fence(); \
    unsigned long long _k_end = read_cycles(); \
    g_kernel_cycles[kid] += _k_end - _k_start; \
    measure_kernel_end(kid); \
  } while (0)

// Convolution operation using Gemmini's tiled_conv_auto
// weights_flat: pre-computed flattened weight matrix [patch_size][out_channels]
void gemmini_conv2d(int batch, int in_rows, int in_cols, int in_channels,
                    int out_channels, int kernel_dim, int stride, int padding,
                    elem_t *input, elem_t *weights_flat, acc_t *bias, elem_t *output,
                    bool relu, acc_scale_t acc_scale) {

  int out_rows = (in_rows + 2 * padding - kernel_dim) / stride + 1;
  int out_cols = (in_cols + 2 * padding - kernel_dim) / stride + 1;

  tiled_conv_auto(batch, in_rows, in_cols, in_channels, out_channels, out_rows,
                  out_cols, stride, 1, 1, padding, kernel_dim, false, false,
                  false, false, false, (elem_t *)input, (elem_t *)weights_flat,
                  (acc_t *)bias, (elem_t *)output, relu ? RELU : NO_ACTIVATION,
                  acc_scale, 0, 0, 0, WS);

  // tiled_conv_auto does not have a fence (unlike tiled_matmul_outer);
  // MEASURE_KERNEL() fences after every call, so callers outside of
  // measurement need their own gemmini_fence() before relying on this
  // function's output.
}

// Fully connected layer using Gemmini
void gemmini_fc(int batch, int in_features, int out_features, elem_t *input,
                elem_t *weights, acc_t *bias, elem_t *output, bool relu, acc_scale_t acc_scale) {
  tiled_matmul_auto(
      batch, out_features, in_features, (elem_t *)input, (elem_t *)weights,
      (acc_t *)bias, (elem_t *)output, in_features, in_features, out_features,
      out_features, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY, relu ? RELU : NO_ACTIVATION, acc_scale, 0,
      false, false, true, false, false, 0, WS);
}

// Non-Local Block implementation
void gemmini_nlb(elem_t *input, elem_t *output) {
  // Theta conv: output NHWC [1][9][9][32], memory layout = [81][32]
  static elem_t nlb_theta_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  MEASURE_KERNEL(K_THETA, gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_THETA_KERNEL, 1, 0, (elem_t *)input,
                 (elem_t *)nlb_theta_weights_flat, (acc_t *)nlb_theta_bias,
                 (elem_t *)nlb_theta_out, false, NLB_THETA_LAYER_CONV_QUANT_ACC_SCALE));

  // Phi conv
  static elem_t nlb_phi_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  MEASURE_KERNEL(K_PHI, gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_PHI_KERNEL, 1, 0, (elem_t *)input, (elem_t *)nlb_phi_weights_flat,
                 (acc_t *)nlb_phi_bias, (elem_t *)nlb_phi_out, false, NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE));

  // G conv
  static elem_t nlb_g_out[CONV2_CHANNELS][CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  MEASURE_KERNEL(K_NLB_G, gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_G_KERNEL, 1, 0, (elem_t *)input, (elem_t *)nlb_g_weights_flat,
                 (acc_t *)nlb_g_bias, (elem_t *)nlb_g_out, false, NLB_G_LAYER_CONV_QUANT_ACC_SCALE));

  // 3-stage split (see NLB_STAGE1_STORE_SCALE/NLB_STAGE2_BERT_SCALE above):
  // the fused single-matmul SOFTMAX (as in braggnn_tune_pt2e.c, matching
  // TVM's actual codegen) was verified on real hardware to overflow I-BERT's
  // int32 sum_exp accumulator with this bert_scale. Split into (1) raw
  // theta@phi -> NO_ACTIVATION, requantized to int8, then (2) an
  // identity-matmul that just triggers SOFTMAX on the already-int8 logits.
  static elem_t nlb_logits_raw[CONV1_2D][CONV1_2D];
  MEASURE_KERNEL(K_ATTN_LOGITS, tiled_matmul_auto(CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I=81, dim_J=81, dim_K=32
                    (elem_t *)nlb_theta_out,           // A [81x32], stride=32
                    (elem_t *)nlb_phi_out,             // B [81x32], transposed to [32x81]
                    NULL, (elem_t *)nlb_logits_raw,    // C [81][81], raw logits (int8)
                    CONV2_FILTERS, CONV2_FILTERS, CONV1_2D, CONV1_2D,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, NLB_STAGE1_STORE_SCALE, 0,
                    false, false, true, false, false, 0, WS));

  static elem_t identity_81[CONV1_2D][CONV1_2D];
  static int identity_81_ready = 0;
  if (!identity_81_ready) {
    for (int i = 0; i < CONV1_2D; i++)
      for (int j = 0; j < CONV1_2D; j++)
        identity_81[i][j] = (i == j) ? 1 : 0;
    identity_81_ready = 1;
  }

  static elem_t attention_out[CONV1_2D][CONV1_2D];
  MEASURE_KERNEL(K_ATTN_SOFTMAX, tiled_matmul_auto(CONV1_2D, CONV1_2D, CONV1_2D,      // dim_K=81 for the identity pass-through
                    (elem_t *)nlb_logits_raw,           // A [81][81], raw int8 logits
                    (elem_t *)identity_81,              // B [81][81], identity
                    NULL, (elem_t *)attention_out,      // C [81][81]
                    CONV1_2D, CONV1_2D, CONV1_2D, CONV1_2D,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    SOFTMAX, ACC_SCALE_IDENTITY, NLB_STAGE2_BERT_SCALE,
                    false, false, false, false, false, 0, WS));

  // Step 2: attended = softmax(attention) @ g = [81][81] @ [81][32] = [81][32]
  static elem_t attended_output[CONV1_2D][CONV2_FILTERS];
  MEASURE_KERNEL(K_ATTENDED_MATMUL, tiled_matmul_auto(CONV1_2D, CONV2_FILTERS, CONV1_2D,
                    (elem_t *)attention_out,      // A [81][81]
                    (elem_t *)nlb_g_out,          // B [81][32]
                    NULL, (elem_t *)attended_output,
                    CONV1_2D, CONV2_FILTERS, CONV2_FILTERS, CONV2_FILTERS,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION,
                    NLB_MATMUL_1_QUANT_ACC_SCALE,
                    0.0, false, false, false, false, false, 0, WS));

  // Out conv: attended_output [81][32] is NHWC [9][9][32] in memory
  static elem_t nlb_output[CONV1_DIM][CONV1_DIM][64];
  MEASURE_KERNEL(K_NLB_OUT_CONV, gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS,
                 NLB_OUT_KERNEL, 1, 0, (elem_t *)attended_output,
                 (elem_t *)nlb_out_weights_flat, (acc_t *)nlb_out_bias,
                 (elem_t *)nlb_output, false, NLB_OUT_CNN_CONV_QUANT_ACC_SCALE));

  // resadd with RELU + fused requant scale
  MEASURE_KERNEL(K_RESADD, tiled_resadd_auto(CONV1_2D, CONV1_FILTERS,
                    NLB_ADD_B_SCALE,       // A_scale for input (skip connection)
                    NLB_ADD_A_SCALE,       // B_scale for nlb_output (out_cnn)
                    CNN_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE,  // C_scale
                    (elem_t *)input,
                    (elem_t *)nlb_output,
                    (elem_t *)output, true, WS));  // relu=true
}

void gemmini_inference(
    float *fp32_input, elem_t *output) {

  // QuantizeLinear: y_scale = 1/INPUT_DEQUANT_SCALE (PT2E/GemminiQuantizer input scale).
  // Matches TVM's own compiled quantize op exactly (gemmini_out/lib1.c's
  // fused_transpose_quantize): round-to-nearest, then clamp to [-128, 127] --
  // a bare (elem_t) cast on an out-of-range or non-integral float multiply is
  // not equivalent (truncates toward zero instead of rounding, and int8
  // overflow from a float->int8 cast is undefined behavior, not saturation).
  elem_t input[BATCH][INPUT_DIM][INPUT_DIM][INPUT_CHANNELS];
  for (int i = 0; i < BATCH * INPUT_DIM * INPUT_DIM * INPUT_CHANNELS; i++){
      float raw = ((float*)fp32_input)[i] * INPUT_DEQUANT_SCALE;
      // Hand-rolled round-half-away-from-zero (same behavior as roundf) --
      // the -baremetal/-pk targets are nostdlib/static and don't link libm,
      // so roundf() is unresolved there even though -linux builds fine.
      float v = raw >= 0.0f ? (float)(int)(raw + 0.5f) : (float)(int)(raw - 0.5f);
      v = v > 127.0f ? 127.0f : v;
      v = v < -128.0f ? -128.0f : v;
      ((elem_t*)input)[i] = (elem_t)v;
  }

  static elem_t conv1_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                         [CONV1_FILTERS]; // 11x11 -> 9x9x64 (3x3, no pad)
  static elem_t nlb_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                       [CONV1_FILTERS]; // NLB output: 9x9x64
  static elem_t conv2_out[CONV2_CHANNELS][CONV2_DIM][CONV2_DIM]
                         [CONV2_FILTERS]; // 9x9 -> 7x7x32 (3x3, no pad)
  static elem_t conv3_out[CONV3_KERNEL][CONV3_DIM][CONV3_DIM]
                         [CONV3_FILTERS]; // 7x7 -> 5x5x8 (3x3, no pad)

  static elem_t fc1_out[FC1_UNITS];
  static elem_t fc2_out[FC2_UNITS];
  static elem_t fc3_out[FC3_UNITS];
  static elem_t fc4_out[FC4_UNITS];

  MEASURE_KERNEL(K_CONV1, gemmini_conv2d(BATCH, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS, CONV1_FILTERS,
                 CONV1_KERNEL, 1, 0, (elem_t *)input, (elem_t *)conv1_weights_flat,
                 (acc_t *)conv1_bias, (elem_t *)conv1_out, false, CNN_LAYERS_0_CONV_QUANT_ACC_SCALE));

  gemmini_nlb((elem_t *)conv1_out, (elem_t *)nlb_out);

  MEASURE_KERNEL(K_CONV2, gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 CONV2_KERNEL, 1, 0, (elem_t *)nlb_out, (elem_t *)conv2_weights_flat,
                 (acc_t *)conv2_bias, (elem_t *)conv2_out, true,
                 CNN_LAYERS_2_CONV_QUANT_ACC_SCALE * CNN_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE));

  MEASURE_KERNEL(K_CONV3, gemmini_conv2d(BATCH, CONV2_DIM, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS,
                 CONV3_KERNEL, 1, 0, (elem_t *)conv2_out,
                 (elem_t *)conv3_weights_flat, (acc_t *)conv3_bias,
                 (elem_t *)conv3_out, true,
                 CNN_LAYERS_4_CONV_QUANT_ACC_SCALE * CNN_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE));

  // No CPU-side flatten: conv3_out's physical layout is already H,W,C
  // contiguous (= CONV3_FLATTENED), matching TVM's approach exactly --
  // TVM never physically reorders this data either (ConvertLayout's
  // NHWC->NCHW permute_dims here is absorbed as pure metadata by
  // FoldPermuteDims). Instead fc1_weights' 200 columns are pre-permuted
  // from the original nn.Linear.weight's C,H,W order into this same H,W,C
  // order at header-generation time (see FixFCFlattenLayout in
  // src/relax/transform/fold_permute_dims.cc for the reference formula),
  // so the raw conv3_out buffer can be passed straight through.
  MEASURE_KERNEL(K_FC1, gemmini_fc(CONV3_CHANNELS, CONV3_FLATTENED, FC1_UNITS, (elem_t *)conv3_out,
             (elem_t *)fc1_weights, (acc_t *)fc1_bias, (elem_t *)fc1_out, true, DENSE_LAYERS_0_GEMM_ACC_SCALE));
  MEASURE_KERNEL(K_FC2, gemmini_fc(1, FC1_UNITS, FC2_UNITS, (elem_t *)fc1_out, (elem_t *)fc2_weights,
             (acc_t *)fc2_bias, (elem_t *)fc2_out, true, DENSE_LAYERS_2_GEMM_ACC_SCALE));

  MEASURE_KERNEL(K_FC3, gemmini_fc(1, FC2_UNITS, FC3_UNITS, (elem_t *)fc2_out, (elem_t *)fc3_weights,
             (acc_t *)fc3_bias, (elem_t *)fc3_out, true, DENSE_LAYERS_4_GEMM_ACC_SCALE));

  MEASURE_KERNEL(K_FC4, gemmini_fc(1, FC3_UNITS, FC4_UNITS, (elem_t *)fc3_out, (elem_t *)fc4_weights,
             (acc_t *)fc4_bias, (elem_t *)fc4_out, true, DENSE_LAYERS_6_GEMM_ACC_SCALE));

  MEASURE_KERNEL(K_OUTPUT, gemmini_fc(1, FC4_UNITS, OUTPUT_UNITS, (elem_t *)fc4_out,
             (elem_t *)output_weights, (acc_t *)output_bias, (elem_t *)output,
             false, DENSE_LAYERS_8_GEMM_ACC_SCALE));
}

// float absolute value
static inline float fabsf_custom(float x) {
  return x >= 0.0f ? x : -x;
}

// Print float with 4 decimal places using only integer printf
// (bare-metal printf does not support %f)
static void print_float4(float f) {
  if (f < 0) {
    printf("-");
    f = -f;
  }
  int integer_part = (int)f;
  int frac_part = (int)((f - (float)integer_part) * 10000.0f + 0.5f);
  if (frac_part >= 10000) {
    integer_part++;
    frac_part -= 10000;
  }
  printf("%d.%04d", integer_part, frac_part);
}


int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  printf("==============================================\n");
  printf("BraggNN Gemmini through native C (PT2E/GemminiQuantizer weights)\n");
  printf("==============================================\n");

  printf("Flushing Gemmini TLB\n");
  gemmini_flush(0);

  // int8 -> pixel coordinates, derived from the PT2E model's final Linear's
  // own output scale (OUTPUT_DEQUANT_SCALE = output_scale * PATCH_SIZE),
  // rather than assuming output_scale == input_scale like the ONNX pipeline did.
  const float DEQUANT_SCALE = OUTPUT_DEQUANT_SCALE;

  elem_t predictions[OUTPUT_UNITS];
  unsigned long long cycle_counts[NUM_TEST_PATCHES];
  unsigned long long total_cycles = 0;
  float total_x_error = 0.0f;
  float total_y_error = 0.0f;

  printf("\nInput shape: [%d, %d, %d, %d]\n", BATCH, INPUT_CHANNELS, INPUT_DIM, INPUT_DIM);
  printf("Running %d inferences (%d warmup + %d measured)\n",
         NUM_TEST_PATCHES + 1, 1, NUM_TEST_PATCHES);

  // Warmup: drop first run to warm caches. Per-kernel DMA counters are
  // collected the same way as the measured passes below, just discarded.
  printf("\n--- Warmup (patch 0) ---\n");
  g_pass_mode = PASS_DMA;
  reset_dma_kernel_table(g_dma_kernel);
  unsigned long long start = read_cycles();
  gemmini_inference(test_inputs[5], predictions);
  unsigned long long end = read_cycles();
  printf("Warmup done\n");
  float pred_x = predictions[0] * DEQUANT_SCALE;
  float pred_y = predictions[1] * DEQUANT_SCALE;
  float actual_x = test_labels[5][0];
  float actual_y = test_labels[5][1];
  float err_x = pred_x - actual_x;
  float err_y = pred_y - actual_y;

  unsigned long long elapsed = end - start;
  // Not accumulated into total_cycles/total_x_error/total_y_error -- this
  // run is the warmup and is deliberately excluded from the measured
  // averages below (which divide by NUM_TEST_PATCHES, not NUM_TEST_PATCHES+1).

  printf("cycles: %llu\n", elapsed);
  printf("pred: ("); print_float4(pred_x); printf(", "); print_float4(pred_y);
  printf("), actual: ("); print_float4(actual_x); printf(", "); print_float4(actual_y);
  printf("), error: ("); print_float4(err_x); printf(", "); print_float4(err_y);
  printf(")\n");

  // Pass 1: per-kernel DMA/wait counters.
  printf("\n==============================================\n");
  printf("Pass 1: DMA/wait counters (per kernel)\n");
  printf("==============================================\n");

  g_pass_mode = PASS_DMA;
  dma_stats_t total_dma_kernel[NUM_KERNELS];
  reset_dma_kernel_table(total_dma_kernel);
  unsigned long long total_kernel_cycles_pass1[NUM_KERNELS];
  reset_kernel_cycles(total_kernel_cycles_pass1);

  for (int i = 0; i < NUM_TEST_PATCHES; i++) {
    printf("\n--- Inference %d/%d ---\n", i + 1, NUM_TEST_PATCHES);

    reset_dma_kernel_table(g_dma_kernel);
    reset_kernel_cycles(g_kernel_cycles);

    unsigned long long start = read_cycles();
    asm volatile(".word 0x8013");  // TracerV start trigger
    gemmini_inference(test_inputs[i], predictions);
    asm volatile(".word 0x10013"); // TracerV end trigger
    unsigned long long end = read_cycles();

    float pred_x = predictions[0] * DEQUANT_SCALE;
    float pred_y = predictions[1] * DEQUANT_SCALE;
    float actual_x = test_labels[i][0];
    float actual_y = test_labels[i][1];
    float err_x = pred_x - actual_x;
    float err_y = pred_y - actual_y;

    unsigned long long elapsed = end - start;
    cycle_counts[i] = elapsed;
    total_cycles += elapsed;
    add_dma_kernel_table(total_dma_kernel, g_dma_kernel);
    add_kernel_cycles(total_kernel_cycles_pass1, g_kernel_cycles);
    total_x_error += fabsf_custom(err_x);
    total_y_error += fabsf_custom(err_y);

    // Printed once, after the whole inference has finished.
    printf("cycles: %llu\n", elapsed);
    print_kernel_cycles(g_kernel_cycles);
    print_dma_kernel_table(g_dma_kernel);
    printf("pred: ("); print_float4(pred_x); printf(", "); print_float4(pred_y);
    printf("), actual: ("); print_float4(actual_x); printf(", "); print_float4(actual_y);
    printf("), error: ("); print_float4(err_x); printf(", "); print_float4(err_y);
    printf(")\n");
  }

  dma_stats_t avg_dma_kernel[NUM_KERNELS];
  reset_dma_kernel_table(avg_dma_kernel);
  add_dma_kernel_table(avg_dma_kernel, total_dma_kernel);
  div_dma_kernel_table(avg_dma_kernel, NUM_TEST_PATCHES);
  div_kernel_cycles(total_kernel_cycles_pass1, NUM_TEST_PATCHES);

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs: %llu\n", NUM_TEST_PATCHES,
         total_cycles / NUM_TEST_PATCHES);
  printf("Avg over %d runs (per kernel):\n", NUM_TEST_PATCHES);
  print_kernel_cycles(total_kernel_cycles_pass1);
  print_dma_kernel_table(avg_dma_kernel);
  printf("Avg error over %d runs: (", NUM_TEST_PATCHES);
  print_float4(total_x_error / NUM_TEST_PATCHES); printf(", ");
  print_float4(total_y_error / NUM_TEST_PATCHES); printf(")\n");
  printf("==============================================\n");
  printf("BraggNN inference completed successfully\n");
  printf("==============================================\n");

  // Pass 2: re-run the same patches with a different set of 8 counters,
  // since only 8 hardware counter slots exist and pass 1 already used all
  // of them for DMA/wait cycles. This pass profiles EXE/control overhead
  // instead, per kernel, so the two passes together show where the
  // cycles really go inside each kernel.
  printf("\n==============================================\n");
  printf("Pass 2: EXE / control-overhead counters (per kernel)\n");
  printf("==============================================\n");

  g_pass_mode = PASS_EXE;
  unsigned long long total_cycles_pass2 = 0;
  exe_stats_t total_exe_kernel[NUM_KERNELS];
  reset_exe_kernel_table(total_exe_kernel);
  unsigned long long total_kernel_cycles_pass2[NUM_KERNELS];
  reset_kernel_cycles(total_kernel_cycles_pass2);

  for (int i = 0; i < NUM_TEST_PATCHES; i++) {
    printf("\n--- Inference %d/%d (pass 2) ---\n", i + 1, NUM_TEST_PATCHES);

    reset_exe_kernel_table(g_exe_kernel);
    reset_kernel_cycles(g_kernel_cycles);

    unsigned long long start2 = read_cycles();
    gemmini_inference(test_inputs[i], predictions);
    unsigned long long end2 = read_cycles();

    unsigned long long elapsed2 = end2 - start2;
    total_cycles_pass2 += elapsed2;
    add_exe_kernel_table(total_exe_kernel, g_exe_kernel);
    add_kernel_cycles(total_kernel_cycles_pass2, g_kernel_cycles);

    printf("cycles: %llu\n", elapsed2);
    print_kernel_cycles(g_kernel_cycles);
    print_exe_kernel_table(g_exe_kernel);
  }

  exe_stats_t avg_exe_kernel[NUM_KERNELS];
  reset_exe_kernel_table(avg_exe_kernel);
  add_exe_kernel_table(avg_exe_kernel, total_exe_kernel);
  div_exe_kernel_table(avg_exe_kernel, NUM_TEST_PATCHES);
  div_kernel_cycles(total_kernel_cycles_pass2, NUM_TEST_PATCHES);

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs (pass 2): %llu\n", NUM_TEST_PATCHES,
         total_cycles_pass2 / NUM_TEST_PATCHES);
  printf("Avg over %d runs (per kernel):\n", NUM_TEST_PATCHES);
  print_kernel_cycles(total_kernel_cycles_pass2);
  print_exe_kernel_table(avg_exe_kernel);
  printf("==============================================\n");

  // Pass 3: re-run the same patches with the MAIN_* controller-overlap
  // counters, per kernel, to directly measure how much Load/Store DMA
  // activity actually overlaps with Ex (compute) in each kernel.
  printf("\n==============================================\n");
  printf("Pass 3: controller-overlap (MAIN_*) counters (per kernel)\n");
  printf("==============================================\n");

  g_pass_mode = PASS_MAIN;
  unsigned long long total_cycles_pass3 = 0;
  main_stats_t total_main_kernel[NUM_KERNELS];
  reset_main_kernel_table(total_main_kernel);
  unsigned long long total_kernel_cycles_pass3[NUM_KERNELS];
  reset_kernel_cycles(total_kernel_cycles_pass3);

  for (int i = 0; i < NUM_TEST_PATCHES; i++) {
    printf("\n--- Inference %d/%d (pass 3) ---\n", i + 1, NUM_TEST_PATCHES);

    reset_main_kernel_table(g_main_kernel);
    reset_kernel_cycles(g_kernel_cycles);

    unsigned long long start3 = read_cycles();
    gemmini_inference(test_inputs[i], predictions);
    unsigned long long end3 = read_cycles();

    unsigned long long elapsed3 = end3 - start3;
    total_cycles_pass3 += elapsed3;
    add_main_kernel_table(total_main_kernel, g_main_kernel);
    add_kernel_cycles(total_kernel_cycles_pass3, g_kernel_cycles);

    // none_busy/ex_busy_not_compute per kernel are computed against that
    // kernel's own cycle count (g_kernel_cycles), not the whole
    // inference's elapsed3.
    printf("cycles: %llu\n", elapsed3);
    print_kernel_cycles(g_kernel_cycles);
    print_main_kernel_table(g_main_kernel, g_kernel_cycles);
  }

  main_stats_t avg_main_kernel[NUM_KERNELS];
  reset_main_kernel_table(avg_main_kernel);
  add_main_kernel_table(avg_main_kernel, total_main_kernel);
  div_main_kernel_table(avg_main_kernel, NUM_TEST_PATCHES);
  div_kernel_cycles(total_kernel_cycles_pass3, NUM_TEST_PATCHES);

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs (pass 3): %llu\n", NUM_TEST_PATCHES,
         total_cycles_pass3 / NUM_TEST_PATCHES);
  printf("Avg over %d runs (per kernel):\n", NUM_TEST_PATCHES);
  print_kernel_cycles(total_kernel_cycles_pass3);
  print_main_kernel_table(avg_main_kernel, total_kernel_cycles_pass3);
  printf("==============================================\n");

  exit(0);
}
