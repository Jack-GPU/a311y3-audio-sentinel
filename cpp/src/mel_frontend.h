/*
 * 板端 Log-Mel 前端（C99，无外部依赖）。
 *
 * 必须与 python/audio_frontend.py 的 ReferenceFrontend 数值一致。
 * 两边任何一处实现不同（窗函数、mel 归一、dB 系数、eps），
 * 都会让 INT8 模型在板上的表现和 PC 上完全不同——这类问题用
 * python/mel_parity.py 定位，不要靠猜。
 *
 * 设计取舍：
 *   - FFT 用 float 做（A55 有 FPU，精度优先），mel 滤波器系数用 Q15，
 *     因为那部分是定点乘累加，量化误差会累积。
 *   - 整段处理而非流式：流式的状态复用由上层（main.cpp）按窗调度。
 */

#ifndef MEL_FRONTEND_H
#define MEL_FRONTEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MEL_MAX_FFT   1024
#define MEL_MAX_MELS  128

typedef struct {
    int   sr;
    int   n_fft;
    int   win;
    int   hop;
    int   n_mels;
    float fmin;
    float fmax;
    float preemph;
    float top_db;
    float eps;
    int   mel_slaney;   /* 1 = slaney 面积归一；0 = htk */
    int   log_mult;     /* 10 (功率) 或 20 (幅度) */
} mel_cfg_t;

typedef struct mel_ctx mel_ctx_t;

/* 成功返回非 NULL；参数非法或内存不足返回 NULL。 */
mel_ctx_t *mel_create(const mel_cfg_t *cfg);
void       mel_destroy(mel_ctx_t *ctx);

/* 处理整段 PCM。out 需至少 n_mels * (*out_frames) 个 float。
 * 返回 0 成功。*out_frames 为 0 表示输入太短。 */
int mel_compute(mel_ctx_t *ctx, const int16_t *pcm, int n_samples,
                float *out, int *out_frames);

/* 帧数预估，便于调用方一次性分配。 */
int mel_frames_for(const mel_ctx_t *ctx, int n_samples);

/* 写入配置指纹，用于和 PC 侧 configs/pipeline.yaml 比对。 */
void mel_describe(const mel_ctx_t *ctx, char *buf, int buflen);

#ifdef __cplusplus
}
#endif

#endif /* MEL_FRONTEND_H */
