/*
 * 把 C 侧 mel 特征导出成文本，喂给 python/mel_parity.py --c-dump 比对。
 *
 * 为什么要有这个工具：PC 参考实现和板端实现在没对齐之前，
 * 差异来源只有两种可能——算法不同，或者数值路径不同。
 * 把两边的中间特征都 dump 出来逐元素比，是唯一能把它们分开的办法。
 *
 * 用法：
 *   ./mel_dump input.wav > c_mel.txt
 *   python ../python/mel_parity.py --c-dump c_mel.txt --wav input.wav
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mel_frontend.h"

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint16_t rd16(const unsigned char *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

int main(int argc, char **argv)
{
    FILE *f;
    unsigned char hdr[44];
    int16_t *pcm;
    long size;
    mel_cfg_t cfg;
    mel_ctx_t *ctx;
    float *feat;
    int frames = 0, m, t;

    if (argc < 2) {
        fprintf(stderr, "用法: %s input.wav\n", argv[0]);
        return 2;
    }

    f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 2;
    }
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr) ||
        memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "仅支持 16-bit PCM WAV\n");
        fclose(f);
        return 2;
    }
    if (rd16(hdr + 20) != 1 || rd16(hdr + 34) != 16) {
        fprintf(stderr, "需要 PCM / 16 bit\n");
        fclose(f);
        return 2;
    }

    fseek(f, 0, SEEK_END);
    size = ftell(f) - (long)sizeof(hdr);
    fseek(f, (long)sizeof(hdr), SEEK_SET);
    if (size <= 0) {
        fprintf(stderr, "空音频\n");
        fclose(f);
        return 2;
    }

    pcm = (int16_t *)malloc((size_t)size);
    if (!pcm) {
        fclose(f);
        return 2;
    }
    if (fread(pcm, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "读取失败\n");
        free(pcm);
        fclose(f);
        return 2;
    }
    fclose(f);

    memset(&cfg, 0, sizeof(cfg));
    cfg.sr = 16000;
    cfg.n_fft = 512;
    cfg.win = 400;
    cfg.hop = 160;
    cfg.n_mels = 40;
    cfg.fmin = 20.0f;
    cfg.fmax = 7600.0f;
    cfg.preemph = 0.97f;
    cfg.top_db = 80.0f;
    cfg.eps = 1e-10f;
    cfg.mel_slaney = 1;
    cfg.log_mult = 10;

    ctx = mel_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "mel_create 失败\n");
        free(pcm);
        return 1;
    }

    {
        const int n = (int)(size / 2);
        const int max_frames = mel_frames_for(ctx, n);
        char desc[256];
        mel_describe(ctx, desc, sizeof(desc));
        fprintf(stderr, "# %s\n", desc);

        feat = (float *)malloc(sizeof(float) * (size_t)cfg.n_mels * (size_t)(max_frames > 0 ? max_frames : 1));
        if (!feat) {
            mel_destroy(ctx);
            free(pcm);
            return 2;
        }
        if (mel_compute(ctx, pcm, n, feat, &frames) != 0) {
            fprintf(stderr, "mel_compute 失败\n");
            free(feat);
            mel_destroy(ctx);
            free(pcm);
            return 1;
        }

        printf("# n_mels=%d frames=%d\n", cfg.n_mels, frames);
        for (m = 0; m < cfg.n_mels; m++) {
            for (t = 0; t < frames; t++) {
                printf("%.8f%c", feat[m * frames + t], (t + 1 == frames) ? '\n' : ' ');
            }
        }
        free(feat);
    }

    mel_destroy(ctx);
    free(pcm);
    return 0;
}
