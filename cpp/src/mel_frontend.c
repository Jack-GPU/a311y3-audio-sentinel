#include "mel_frontend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MEL_PI 3.14159265358979323846f
#define Q15    32768.0f

struct mel_ctx {
    mel_cfg_t cfg;
    int   n_bins;
    float win_coef[MEL_MAX_FFT];
    float fb[MEL_MAX_MELS * (MEL_MAX_FFT / 2 + 1)];
};

/* ---------- mel scale（与 Python 侧逐行对应） ---------- */

static float hz_to_mel(float f, int slaney)
{
    if (!slaney) {
        return 2595.0f * log10f(1.0f + f / 700.0f);
    }
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    if (f >= min_log_hz) {
        return min_log_mel + logf(f / min_log_hz) / logstep;
    }
    return f / f_sp;
}

static float mel_to_hz(float m, int slaney)
{
    if (!slaney) {
        return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
    }
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    if (m >= min_log_mel) {
        return min_log_hz * expf(logstep * (m - min_log_mel));
    }
    return f_sp * m;
}

static float q15(float x)
{
    float v = roundf(x * Q15);
    if (v > Q15 - 1.0f) v = Q15 - 1.0f;
    if (v < -Q15)       v = -Q15;
    return v / Q15;
}

/* ---------- radix-2 FFT（原位，in-place） ---------- */

static void fft_radix2(float *re, float *im, int n)
{
    int i, j, k, len;

    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }

    for (len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * MEL_PI / (float)len;
        const float wr = cosf(ang);
        const float wi = sinf(ang);
        for (i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (k = 0; k < len / 2; k++) {
                const int a = i + k;
                const int b = a + len / 2;
                const float ur = re[a], ui = im[a];
                const float xr = re[b], xi = im[b];
                const float vr = xr * cr - xi * ci;
                const float vi = xr * ci + xi * cr;
                re[a] = ur + vr; im[a] = ui + vi;
                re[b] = ur - vr; im[b] = ui - vi;
                {
                    const float ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = ncr;
                }
            }
        }
    }
}

/* ---------- 生命周期 ---------- */

mel_ctx_t *mel_create(const mel_cfg_t *cfg)
{
    mel_ctx_t *ctx;
    int m, b;

    if (!cfg || cfg->n_fft <= 0 || cfg->n_fft > MEL_MAX_FFT ||
        (cfg->n_fft & (cfg->n_fft - 1)) != 0 ||
        cfg->win <= 0 || cfg->win > cfg->n_fft ||
        cfg->hop <= 0 || cfg->n_mels <= 0 || cfg->n_mels > MEL_MAX_MELS) {
        return NULL;
    }

    ctx = (mel_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    ctx->cfg = *cfg;
    ctx->n_bins = cfg->n_fft / 2 + 1;

    /* Hann 周期窗：分母是 win 而不是 win-1，差这一个点就够让特征漂移 */
    for (m = 0; m < cfg->win; m++) {
        ctx->win_coef[m] = 0.5f - 0.5f * cosf(2.0f * MEL_PI * (float)m / (float)cfg->win);
    }

    /* 三角滤波器组，按 Slaney 面积归一 */
    {
        const float fmax = (cfg->fmax > 0.0f) ? cfg->fmax : (float)cfg->sr / 2.0f;
        float mel_pts[MEL_MAX_MELS + 2];
        float hz_pts[MEL_MAX_MELS + 2];
        for (m = 0; m < cfg->n_mels + 2; m++) {
            mel_pts[m] = hz_to_mel(cfg->fmin, cfg->mel_slaney) +
                         (hz_to_mel(fmax, cfg->mel_slaney) -
                          hz_to_mel(cfg->fmin, cfg->mel_slaney)) *
                         (float)m / (float)(cfg->n_mels + 1);
            hz_pts[m] = mel_to_hz(mel_pts[m], cfg->mel_slaney);
        }
        for (m = 0; m < cfg->n_mels; m++) {
            const float lo = hz_pts[m], ce = hz_pts[m + 1], hi = hz_pts[m + 2];
            float norm = 1.0f;
            if (cfg->mel_slaney) {
                const float width = hz_pts[m + 2] - hz_pts[m];
                norm = (width > 1e-12f) ? (2.0f / width) : 0.0f;
            }
            for (b = 0; b < ctx->n_bins; b++) {
                const float f = (float)cfg->sr / 2.0f * (float)b / (float)(ctx->n_bins - 1);
                float w = 0.0f;
                if (f > lo && f < hi) {
                    w = (f <= ce) ? (f - lo) / (ce - lo) : (hi - f) / (hi - ce);
                }
                ctx->fb[m * ctx->n_bins + b] = q15(w * norm);
            }
        }
    }

    return ctx;
}

void mel_destroy(mel_ctx_t *ctx)
{
    free(ctx);
}

int mel_frames_for(const mel_ctx_t *ctx, int n_samples)
{
    if (!ctx || n_samples < ctx->cfg.win) {
        return 0;
    }
    return (n_samples - ctx->cfg.win) / ctx->cfg.hop + 1;
}

int mel_compute(mel_ctx_t *ctx, const int16_t *pcm, int n_samples,
                float *out, int *out_frames)
{
    const mel_cfg_t *c;
    int n_frames, t, b, m;
    float *re = NULL, *im = NULL, *power = NULL;
    float peak_db = -1e30f;

    if (!ctx || !pcm || !out || !out_frames) {
        return -1;
    }
    c = &ctx->cfg;
    n_frames = mel_frames_for(ctx, n_samples);
    *out_frames = n_frames;
    if (n_frames <= 0) {
        return 0;
    }

    re = (float *)malloc(sizeof(float) * (size_t)c->n_fft);
    im = (float *)malloc(sizeof(float) * (size_t)c->n_fft);
    power = (float *)malloc(sizeof(float) * (size_t)c->n_bins);
    if (!re || !im || !power) {
        free(re); free(im); free(power);
        return -2;
    }

    for (t = 0; t < n_frames; t++) {
        const int start = t * c->hop;
        float prev = 0.0f;
        int i;

        for (i = 0; i < c->n_fft; i++) {
            float x = 0.0f;
            if (i < c->win && start + i < n_samples) {
                /* int16 → [-1,1)，再预加重 */
                x = (float)pcm[start + i] / 32768.0f;
                x = (i == 0) ? x : (x - c->preemph * prev);
                prev = (float)pcm[start + i] / 32768.0f;
                x *= ctx->win_coef[i];
            }
            re[i] = x;
            im[i] = 0.0f;
        }

        fft_radix2(re, im, c->n_fft);

        for (b = 0; b < ctx->n_bins; b++) {
            power[b] = re[b] * re[b] + im[b] * im[b];
        }

        for (m = 0; m < c->n_mels; m++) {
            float acc = 0.0f;
            const float *w = &ctx->fb[m * ctx->n_bins];
            for (b = 0; b < ctx->n_bins; b++) {
                acc += w[b] * power[b];
            }
            if (acc < c->eps) {
                acc = c->eps;
            }
            {
                float db = (float)c->log_mult * log10f(acc);
                out[m * n_frames + t] = db;
                if (db > peak_db) {
                    peak_db = db;
                }
            }
        }
    }

    /* top_db 截断 + 归一化（与 Python feature_norm=db_over_top_db 对应） */
    for (m = 0; m < c->n_mels; m++) {
        for (t = 0; t < n_frames; t++) {
            float db = out[m * n_frames + t];
            float lo = peak_db - c->top_db;
            if (db < lo)      db = lo;
            if (db > 0.0f)    db = 0.0f;
            out[m * n_frames + t] = db / c->top_db;
        }
    }

    free(re);
    free(im);
    free(power);
    return 0;
}

void mel_describe(const mel_ctx_t *ctx, char *buf, int buflen)
{
    if (!ctx || !buf || buflen <= 0) {
        return;
    }
    snprintf(buf, (size_t)buflen,
             "sr=%d n_fft=%d win=%d hop=%d n_mels=%d fmin=%.1f fmax=%.1f "
             "preemph=%.2f norm=%s log=%dx top_db=%.0f",
             ctx->cfg.sr, ctx->cfg.n_fft, ctx->cfg.win, ctx->cfg.hop,
             ctx->cfg.n_mels, ctx->cfg.fmin, ctx->cfg.fmax, ctx->cfg.preemph,
             ctx->cfg.mel_slaney ? "slaney" : "htk", ctx->cfg.log_mult,
             ctx->cfg.top_db);
}
