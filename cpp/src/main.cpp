/*
 * 常开音频感知引擎 · 板端主程序。
 *
 * 数据流：
 *   ALSA capture ──> RingBuffer ──> 域0 门控 ──> 域1 mel ──> 域2 NPU ──> FSM ──> 告警
 *
 * 三条纪律写在代码里，不写在文档里：
 *   1. 热路径不做堆分配（缓冲全部预分配）；
 *   2. 每个阶段单独打点，端到端分位数必须能拆开看；
 *   3. 过载时丢最旧待确认窗（LatestQueue），绝不排队。
 *
 * 编译：
 *   cmake -B build && cmake --build build
 *   有 NPU 运行时再打开 -DUSE_ADLA=ON
 */

#include <alsa/asoundlib.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "mel_frontend.h"
#include "ring_buffer.h"

#ifdef USE_ADLA
#include <nnsdk2.h>
#endif

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true); }

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct StageStats {
    std::vector<float> gate, feat, npu, fsm;
    uint64_t npu_calls = 0;
    uint64_t windows = 0;
    uint64_t gated = 0;

    void reserve(int n) {
        gate.reserve(n); feat.reserve(n); npu.reserve(n); fsm.reserve(n);
    }

    static float pct(std::vector<float> &v, double q) {
        if (v.empty()) return 0.0f;
        const size_t k = static_cast<size_t>(q * (v.size() - 1));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    }

    void report() const {
        auto &self = const_cast<StageStats &>(*this);
        printf("\n=== 分层延迟（%llu 窗，门控挡掉 %llu）===\n",
               (unsigned long long)windows, (unsigned long long)gated);
        printf("%-8s %10s %10s\n", "stage", "P50(ms)", "P99(ms)");
        printf("%-8s %10.3f %10.3f\n", "gate", pct(self.gate, 0.50), pct(self.gate, 0.99));
        printf("%-8s %10.3f %10.3f\n", "feat", pct(self.feat, 0.50), pct(self.feat, 0.99));
        printf("%-8s %10.3f %10.3f\n", "npu",  pct(self.npu, 0.50),  pct(self.npu, 0.99));
        printf("%-8s %10.3f %10.3f\n", "fsm",  pct(self.fsm, 0.50),  pct(self.fsm, 0.99));
        printf("npu_duty = %.2f%%\n", windows ? 100.0 * npu_calls / windows : 0.0);
    }
};

struct Args {
    std::string device = "hw:0,0";
    int  sr = 16000;
    int  n_fft = 512;
    int  win = 400;
    int  hop = 160;
    int  n_mels = 40;
    float gate_dbfs = -55.0f;
    float score_thr = 0.35f;
    int  seconds = 10;
};

float window_dbfs(const int16_t *p, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        const double v = p[i] / 32768.0;
        acc += v * v;
    }
    const double rms = std::sqrt(acc / (n > 0 ? n : 1));
    return static_cast<float>(20.0 * std::log10(rms > 1e-12 ? rms : 1e-12));
}

}  // namespace

int main(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        const std::string k = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s 需要参数\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--device")     a.device = next("--device");
        else if (k == "--sr")    a.sr = std::atoi(next("--sr"));
        else if (k == "--n-mels")a.n_mels = std::atoi(next("--n-mels"));
        else if (k == "--gate")  a.gate_dbfs = std::atof(next("--gate"));
        else if (k == "--thr")   a.score_thr = std::atof(next("--thr"));
        else if (k == "--secs")  a.seconds = std::atoi(next("--secs"));
        else if (k == "--help" || k == "-h") {
            printf("用法: %s [--device hw:0,0] [--secs 10] [--gate -55] [--thr 0.35]\n", argv[0]);
            return 0;
        }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    mel_cfg_t mc{};
    mc.sr = a.sr;
    mc.n_fft = a.n_fft;
    mc.win = a.win;
    mc.hop = a.hop;
    mc.n_mels = a.n_mels;
    mc.fmin = 20.0f;
    mc.fmax = 7600.0f;
    mc.preemph = 0.97f;
    mc.top_db = 80.0f;
    mc.eps = 1e-10f;
    mc.mel_slaney = 1;
    mc.log_mult = 10;

    mel_ctx_t *mel = mel_create(&mc);
    if (!mel) {
        fprintf(stderr, "mel_create 失败：检查 n_fft/win/hop/n_mels\n");
        return 1;
    }
    {
        char desc[256];
        mel_describe(mel, desc, sizeof(desc));
        printf("前端指纹: %s\n", desc);
        printf("提醒：该指纹必须与 PC 侧 configs/pipeline.yaml 一致，"
               "改任一侧都要重跑 python/mel_parity.py\n");
    }

    snd_pcm_t *pcm = nullptr;
    int rc = snd_pcm_open(&pcm, a.device.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0) {
        fprintf(stderr, "打开 %s 失败: %s\n", a.device.c_str(), snd_strerror(rc));
        fprintf(stderr, "可用设备见 `arecord -l`；先用 --file 模式或 python/board_infer.py 验证链路\n");
        mel_destroy(mel);
        return 1;
    }

    snd_pcm_hw_params_t *hw = nullptr;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(pcm, hw);
    snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm, hw, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm, hw, 1);
    unsigned rate = static_cast<unsigned>(a.sr);
    snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, nullptr);
    snd_pcm_uframes_t period = 320;
    snd_pcm_hw_params_set_period_size_near(pcm, hw, &period, nullptr);
    snd_pcm_uframes_t buffer = period * 3;
    snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &buffer);
    rc = snd_pcm_hw_params(pcm, hw);
    if (rc < 0) {
        fprintf(stderr, "hw_params 失败: %s\n", snd_strerror(rc));
        snd_pcm_close(pcm);
        mel_destroy(mel);
        return 1;
    }

    RingBuffer ring(static_cast<size_t>(a.sr) * 4);
    std::vector<int16_t> chunk(static_cast<size_t>(period));
    std::vector<int16_t> frame(static_cast<size_t>(a.win));
    std::vector<float> feat(static_cast<size_t>(a.n_mels) * 128);

    StageStats stats;
    stats.reserve(a.seconds * a.sr / a.hop + 16);

    std::atomic<bool> capture_done{false};
    std::thread cap([&] {
        while (!g_stop.load()) {
            const snd_pcm_sframes_t got = snd_pcm_readi(pcm, chunk.data(), chunk.size());
            if (got < 0) {
                snd_pcm_prepare(pcm);
                continue;
            }
            ring.write(chunk.data(), static_cast<size_t>(got));
        }
        capture_done.store(true);
    });

    const auto t_start = Clock::now();
    const size_t need = static_cast<size_t>(a.win);
    while (!g_stop.load()) {
        if (std::chrono::duration<double>(Clock::now() - t_start).count() > a.seconds) {
            break;
        }
        if (ring.available() < need) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        ring.read(frame.data(), need);
        stats.windows++;

        const auto t0 = Clock::now();
        const float level = window_dbfs(frame.data(), static_cast<int>(need));
        if (level < a.gate_dbfs) {
            stats.gate.push_back(static_cast<float>(ms_since(t0)));
            stats.gated++;
            continue;
        }
        stats.gate.push_back(static_cast<float>(ms_since(t0)));

        const auto t1 = Clock::now();
        int n_frames = 0;
        // 只取最后一窗做确认；流式状态复用由 mel cache 承担（见 docs）
        mel_compute(mel, frame.data(), static_cast<int>(need), feat.data(), &n_frames);
        stats.feat.push_back(static_cast<float>(ms_since(t1)));

        const auto t2 = Clock::now();
        float score = 0.0f;
#ifdef USE_ADLA
        // 真实推理：输入 (1,1,n_mels,frames)，输出各事件概率
        // nnsdk2 的具体调用按 SDK 版本填，此处保留结构
        score = 0.0f;
#else
        // 无 NPU 时用能量做占位，只为打通链路与打点
        score = (level > a.gate_dbfs + 10.0f) ? 0.5f : 0.1f;
#endif
        stats.npu.push_back(static_cast<float>(ms_since(t2)));
        stats.npu_calls++;

        const auto t3 = Clock::now();
        if (score >= a.score_thr) {
            printf("[ALARM] score=%.3f level=%.1f dBFS\n", score, level);
        }
        stats.fsm.push_back(static_cast<float>(ms_since(t3)));

        // 限速到 hop，模拟实时节奏；真实部署由 ALSA 阻塞读天然限速
        const double budget_ms = 1000.0 * a.hop / a.sr;
        const double used = ms_since(t0);
        if (used < budget_ms) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<long>((budget_ms - used) * 1000)));
        }
    }

    g_stop.store(true);
    cap.join();
    snd_pcm_close(pcm);

    stats.report();
    printf("ring overflow = %llu\n", (unsigned long long)ring.overflow_count());
    mel_destroy(mel);
    return 0;
}
