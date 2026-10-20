/*
 * 有界环形缓冲 + keep-latest 队列。
 *
 * 两个不同的东西：
 *   RingBuffer  采集线程写、处理线程读，解决速率不匹配。
 *   LatestQueue depth=1 的待确认队列，处理不过来时丢掉最旧的，
 *                而不是排队。排队只会把 P99 拖长，不会提高吞吐。
 *
 * 单写单读，不加锁（内存序用 acquire/release 保证），热路径不分配、
 * 不做系统调用。
 */

#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : buf_(capacity), cap_(capacity), write_(0), read_(0) {}

    size_t available() const {
        const size_t w = write_.load(std::memory_order_acquire);
        const size_t r = read_.load(std::memory_order_acquire);
        return w - r;
    }

    size_t space() const { return cap_ - available(); }

    // 写满时丢弃本次写入，返回实际写入的样本数（0 表示溢出）。
    size_t write(const int16_t *src, size_t n) {
        const size_t w = write_.load(std::memory_order_relaxed);
        const size_t r = read_.load(std::memory_order_acquire);
        const size_t used = w - r;
        if (used + n > cap_) {
            overflow_.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        const size_t pos = w % cap_;
        const size_t first = (pos + n <= cap_) ? n : (cap_ - pos);
        std::memcpy(&buf_[pos], src, first * sizeof(int16_t));
        if (first < n) {
            std::memcpy(&buf_[0], src + first, (n - first) * sizeof(int16_t));
        }
        write_.store(w + n, std::memory_order_release);
        return n;
    }

    // 读 n 个样本；不足则返回实际可读数量。
    size_t read(int16_t *dst, size_t n) {
        const size_t w = write_.load(std::memory_order_acquire);
        const size_t r = read_.load(std::memory_order_relaxed);
        const size_t avail = w - r;
        const size_t take = (n < avail) ? n : avail;
        if (take == 0) {
            return 0;
        }
        const size_t pos = r % cap_;
        const size_t first = (pos + take <= cap_) ? take : (cap_ - pos);
        std::memcpy(dst, &buf_[pos], first * sizeof(int16_t));
        if (first < take) {
            std::memcpy(dst + first, &buf_[0], (take - first) * sizeof(int16_t));
        }
        read_.store(r + take, std::memory_order_release);
        return take;
    }

    uint64_t overflow_count() const {
        return overflow_.load(std::memory_order_relaxed);
    }

private:
    std::vector<int16_t> buf_;
    size_t cap_;
    std::atomic<size_t> write_;
    std::atomic<size_t> read_;
    std::atomic<uint64_t> overflow_{0};
};

/* depth=1 的待确认队列：新窗口直接覆盖旧窗口，并累计丢弃次数。 */
template <typename T>
class LatestQueue {
public:
    LatestQueue() : has_(false) {}

    void push(T value) {
        if (has_.load(std::memory_order_acquire)) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        value_ = value;
        has_.store(true, std::memory_order_release);
    }

    bool pop(T &out) {
        if (!has_.load(std::memory_order_acquire)) {
            return false;
        }
        out = value_;
        has_.store(false, std::memory_order_release);
        return true;
    }

    uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

private:
    T value_{};
    std::atomic<bool> has_;
    std::atomic<uint64_t> dropped_{0};
};

#endif /* RING_BUFFER_H */
