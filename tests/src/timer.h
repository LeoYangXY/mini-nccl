/*
 * tests/src/timer.h — 计时器头文件
 * ----------------------------------------------------------------------------
 * 声明测试用的计时类 timer：构造时记录起点，elapsed()/reset() 返回/重置经过的
 * 秒数。用于性能测试的时间测量。
 */

#ifndef _408319ecdd5b47b28bf8f511c4fdf816
#define _408319ecdd5b47b28bf8f511c4fdf816

#include <cstdint>

// Can't include <chrono> because of bug with gcc 10.3.0
class timer {
  std::uint64_t t0;
public:
  timer();
  double elapsed() const;
  double reset();
};

#endif
