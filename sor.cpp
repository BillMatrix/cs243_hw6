#include <iostream>
#include <random>
#include <exception>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <string>
#include <time.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <semaphore.h>
#ifdef __linux__
#include <linux/futex.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <limits.h>
#endif

template<typename T> class Matrix;

/**
Matrix: a 2D rectangular matrix, heap-allocated.
Stored in row-major, 0-indexed order.

Use .at(i, j) to access the i-th row, j-th column, both for
reading and writing.
**/

template<typename T>
class Matrix
{
public:
    typedef T data_type;
    typedef size_t size_type;

private:
    size_type m_row, m_col;
    T *m_data;

    void alloc_data(size_t rows, size_t cols) {
        // make sure we allocate at least 2 MB always
        // this triggers the mmap path in malloc, which in turn causes the kernel
        // to give us allocation at a random address (due to ASLR)
        // this means we don't see cache effects from repeatedly allocating and freeing
        // the matrices
        // it is ok to overallocate because the memory which is not touched won't be
        // paged in
        size_t data_size = rows * cols;
        if (data_size * sizeof(data_type) < 2 * 1024 * 1024)
            data_size = 2 * 1024 * 1024 / sizeof(data_type);
        m_data = new data_type[data_size];
    }

public:
    Matrix(size_type rows, size_type cols) : m_row(rows), m_col(cols), m_data(nullptr) {
        if (rows == 0 && cols == 0)
            return;
        if (rows * cols < rows)
            throw std::overflow_error("matrix too big");
        alloc_data(rows, cols);
    }
    ~Matrix() {
        delete[] m_data;
    }
    Matrix(const Matrix& other) : m_row(other.m_row), m_col(other.m_col) {
        alloc_data(m_row, m_col);
        memcpy(m_data, other.m_data, m_row * m_col * sizeof(data_type));
    }
    Matrix(Matrix&& other) noexcept : m_row(other.m_row), m_col(other.m_col), m_data(other.m_data) {
        other.m_data = nullptr;
        other.m_row = 0;
        other.m_col = 0;
    }
    Matrix& operator=(const Matrix& other) {
        delete[] m_data;
        m_row = other.m_row;
        m_col = other.m_col;
        alloc_data(m_row, m_col);
        memcpy(m_data, other.m_data, m_row * m_col * sizeof(data_type));
        return *this;
    }
    Matrix& operator=(Matrix&& other) noexcept {
        delete[] m_data;
        m_row = other.m_row;
        m_col = other.m_col;
        m_data = other.m_data;
        other.m_data = nullptr;
        other.m_row = 0;
        other.m_col = 0;
        return *this;
    }
    void set_zero() {
        std::memset(m_data, 0, m_row * m_col * sizeof(data_type));
    }

    size_type rows() const {
        return m_row;
    }
    size_type cols() const {
        return m_col;
    }
    size_type stride() const {
        return m_col;
    }
    data_type *data() {
        return m_data;
    }
    size_t data_size() {
        return m_row*m_col*sizeof(data_type);
    }

    // standard row major layout
    data_type& at(size_type i, size_type j) {
        return m_data[i*m_col + j];
    }
    const data_type& at(size_type i, size_type j) const {
        return m_data[i*m_col + j];
    }
};

/**
SyncVariable: a variable that you can wait on.

It supports two operations: one increments the variable,
and the other waits until the variable reaches the given value.

You should use this primitive to implement pipelined parallelism.

Two implementations are provided. One is generic, using C++11 mutexes
and condition variables, the other is Linux/GCC-specific but has less
overhead.
If you are interested in the Linux implementation, see "man 2 futex".
**/

#ifdef __linux__
static int
futex(volatile int *uaddr, int futex_op, int val,
     const struct timespec *timeout, int *uaddr2, int val3)
{
   return syscall(SYS_futex, uaddr, futex_op, val,
                  timeout, uaddr, val3);
}

class SyncVariable {
private:
    volatile int count = 0;

public:
    void increment() {
        __atomic_add_fetch(&count, 1, __ATOMIC_RELEASE);
        futex(&count, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
    }

    void wait_until(int value) {
        int v;
        while ((v = __atomic_load_n(&count, __ATOMIC_ACQUIRE)) < value)
            futex(&count, FUTEX_WAIT, v, nullptr, nullptr, 0);
    }
};
#else
class SyncVariable {
private:
    size_t count = 0;
    std::mutex mutex;
    std::condition_variable cond;

public:
    void increment() {
        std::lock_guard<std::mutex> lock(mutex);
        count++;
        cond.notify_all();
    }

    void wait_until(size_t value) {
        std::unique_lock<std::mutex> lock(mutex);
        while (count < value)
            cond.wait(lock);
    }
};
#endif

template<typename T, typename Distribution, typename RandomNumberGenerator>
static Matrix<T>
generate_matrix(size_t rows, size_t cols, Distribution& d, RandomNumberGenerator& g)
{
    Matrix<T> into(rows, cols);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            into.at(i, j) = d(g);
    return into;
}

class Timer {
private:
    clockid_t m_clock;
    struct timespec m_start_time;

public:
    Timer(clockid_t clock) : m_clock(clock) {
        clock_gettime(clock, &m_start_time);
    }
    uint64_t read() {
        struct timespec now;
        clock_gettime(m_clock, &now);
        uint64_t start_time_us = (uint64_t)m_start_time.tv_sec * 1000000 +
            m_start_time.tv_nsec / 1000;
        uint64_t now_us = (uint64_t)now.tv_sec * 1000000 +
            now.tv_nsec / 1000;
        return now_us - start_time_us;
    }
};

// (optional) BEGIN YOUR CODE HERE

#define BLOCK_SIZE 128

// (optional) END YOUR CODE HERE

static
void serial_sor(Matrix<float>& m, float c)
{
    for (size_t i = 1; i < m.rows(); i++) {
        for (size_t j = 1; j < m.cols(); j++) {
            m.at(i, j) = c * (m.at(i-1, j) + m.at(i, j-1));
        }
    }
}

#define BLOCK_SIZE 128

static
void parallel_sor(Matrix<float>& m, float c)
{
  SyncVariable syncVars[m.rows()];
  #pragma omp parallel for collapse(1)
  for (size_t p = 1; p < m.rows(); p += BLOCK_SIZE) {
    for (size_t j = 1; j < m.cols(); j++) {
      for (size_t pp = p; pp < std::min(m.rows(), p + BLOCK_SIZE); pp++) {
	if (p == 1) {
	  m.at(pp, j) = c * (m.at(pp - 1, j) + m.at(pp, j - 1));
	} else {
	  syncVars[int(p - BLOCK_SIZE)].wait_until(j);
	  m.at(pp, j) = c * (m.at(pp - 1, j) + m.at(pp, j - 1));
	}
      }
      syncVars[p].increment();
    }
  }
}

int main(int argc, const char** argv)
{
    size_t row1, col1;
    if (argc < 3) {
        std::cerr << "usage:" << argv[0] << " <ROW1> <COL1>" << std::endl;
        return 1;
    }
    row1 = std::stoul(argv[1]);
    col1 = std::stoul(argv[2]);

    std::mt19937_64 random_engine(123);
    std::normal_distribution<float> distribution{0, 1};

    const int NUM_ITERATIONS = 20;
    uint64_t sum_time = 0;
    uint64_t sum_time_squared = 0;

    // ignore the first 5 iterations as the processor warms up
    for (int i = 5; i < 5+NUM_ITERATIONS; i++) {
        Matrix<float> m = generate_matrix<float>(row1, col1, distribution, random_engine);
        float c = distribution(random_engine);

        Timer tm(CLOCK_MONOTONIC);

        // serial_sor(m, c);

        parallel_sor(m, c);

        uint64_t time = tm.read();
        if (i < 5)
            continue;
        std::cerr << "Iteration " << (i-5+1) << ": " << time << " us" << std::endl;
        sum_time += time;
        sum_time_squared += time * time;
    }

    double avg_time = ((double)sum_time/NUM_ITERATIONS);
    double avg_time_squared = ((double)sum_time_squared/NUM_ITERATIONS);
    double std_dev = sqrt(avg_time_squared - avg_time * avg_time);
    std::cerr << std::setprecision(0) << std::fixed;
    std::cerr << "Avg time: " << avg_time << " us" << std::endl;
    std::cerr << "Stddev: ±" << std_dev << " us" << std::endl;
    return 0;
}
