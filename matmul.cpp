#include <iostream>
#include <random>
#include <exception>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <string>
#include <time.h>

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

static
void serial_matmul(const Matrix<float>& m1, const Matrix<float>& m2, Matrix<float>& m3)
{
    for (size_t i = 0; i < m1.rows(); i++) {
        for (size_t j = 0; j < m2.cols(); j++) {
            float acc = 0;
            for (size_t k = 0; k < m2.rows(); k++)
                acc += m1.at(i, k) * m2.at(k, j);
            m3.at(i, j) = acc;
        }
    }
}

static
void naive_parallel_matmul(const Matrix<float>& m1, const Matrix<float>& m2, Matrix<float>& m3)
{
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m1.rows(); i++) {
        for (size_t j = 0; j < m2.cols(); j++) {
            float acc = 0;
            for (size_t k = 0; k < m2.rows(); k++)
                acc += m1.at(i, k) * m2.at(k, j);
            m3.at(i, j) = acc;
        }
    }
}

#define BLOCK_SIZE 128

static
void blocked_parallel_matmul(const Matrix<float>& m1, const Matrix<float>& m2, Matrix<float>& m3)
{
    m3.set_zero();

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m1.rows(); i += BLOCK_SIZE) {
        for (size_t j = 0; j < m2.cols(); j += BLOCK_SIZE) {
            for (size_t k = 0; k < m2.rows(); k ++) {
                for (size_t ii = i; ii < std::min(m1.rows(), i+BLOCK_SIZE); ii++) {
                    for (size_t jj = j; jj < std::min(m2.cols(), j+BLOCK_SIZE); jj++) {
                        m3.at(ii, jj) += m1.at(ii, k) * m2.at(k, jj);
                    }
                }
            }
        }
    }
}

int main(int argc, const char** argv)
{
    size_t row1, col1, col2;
    if (argc < 4) {
        std::cerr << "usage:" << argv[0] << " <ROW1> <COL1> <COL2>" << std::endl;
        return 1;
    }
    row1 = std::stoul(argv[1]);
    col1 = std::stoul(argv[2]);
    col2 = std::stoul(argv[3]);

    std::mt19937_64 random_engine;
    std::normal_distribution<float> distribution{0, 1};

    const int NUM_ITERATIONS = 20;
    uint64_t sum_time = 0;
    uint64_t sum_time_squared = 0;

    // ignore the first 5 iterations as the processor warms up
    for (int i = 5; i < 5+NUM_ITERATIONS; i++) {
        Matrix<float> m1 = generate_matrix<float>(row1, col1, distribution, random_engine);
        Matrix<float> m2 = generate_matrix<float>(col1, col2, distribution, random_engine);
        Matrix<float> result(m1.rows(), m2.cols());

        Timer tm(CLOCK_MONOTONIC);

        serial_matmul(m1, m2, result);
        // naive_parallel_matmul(m1, m2, result);
        // blocked_parallel_matmul(m1, m2, result);

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
