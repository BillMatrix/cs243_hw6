#include <iostream>
#include <random>
#include <exception>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <string>
#include <cassert>
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
template<typename T, typename Distribution, typename RandomNumberGenerator>
static std::vector<T>
generate_vector(size_t size, Distribution& d, RandomNumberGenerator& g)
{
    std::vector<T> into(size);
    for (size_t i = 0; i < size; i++)
        into[i] = d(g);
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

/**
MatrixView: a view (submatrix) of an existing Matrix or MatrixView

Creating a MatrixView from a Matrix is cheap (only pointer adjustments)
*/
template<typename T>
class MatrixView
{
public:
    typedef T data_type;
    typedef size_t size_type;

private:
    size_type m_row, m_col;
    size_type m_stride;
    T *m_data;

public:
    MatrixView(const Matrix<data_type>& m) : m_row(m.rows()), m_col(m.cols()), m_stride(m.stride()), m_data((const_cast<Matrix<T>&>(m)).data()) {
    }
    MatrixView(data_type* data, size_t row, size_t col, size_t stride) : m_row(row), m_col(col), m_stride(stride), m_data(data) {
    }

    MatrixView(const MatrixView<data_type>& view, size_type i0, size_type j0, size_type rows, size_type cols) : m_row(rows), m_col(cols), m_stride(view.m_stride), m_data(const_cast<T*>(&view.at(i0, j0))) {
        assert(i0 < view.rows() && j0 < view.cols() &&
               i0+rows <= view.rows() && j0+cols <= view.cols());
    }

    size_type rows() const {
        return m_row;
    }
    size_type cols() const {
        return m_col;
    }

    // standard row major layout
    data_type& at(size_type i, size_type j) {
        return m_data[i*m_stride + j];
    }
    const data_type& at(size_type i, size_type j) const {
        return m_data[i*m_stride + j];
    }
};

/**
VectorView: a view (slice) of an existing std::vector or VectorView

Creating a VectorView from a vector is cheap (only pointer adjustments)
*/
template<typename T>
class VectorView
{
public:
    typedef T data_type;
    typedef size_t size_type;

private:
    size_type m_size;
    T *m_data;

public:
    VectorView(const std::vector<data_type>& m) : m_size(m.size()), m_data((const_cast<std::vector<T>&>(m)).data()) {
    }
    VectorView(data_type* data, size_t size) : m_size(size), m_data(data) {
    }
    VectorView(const VectorView<data_type>& view, size_type off, size_type size) : m_size(size), m_data(const_cast<T*>(&view.at(off))) {
    }

    size_type size() const {
        return m_size;
    }

    data_type& at(size_type i) {
        return m_data[i];
    }
    const data_type& at(size_type i) const {
        return m_data[i];
    }
};

// (optional) BEGIN YOUR CODE HERE
///////////////////////////////////////////////////////////////////
static Matrix<float>
serial_matmul(const MatrixView<float>& m1, const MatrixView<float>& m2)
{
    static const int BLOCK_SIZE = 16;

    assert(m1.cols() == m2.rows());
    Matrix<float> m3(m1.rows(), m2.cols());
    m3.set_zero();

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

    return m3;
}

static Matrix<float>
serial_cwise_unary_op(const MatrixView<float>& m, float(*op)(float))
{
    Matrix<float> m2(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); i ++) {
        for (size_t j = 0; j < m.cols(); j ++) {
            m2.at(i, j) = op(m.at(i, j));
        }
    }

    return m2;
}
static Matrix<float>
serial_cwise_add(const MatrixView<float>& m1, const MatrixView<float>& m2)
{
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    Matrix<float> m3(m1.rows(), m1.cols());

    for (size_t i = 0; i < m1.rows(); i ++) {
        for (size_t j = 0; j < m1.cols(); j ++) {
            m3.at(i, j) = m1.at(i, j) + m2.at(i, j);
        }
    }

    return m3;
}
static Matrix<float>
serial_cwise_mul(const MatrixView<float>& m1, const MatrixView<float>& m2)
{
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    Matrix<float> m3(m1.rows(), m1.cols());
    for (size_t i = 0; i < m1.rows(); i ++) {
        for (size_t j = 0; j < m1.cols(); j ++) {
            m3.at(i, j) = m1.at(i, j) * m2.at(i, j);
        }
    }
    return m3;
}

// broadcast_add_second: broadcast the vector and add it to the second dimension of m
static Matrix<float>
serial_broadcast_add_second(const MatrixView<float>& m, const VectorView<float>& v)
{
    assert(m.cols() == v.size());
    Matrix<float> m2(m.rows(), m.cols());

    for (size_t i = 0; i < m.rows(); i ++) {
        for (size_t j = 0; j < m.cols(); j ++) {
            m2.at(i, j) = m.at(i, j) + v.at(j);
        }
    }

    return m2;
}


static void print_mat(const MatrixView<float> mat){
    for (size_t i = 0; i < mat.rows(); i ++) {
      for (size_t j = 0; j < mat.cols(); j ++) {
	std::cout << mat.at(i, j) << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
}

static void print_mat(const Matrix<float> mat){
    for (size_t i = 0; i < mat.rows(); i ++) {
      for (size_t j = 0; j < mat.cols(); j ++) {
	std::cout << mat.at(i, j) << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
}


static void set_matrix_data(Matrix<float>& orig, const Matrix<float>& data, size_t i_start,
			    size_t j_start, size_t height, size_t width){
  for (size_t i = i_start; i < i_start + height; i++){
    for (size_t j = j_start; j < j_start + width; j++){
      orig.at(i, j) = data.at(i - i_start, j - j_start);
    }
  }
}


// (optional) END YOUR CODE HERE

static Matrix<float>
matmul(const MatrixView<float>& m1, const MatrixView<float>& m2)
{
    static const int BLOCK_SIZE = 16;

    assert(m1.cols() == m2.rows());
    Matrix<float> m3(m1.rows(), m2.cols());
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

    return m3;
}
static Matrix<float>
cwise_unary_op(const MatrixView<float>& m, float(*op)(float))
{
    Matrix<float> m2(m.rows(), m.cols());

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.rows(); i ++) {
        for (size_t j = 0; j < m.cols(); j ++) {
            m2.at(i, j) = op(m.at(i, j));
        }
    }

    return m2;
}
static Matrix<float>
cwise_add(const MatrixView<float>& m1, const MatrixView<float>& m2)
{
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    Matrix<float> m3(m1.rows(), m1.cols());

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m1.rows(); i ++) {
        for (size_t j = 0; j < m1.cols(); j ++) {
            m3.at(i, j) = m1.at(i, j) + m2.at(i, j);
        }
    }

    return m3;
}
static Matrix<float>
cwise_mul(const MatrixView<float>& m1, const MatrixView<float>& m2)
{
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    Matrix<float> m3(m1.rows(), m1.cols());

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m1.rows(); i ++) {
        for (size_t j = 0; j < m1.cols(); j ++) {
            m3.at(i, j) = m1.at(i, j) * m2.at(i, j);
        }
    }

    return m3;
}

// broadcast_add_second: broadcast the vector and add it to the second dimension of m
static Matrix<float>
broadcast_add_second(const MatrixView<float>& m, const VectorView<float>& v)
{
    assert(m.cols() == v.size());
    Matrix<float> m2(m.rows(), m.cols());

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.rows(); i ++) {
        for (size_t j = 0; j < m.cols(); j ++) {
            m2.at(i, j) = m.at(i, j) + v.at(j);
        }
    }

    return m2;
}
static float
sigmoid(float x)
{
    return 1/(1+std::exp(-x));
}




static
void kernel_lstm(const Matrix<float>& weights, const std::vector<float>& biases,
                 const Matrix<float>& x, const Matrix<float>& h, const Matrix<float>& c,
                 Matrix<float>& hprime, Matrix<float>& cprime)
{
    // unpack the packed weights
    size_t xsize = x.cols();
    size_t hsize = h.cols();

    //print_mat(weights);
    const MatrixView<float> Wf(weights, 0, 0, hsize, hsize);
    const MatrixView<float> Wi(weights, 0, hsize, hsize, hsize);
    const MatrixView<float> Wo(weights, 0, 2*hsize, hsize, hsize);
    const MatrixView<float> Wc(weights, 0, 3*hsize, hsize, hsize);
    const MatrixView<float> Uf(weights, hsize, 0, xsize, hsize);
    const MatrixView<float> Ui(weights, hsize, hsize, xsize, hsize);
    const MatrixView<float> Uo(weights, hsize, 2*hsize, xsize, hsize);
    const MatrixView<float> Uc(weights, hsize, 3*hsize, xsize, hsize);

    const VectorView<float> bf(biases, 0, hsize);
    const VectorView<float> bi(biases, hsize, hsize);
    const VectorView<float> bo(biases, 2*hsize, hsize);
    const VectorView<float> bc(biases, 3*hsize, hsize);

    //print_mat(Wf); print_mat(Wi); print_mat(Wo); print_mat(Wc); print_mat(Uf); print_mat(Ui);  print_mat(Uo);  print_mat(Uc);

    auto f = cwise_unary_op(broadcast_add_second(cwise_add(matmul(h, Wf), matmul(x, Uf)), bf), sigmoid);
    auto i = cwise_unary_op(broadcast_add_second(cwise_add(matmul(h, Wi), matmul(x, Ui)), bi), sigmoid);
    auto o = cwise_unary_op(broadcast_add_second(cwise_add(matmul(h, Wo), matmul(x, Uo)), bo), sigmoid);

    auto tmp = cwise_unary_op(broadcast_add_second(cwise_add(matmul(h, Wc), matmul(x, Uc)), bc), std::tanh);
    cprime = cwise_add(cwise_mul(f, c), cwise_mul(i, tmp));
    hprime = cwise_mul(o, cwise_unary_op(cprime, std::tanh));
}



static
void serial_lstm(const Matrix<float>& weights, const std::vector<float>& biases,
                 const Matrix<float>& x, const Matrix<float>& h, const Matrix<float>& c,
                 Matrix<float>& hprime, Matrix<float>& cprime)
{

  static const size_t COL_BLOCK = 32;
  static const size_t ROW_BLOCK = 32;

  size_t xsize = x.cols();
  size_t hsize = h.cols();
  size_t bsize = x.rows(); // batch size
  const VectorView<float> bf(biases, 0, hsize);
  const VectorView<float> bi(biases, hsize, hsize);
  const VectorView<float> bo(biases, 2*hsize, hsize);
  const VectorView<float> bc(biases, 3*hsize, hsize);

  for (size_t p = 0; p < hsize; p+= COL_BLOCK){
    size_t width = std::min(hsize - p, COL_BLOCK);
    std::cout << "width" << width << std::endl;
    const MatrixView<float> Wf_p(weights, 0, p, hsize, width);
    const MatrixView<float> Wi_p(weights, 0, hsize + p, hsize, width);
    const MatrixView<float> Wo_p(weights, 0, 2*hsize + p, hsize, width);
    const MatrixView<float> Wc_p(weights, 0, 3*hsize + p, hsize, width);
    const MatrixView<float> Uf_p(weights, hsize, p, xsize, width);
    const MatrixView<float> Ui_p(weights, hsize, hsize + p, xsize, width);
    const MatrixView<float> Uo_p(weights, hsize, 2*hsize + p, xsize, width);
    const MatrixView<float> Uc_p(weights, hsize, 3*hsize + p, xsize, width);
    const VectorView<float> bf_p(bf, p, width);
    const VectorView<float> bi_p(bi, p, width);
    const VectorView<float> bo_p(bo, p, width);
    const VectorView<float> bc_p(bc, p, width);

    for (size_t pp = 0; pp < bsize; pp += ROW_BLOCK){
      size_t height = std::min(bsize - pp, ROW_BLOCK);
      std::cout << "height" << height << std::endl;
      const MatrixView<float> h_pp(h, pp, 0, height, hsize);
      auto h_Wf_pp_p = serial_matmul(h_pp, Wf_p);
      auto h_Wi_pp_p = serial_matmul(h_pp, Wi_p);
      auto h_Wo_pp_p = serial_matmul(h_pp, Wo_p);
      auto h_Wc_pp_p = serial_matmul(h_pp, Wc_p);

      const MatrixView<float> x_pp(x, pp, 0, height, xsize);
      auto x_Uf_pp_p = serial_matmul(x_pp, Uf_p);
      auto x_Ui_pp_p = serial_matmul(x_pp, Ui_p);
      auto x_Uo_pp_p = serial_matmul(x_pp, Uo_p);
      auto x_Uc_pp_p = serial_matmul(x_pp, Uc_p);

      auto f_pp_p = serial_cwise_unary_op(serial_broadcast_add_second(serial_cwise_add(h_Wf_pp_p, x_Uf_pp_p), bf_p), sigmoid);
      auto i_pp_p = serial_cwise_unary_op(serial_broadcast_add_second(serial_cwise_add(h_Wi_pp_p, x_Ui_pp_p), bi_p), sigmoid);
      auto j_pp_p = serial_cwise_unary_op(serial_broadcast_add_second(serial_cwise_add(h_Wc_pp_p, x_Uc_pp_p), bc_p), std::tanh);

      const MatrixView<float> c_pp_p (c, pp, p, height, width);
      std::cout << "c_pp_p" << std::endl;
      auto cprime_pp_p = serial_cwise_add(serial_cwise_mul(f_pp_p, c_pp_p), serial_cwise_mul(i_pp_p, j_pp_p));
      std::cout << "cprime_pp_p" << std::endl;


      set_matrix_data(cprime, cprime_pp_p, pp, p, height, width);

      auto o_pp_p = serial_cwise_unary_op(serial_broadcast_add_second(serial_cwise_add(h_Wo_pp_p, x_Uo_pp_p), bo_p), sigmoid);
      auto hprime_pp_p = serial_cwise_mul(o_pp_p, serial_cwise_unary_op(cprime_pp_p, std::tanh));
      set_matrix_data(hprime, hprime_pp_p, pp, p, height, width);
    }
  }
}

static
void parallel_lstm(const Matrix<float>& weights, const std::vector<float>& biases,
                   const Matrix<float>& x, const Matrix<float>& h, const Matrix<float>& c,
                   Matrix<float>& hprime, Matrix<float>& cprime)
{
// BEGIN YOUR CODE HERE

// Write the parallel implementation of LSTM here

// END YOUR CODE HERE
}


int main(int argc, const char** argv)
{
    if (argc < 4) {
        std::cerr << "usage:" << argv[0] << " <B> <X> <H>" << std::endl;
        return 1;
    }
    size_t batchsize = std::stoul(argv[1]);
    size_t xsize = std::stoul(argv[2]);
    size_t hsize = std::stoul(argv[3]);

    //
    size_t method = std::stoul(argv[4]);
    //

    std::mt19937_64 random_engine;
    std::normal_distribution<float> distribution{0, 1};

    const int NUM_ITERATIONS = 20;
    uint64_t sum_time = 0;
    uint64_t sum_time_squared = 0;

    auto weights = generate_matrix<float>(hsize+xsize, 4*hsize, distribution, random_engine);
    auto biases = generate_vector<float>(4*hsize, distribution, random_engine);


    // ignore the first 5 iterations as the processor warms up
    for (int i = 0; i < 5+NUM_ITERATIONS; i++) {
        const Matrix<float> x = generate_matrix<float>(batchsize, xsize, distribution, random_engine);
        const Matrix<float> h = generate_matrix<float>(batchsize, hsize, distribution, random_engine);
        const Matrix<float> c = generate_matrix<float>(batchsize, hsize, distribution, random_engine);


        Timer tm(CLOCK_MONOTONIC);

	if (method == 1){
	  Matrix<float> hprime(batchsize, hsize);
	  Matrix<float> cprime(batchsize, hsize);
	  kernel_lstm(weights, biases, x, h, c, hprime, cprime);
	}
	else if (method == 2){
	  Matrix<float> hprime_(batchsize, hsize);
	  Matrix<float> cprime_(batchsize, hsize);
	  serial_lstm(weights, biases, x, h, c, hprime_, cprime_);
	}
	else{
	  Matrix<float> hprime__(batchsize, hsize);
	  Matrix<float> cprime__(batchsize, hsize);
	  parallel_lstm(weights, biases, x, h, c, hprime__, cprime__);
	}
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
