#include <RcppEigen.h>
#include "bvharprob.h"

// [[Rcpp::depends(RcppEigen)]]

//' Generate Multivariate Normal Random Vector
//' 
//' This function samples n x muti-dimensional normal random matrix.
//' 
//' @param num_sim Number to generate process
//' @param mu Mean vector
//' @param sig Variance matrix
//' @details
//' Consider \eqn{x_1, \ldots, x_n \sim N_m (\mu, \Sigma)}.
//' 
//' 1. Lower triangular Cholesky decomposition: \eqn{\Sigma = L L^T}
//' 2. Standard normal generation: \eqn{Z_{i1}, Z_{in} \stackrel{iid}{\sim} N(0, 1)}
//' 3. \eqn{Z_i = (Z_{i1}, \ldots, Z_{in})^T}
//' 4. \eqn{X_i = L Z_i + \mu}
//' 
//' This function does not care of \eqn{\mu}.
//' 
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  // for (int i = 0; i < dim; i++) {
  //   for (int j = 0; j < dim; j++) {
  //     if (sig(i, j) != sig(j, i)) {
  //       Rcpp::stop("'sig' must be a symmetric matrix.");
  //     }
  //   }
  // }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  res = standard_normal * sig.sqrt(); // epsilon(t) = Sigma^{1/2} Z(t)
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Multivariate Normal Random Vector using Cholesky Decomposition
//' 
//' This function samples n x muti-dimensional normal random matrix with using Cholesky decomposition.
//' 
//' @param num_sim Number to generate process
//' @param mu Mean vector
//' @param sig Variance matrix
//' @details
//' This function computes \eqn{\Sigma^{1/2}} by choleksy decomposition.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  // for (int i = 0; i < dim; i++) {
  //   for (int j = 0; j < dim; j++) {
  //     if (sig(i, j) != sig(j, i)) {
  //       Rcpp::stop("'sig' must be a symmetric matrix.");
  //     }
  //   }
  // }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(sig);
  Eigen::MatrixXd sig_sqrt = lltOfscale.matrixU(); // use upper because now dealing with row vectors
  res = standard_normal * sig_sqrt;
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Matrix Normal Random Matrix
//' 
//' This function samples one matrix gaussian matrix.
//' 
//' @param mat_mean Mean matrix
//' @param mat_scale_u First scale matrix
//' @param mat_scale_v Second scale matrix
//' @details
//' Consider s x m matrix \eqn{Y_1, \ldots, Y_n \sim MN(M, U, V)} where M is s x m, U is s x s, and V is m x m.
//' 
//' 1. Lower triangular Cholesky decomposition: \eqn{U = P P^T} and \eqn{V = L L^T}
//' 2. Standard normal generation: s x m matrix \eqn{Z_i = [z_{ij} \sim N(0, 1)]} in row-wise direction.
//' 3. \eqn{Y_i = M + P Z_i L^T}
//' 
//' This function only generates one matrix, i.e. \eqn{Y_1}.
//' 
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, 
                                Eigen::Map<Eigen::MatrixXd> mat_scale_u, 
                                Eigen::Map<Eigen::MatrixXd> mat_scale_v) {
  int num_rows = mat_mean.rows();
  int num_cols = mat_mean.cols();
  if (mat_scale_u.rows() != mat_scale_u.cols()) {
    Rcpp::stop("Invalid 'mat_scale_u' dimension.");
  }
  if (num_rows != mat_scale_u.rows()) {
    Rcpp::stop("Invalid 'mat_scale_u' dimension.");
  }
  if (mat_scale_v.rows() != mat_scale_v.cols()) {
    Rcpp::stop("Invalid 'mat_scale_v' dimension.");
  }
  if (num_cols != mat_scale_v.rows()) {
    Rcpp::stop("Invalid 'mat_scale_v' dimension.");
  }
  Eigen::LLT<Eigen::MatrixXd> lltOfscaleu(mat_scale_u);
  Eigen::LLT<Eigen::MatrixXd> lltOfscalev(mat_scale_v);
  // Cholesky decomposition (lower triangular)
  Eigen::MatrixXd chol_scale_u = lltOfscaleu.matrixL();
  Eigen::MatrixXd chol_scale_v = lltOfscalev.matrixL();
  // standard normal
  Eigen::MatrixXd mat_norm(num_rows, num_cols);
  // Eigen::MatrixXd res(num_rows, num_cols, num_sim);
  Eigen::MatrixXd res(num_rows, num_cols);
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      mat_norm(i, j) = norm_rand();
    }
  }
  res = mat_mean + chol_scale_u * mat_norm * chol_scale_v.transpose();
  return res;
}

//' Generate Lower Triangular Matrix of IW
//' 
//' This function generates \eqn{A = L (Q^{-1})^T}.
//' 
//' @param mat_scale Scale matrix of IW
//' @param shape Shape of IW
//' @details
//' This function is the internal function for IW sampling and MNIW sampling functions.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_iw_tri(Eigen::Map<Eigen::MatrixXd> mat_scale, double shape) {
  int dim = mat_scale.cols();
  if (shape <= dim - 1) {
    Rcpp::stop("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  // upper triangular bartlett decomposition
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
  // generate in row direction
  for (int i = 0; i < dim; i++) {
    // diagonal
    mat_bartlett(i, i) = sqrt(chisq_rand(shape - (double)i)); // qii^2 ~ chi^2(nu - i + 1)
    // upper triangular (j > i) ~ N(0, 1)
    for (int j = i + 1; j < dim; j++) {
      mat_bartlett(i, j) = norm_rand();
    }
  }
  // cholesky decomposition (lower triangular)
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(mat_scale);
  Eigen::MatrixXd chol_scale = lltOfscale.matrixL();
  // lower triangular
  Eigen::MatrixXd chol_res = chol_scale * mat_bartlett.inverse().transpose();
  return chol_res;
}

//' Generate Inverse-Wishart Random Matrix
//' 
//' This function samples one matrix IW matrix.
//' 
//' @param mat_scale Scale matrix
//' @param shape Shape
//' @details
//' Consider \eqn{\Sigma \sim IW(\Psi, \nu)}.
//' 
//' 1. Upper triangular Bartlett decomposition: m x m matrix \eqn{Q = [q_{ij}]} upper triangular with
//'     1. \eqn{q_{ii}^2 \chi_{\nu - i + 1}^2}
//'     2. \eqn{q_{ij} \sim N(0, 1)} with i < j (upper triangular)
//' 2. Lower triangular Cholesky decomposition: \eqn{\Psi = L L^T}
//' 3. \eqn{A = L (Q^{-1})^T}
//' 4. \eqn{\Sigma = A A^T \sim IW(\Psi, \nu)}
//' 
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_iw(Eigen::Map<Eigen::MatrixXd> mat_scale, double shape) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape);
  Eigen::MatrixXd res = chol_res * chol_res.transpose(); // dim x dim
  return res;
}

//' Generate Normal-IW Random Family
//' 
//' This function samples normal inverse-wishart matrices.
//' 
//' @param num_sim Number to generate
//' @param mat_mean Mean matrix of MN
//' @param mat_scale_u First scale matrix of MN
//' @param mat_scale Scale matrix of IW
//' @param shape Shape of IW
//' @details
//' Consider \eqn{(Y_i, \Sigma_i) \sim MIW(M, U, \Psi, \nu)}.
//' 
//' 1. Generate upper triangular factor of \eqn{\Sigma_i = C_i C_i^T} in the upper triangular Bartlett decomposition.
//' 2. Standard normal generation: s x m matrix \eqn{Z_i = [z_{ij} \sim N(0, 1)]} in row-wise direction.
//' 3. Lower triangular Cholesky decomposition: \eqn{U = P P^T}
//' 4. \eqn{A_i = M + P Z_i C_i^T}
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List sim_mniw(int num_sim,
                    Eigen::MatrixXd mat_mean, 
                    Eigen::Map<Eigen::MatrixXd> mat_scale_u, 
                    Eigen::Map<Eigen::MatrixXd> mat_scale, 
                    double shape) {
  int ncol_mn = mat_mean.cols();
  int nrow_mn = mat_mean.rows();
  int dim_iw = mat_scale.cols();
  if (dim_iw != mat_scale.rows()) {
    Rcpp::stop("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd chol_res(dim_iw, dim_iw);
  Eigen::MatrixXd mat_scale_v(dim_iw, dim_iw);
  // result matrices: bind in column wise
  Eigen::MatrixXd res_mn(nrow_mn, num_sim * ncol_mn); // [Y1, Y2, ..., Yn]
  Eigen::MatrixXd res_iw(dim_iw, num_sim * dim_iw); // [Sigma1, Sigma2, ... Sigma2]
  for (int i = 0; i < num_sim; i++) {
    chol_res = sim_iw_tri(mat_scale, shape);
    mat_scale_v = chol_res * chol_res.transpose();
    res_iw.block(0, i * dim_iw, dim_iw, dim_iw) = mat_scale_v;
    // MN(mat_mean, mat_scale_u, mat_scale_v)
    res_mn.block(0, i * ncol_mn, nrow_mn, ncol_mn) = sim_matgaussian(
      mat_mean, 
      mat_scale_u, 
      Eigen::Map<Eigen::MatrixXd>(mat_scale_v.data(), dim_iw, dim_iw)
    );
  }
  return Rcpp::List::create(
    Rcpp::Named("mn") = res_mn,
    Rcpp::Named("iw") = res_iw
  );
}

