#include <Eigen/Dense>

using namespace std;
using Eigen::VectorXd;
typedef Eigen::SparseMatrix<double> SpMat;


VectorXd normalizar(VectorXd x, int n)
{
    double sum = 0.00;
    for (int i = 0; i < n; i++)
    {
        sum += x[i];
    }
    for (int j = 0; j < n; j++)
    {
        x[j] /= sum;
    }
    return x;
}

double summation (SpMat A, VectorXd x, int i, int low, int sup){
       double res = 0.00;
       for (int j = low; j <= sup; j++){
           res = res + (A.coeff(i, j) * x[j]);
           }
      return res;
    }