#include <Eigen/Dense>

using namespace std;
using Eigen::VectorXd;

void normalizar(VectorXd x, int n)
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
}