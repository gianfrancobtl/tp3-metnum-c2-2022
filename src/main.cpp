#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "jacobi.cpp"
#include "gauss_seidel.cpp"
#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sys/time.h>
#include <random>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<float> T;

SpMat generarMatrizDesdeArchivo(ifstream &);
VectorXd generarVectorDesdeArchivo(ifstream &, int);
void fillRandomVector(VectorXd v, int n);

// El programa requiere 3 parametros: dos archivos txt (una matriz y un vector) y la cantidad de repeticiones.-
int main(int argc, char *argv[])
{
    int n, reps;
    SpMat A;
    VectorXd b;
    pair<VectorXd, VectorXd> x_jacobi, x_gauss_seidel;

    cout << "Corriendo el programa..." << endl;

    // Argumentos : 0 - main - 1 archivo entrada
    if (argc != 4)
    {
        cout << "Error: faltan argumentos." << endl;
        return 1;
    }

    // Lee archivo entrada.-
    ifstream matrizDeEntrada(argv[1]);
    ifstream vectorDeEntrada(argv[2]);
    reps = std::stoi(argv[3]);

    A = generarMatrizDesdeArchivo(matrizDeEntrada);
    n = A.cols();
    b = generarVectorDesdeArchivo(vectorDeEntrada, n);
    SpMat new_A = A.transpose() * A;

    cout << A;
    cout << new_A;
    cout << b << endl;

    VectorXd x_ini(n);
    fillRandomVector(x_ini, n);

    x_jacobi = jacobi(A, A.transpose() * b, reps, x_ini, b);
    x_gauss_seidel = gauss_seidel(A, A.transpose() * b, reps, x_ini, b);

    for (int i = 0; i < n; i++)
    {
        cout << x_jacobi.first[i] << "  " << x_gauss_seidel.first[i] << endl;
    }

    for (int i = 0; i < n; i++)
    {
        cout << x_jacobi.second[i] << "  " << x_gauss_seidel.second[i] << endl;
    }

    // Fin de la ejecución
    return 0;
}

SpMat generarMatrizDesdeArchivo(ifstream &archivoDeEntrada)
{
    // Variables n (cantidad de páginas) - k (cantidad de links) - p1 (pagina 1) - p2 (pagina 2).-
    int n, k, p1, p2;
    archivoDeEntrada >> n;
    archivoDeEntrada >> k;

    std::vector<T> tl;
    tl.reserve(k);
    if (archivoDeEntrada.is_open())
    {
        for (int i = 0; i < k; i++)
        {
            archivoDeEntrada >> p1;
            archivoDeEntrada >> p2;

            // Seteo del valor 1 a la posición [p2][p1]:
            tl.push_back(T(p2 - 1, p1 - 1, 1));
        }
    }

    // Nueva instancia del resultado;
    // Se setean la matriz W y la cantidad total de links entre las páginas.
    SpMat A(n, n);
    A.setFromTriplets(tl.begin(), tl.end());

    return A;
}

VectorXd generarVectorDesdeArchivo(ifstream &vectorDeEntrada, int n)
{
    double p;
    double val;
    vectorDeEntrada >> p;

    VectorXd b(n);

    if (vectorDeEntrada.is_open())
    {
        for (int i = 0; i < n; i++)
        {
            vectorDeEntrada >> val;

            // Seteo del valor 1 a la posición [p2][p1]:
            b[i] = val;
        }
    }
    return b;
}

void fillRandomVector(VectorXd v, int n)
{
    srand((unsigned)time(NULL));
    double a = rand() % 100;
    for (int i = 0; i < n; i++)
    {
        double b = rand() % 20 + 1;
        v[i] = b;
    }
}
