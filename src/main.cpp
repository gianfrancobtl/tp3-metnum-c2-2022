#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "iterative_methods.cpp"
#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sys/time.h>
#include <random>
#include <Eigen/SparseLU>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

SpMat generarMatrizDesdeArchivo(ifstream &, double);
VectorXd fillRandomVector(int);
SpMat generarD(SpMat, int);
VectorXd generarE(int);
double calcularGrado(SpMat, double);

// El programa requiere 2 parametros: un archivo txt y la cantidad de repeticiones.-
int main(int argc, char *argv[])
{
    int n, reps;
    SpMat A;
    VectorXd b, b_2, x_direct, x_eg;
    pair<VectorXd, VectorXd> x_jacobi, x_gauss_seidel;

    cout << "Corriendo el programa..." << endl;

    // Argumentos : 0 - main - 1 archivo entrada
    if (argc != 3)
    {
        cout << "Error: faltan argumentos." << endl;
        return 1;
    }

    // Lee archivo entrada:
    ifstream matrizDeEntrada(argv[1]);
    reps = std::stoi(argv[2]);

    // Seteamos un p por defecto de 0.75:
    double p = 0.75;

    // Generamos la matriz A:
    A = generarMatrizDesdeArchivo(matrizDeEntrada, p);

    // Generamos el vector b (lleno de unos):
    b = generarE(A.cols());

    // Generamos un vector aleatorio inicial para los métodos iterativos:
    VectorXd x_ini = fillRandomVector(A.cols());

    // ELIMINACION GAUSSIANA
    x_direct = eg(A, b);
    cout << "Resultado directo por eliminación gaussiana" << endl;
    cout << x_direct << endl;

    cout << "============================" << endl;

    // JACOBI
    x_jacobi = jacobi(A, b, reps, x_ini, x_direct);
    cout << "Resultado de Jacobi" << endl;
    cout << x_jacobi.first << endl;

    cout << "============================" << endl;

    // GAUSS SEIDEL
    x_gauss_seidel = gauss_seidel(A, b, reps, x_ini, x_direct);
    cout << "Resultado de Gauss Seidel" << endl;
    cout << x_gauss_seidel.first << endl;

    // ERROR JACOBI
    // cout << x_jacobi.second << endl;

    // ERROR GAUSS SEIDEL
    // cout << x_gauss_seidel.second << endl;

    // Fin de la ejecución
    return 0;
}

SpMat generarMatrizDesdeArchivo(ifstream &archivoDeEntrada, double p)
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
    SpMat W(n, n);

    // Generación de los "ingredientes" de A:
    W.setFromTriplets(tl.begin(), tl.end());

    SpMat I(W.cols(), W.cols());
    I.setIdentity();
    SpMat D = generarD(W, n);

    // Generación de A:
    SpMat A = I - (p * W * D);

    return A;
}

VectorXd fillRandomVector(int n)
{
    VectorXd v(n);
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
    {
        double b = rand() % 20 + 1;
        v[i] = b;
    }
    return v;
}

SpMat generarD(SpMat W, int n)
{
    SpMat D(n, n);
    std::vector<T> tl;
    tl.reserve(n);
    for (int j = 0; j < n; j++)
    {
        double valor = calcularGrado(W, j);
        if (valor == 0)
        {
            tl.push_back(T(j, j, 0.00));
        }
        else
        {
            tl.push_back(T(j, j, 1.00 / (double)valor));
        }
    }
    D.setFromTriplets(tl.begin(), tl.end());

    return D;
}

double calcularGrado(SpMat W, double j)
{
    double res = 0.00;
    for (int i = 0; i < W.cols(); i++)
    {
        res += W.coeff(i, j);
    }
    return res;
}

VectorXd generarE(int n)
{
    VectorXd e(n);

    for (int i = 0; i < n; i++)
    {
        e[i] = 1.00;
    }

    return e;
}