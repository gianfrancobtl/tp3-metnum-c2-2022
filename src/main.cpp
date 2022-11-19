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
#include<Eigen/SparseLU>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseLU;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

SpMat generarMatrizDesdeArchivo(ifstream &, double);
VectorXd generarVectorDesdeArchivo(ifstream &, int);
VectorXd fillRandomVector(VectorXd v, int n);
SpMat generarD(SpMat, int);
VectorXd generarE(int);
VectorXd generarZ(SpMat, int, double);
double calcularGrado(SpMat, double);

// El programa requiere 3 parametros: dos archivos txt (una matriz y un vector) y la cantidad de repeticiones.-
int main(int argc, char *argv[])
{
    int n, reps;
    SpMat A;
    VectorXd b, b_2, x_direct, x_eg;
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

    double p;
    vectorDeEntrada >> p;

    A = generarMatrizDesdeArchivo(matrizDeEntrada, p);
    n = A.cols();
    x_direct = generarVectorDesdeArchivo(vectorDeEntrada, n);
    b = generarE(n);
    

    VectorXd x_ini(n);
    x_ini = fillRandomVector(x_ini, n);

    x_jacobi = jacobi(A, b, reps, x_ini, x_direct);
    
    x_gauss_seidel = gauss_seidel(A, b, reps, x_ini, x_direct);



    // ELIMINACION GAUSSIANA (OJO QUE PUEDEN CAMBIAR A y b)
    SparseLU<SpMat, COLAMDOrdering<int> >   solver;
    // Compute the ordering permutation vector from the structural pattern of A
    solver.analyzePattern(A); 
    // Compute the numerical factorization 
    solver.factorize(A); 
    x_eg = solver.solve(b);
    x_eg = normalizar(x_eg, A.cols()); 

    
    // JACOBI 
    cout << x_jacobi.first << endl;

    cout << "============================" << endl;

    // ERROR JACOBI
    cout << x_jacobi.second << endl;

    cout << "============================" << endl;
    cout << "============================" << endl;

    // GAUSS SEIDEL
    cout << x_gauss_seidel.first << endl;

    cout << "============================" << endl;

    // ERROR GAUSS SEIDEL
    cout << x_gauss_seidel.second << endl;

    cout << "============================" << endl;
    cout << "============================" << endl;

    // ELIMINACION GAUSSIANA
    cout << x_eg << endl;

    cout << "============================" << endl;


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

VectorXd generarVectorDesdeArchivo(ifstream &vectorDeEntrada, int n)
{
    double val;

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

VectorXd fillRandomVector(VectorXd v, int n)
{
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

VectorXd generarZ(SpMat W, int n, double p)
{
    VectorXd z(n);

    for (int j = 0; j < n; j++)
    {
        double valor = calcularGrado(W, j);
        if (valor == 0)
        {
            z[j] = 1 / n;
        }
        else
        {
            z[j] = (1 - p) / n;
        }
    }
    return z;
}