#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sys/time.h>
//#include "jacobi.cpp"

using namespace std;

using Eigen::DecompositionOptions;
using Eigen::MatrixXd;
using Eigen::SimplicialLLT;
using Eigen::UpLoType;
using Eigen::VectorXd;

typedef Eigen::SparseMatrix<float> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<float> T;

// void buildProblem(std::vector<T> &coefficients, Eigen::VectorXd &b, int n);
// void saveAsBitmap(const Eigen::VectorXd &x, int n, const char *filename);

/**
https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
**/

SpMat generarMatrizDesdeArchivo(ifstream &);
void fillRandomVector(VectorXd v, int n);

// El programa requiere 2 parametros: un archivo de entrada y un valor p.-
int main(int argc, char *argv[])
{
    int n;

    cout << "Corriendo el programa..." << endl;

    // Argumentos : 0 - main - 1 archivo entrada
    if (argc != 3)
    {
        cout << "Error: faltan argumentos." << endl;
        return 1;
    }

    // Lee archivo entrada.-
    ifstream archivoDeEntrada(argv[1]);
    // int reps = argv[2];

    SpMat A = generarMatrizDesdeArchivo(archivoDeEntrada);
    n = A.cols();
    VectorXd x_ini;
    fillRandomVector(x_ini, n);
    // pair<VectorXd, VectorXd> x_jacobi = jacobi(A, b, reps, x_ini, x_direct);

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

    // Trasponemos la matriz para que sea no singular.-
    SpMat new_A = A.transpose() * A;
    return new_A;
}

void fillRandomVector(VectorXd v, int n)
{
    srand((unsigned)time(NULL));
    double a = rand() % 100 + 1;
    for (int i = 0; i < n; i++)
    {
        double b = rand() % 20 + 1;
        v[i] = b;
    }
}