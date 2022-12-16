#include <iostream>
#include <fstream>
#include <cstring>

using namespace std;

// El programa requiere 2 parametros: un archivo txt y la cantidad de repeticiones.-
int main(int argc, char *argv[])
{
    // Argumentos : 1 - main - 
    if (argc != 2)
    {
        cout << "Error: debe ingresar un parámetro." << endl;
        return 1;
    }

    int n = std::stoi(argv[1]);

    ofstream archivoSalida;
    archivoSalida.setf(ios::fixed, ios::floatfield);
    archivoSalida.precision(10);

    // Seteo los nombres de los archivos a guardar.-
    string archivoSalida_c = "../../tests/test_densidad_";
    archivoSalida_c += argv[1];
    archivoSalida_c += "_conexiones_2.txt";

    archivoSalida.open(archivoSalida_c);
    archivoSalida << 500 << endl;
    archivoSalida << n * 2500 - 2500 << endl;
    for (int i = 1; i <= n * 50; i++)
    {
        for (int j = 1; j <= n * 50; j++){
            if (i != j){
                archivoSalida << i << ' ' << j << endl;
            }
        }
    }
    archivoSalida.close();
}