#include <iostream>
#include <omp.h>
using namespace std;
void parallelBubbleSort(int arr[], int n)
{
    bool swapped;
#pragma omp parallel default(none) shared(arr, n, swapped)
    {
        do
        {
            swapped = false;
#pragma omp for
            for (int i = 0; i < n - 1; ++i)
            { // i is pre-increment operator
                if (arr[i] > arr[i + 1])
                {
                    swap(arr[i], arr[i + 1]);
                    swapped = true;
                }
            }
        } while (swapped);
    }
}
void sequentialBubbleSort(int arr[], int n)
{
    bool swapped;

    do
    {
        swapped = false;

        for (int i = 0; i < n - 1; ++i)
        { // i is pre-increment operator
            if (arr[i] > arr[i + 1])
            {
                swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
    } while (swapped);
}

int main()
{
    int n = 10;
    int arr[n];
    for (int i = 0; i < n; ++i)
    {
        arr[i] = rand() % 1000;
    }
    cout << "Unsorted array :" << endl;
    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";
    }
    cout << endl;
    double start, end;
    start = omp_get_wtime();
    parallelBubbleSort(arr, n);
    end = omp_get_wtime();
    cout << "Sorted array using parallel bubble sort:" << endl;

    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";
    }

    cout << "\nTotal time taken by parallel algorithm :" << end - start << " in seconds";

    
    double starting, ending;
    starting = omp_get_wtime();
    sequentialBubbleSort(arr,n);
    ending = omp_get_wtime();
    cout << "\nSorted array using sequential bubble sort:" << endl;

    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";
    }

    cout << "\nTotal time taken by sequential algorithm :" << ending - starting << " in seconds";
    return 0;
}
