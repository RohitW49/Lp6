#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void merge(vector<int> &arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
    {
        L[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++)
    {
        R[j] = arr[mid + 1 + j];
    }

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSortSequential(vector<int> &arr, int left, int right)
{
    if (left >= right)
        return;
    int mid = left + (right - left) / 2;
    mergeSortSequential(arr, left, mid);
    mergeSortSequential(arr, mid + 1, right);

    merge(arr, left, mid, right);
}

void mergeSortParallel(vector<int> &arr, int left, int right)
{
    if (left >= right)
        return;
    int mid = left + (right - left) / 2;
#pragma omp parallel sections
    {
#pragma omp section
        mergeSortSequential(arr, left, mid);
#pragma omp section
        mergeSortSequential(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main()
{
    int n = 20;
    vector<int> arr(n);
    for (int i = 0; i < n; ++i)
    {
        arr[i] = rand() % 1000;
    }
    cout << "Unsorted array :"<<endl;
    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";
    }
    cout << endl;
    double start,end;
    start=omp_get_wtime();
    mergeSortSequential(arr, 0, n - 1);
    end=omp_get_wtime();
    cout << "Sorted array using sequential algorithm :"<<endl;

    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";    
    }

    cout<<"\nTotal time taken for sequential :"<<end-start<<" in seconds";
    start=omp_get_wtime();
    mergeSortParallel(arr, 0, n - 1);
    end=omp_get_wtime();
    cout << "\nSorted array using parallel algorithm :"<<endl;

    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";    
    }

    cout<<"\nTotal time taken for parallel :"<<end-start<<" in seconds";
    return 0;
}

