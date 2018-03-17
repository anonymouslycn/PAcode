
#include <iostream>
#define ASCENDING true
using namespace std;

void compare(int *arr, int a, int b, bool dir) {
    if (dir == (arr[a] > arr[b])) {
        int temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
    }
}

void halfCleaner(int *arr, int group ,int length, bool dir, int flip) {
    for(int a = 0; a< group ; a++) {
        int begin = a * length;
        int delta;
        if (flip==1) delta = length - 1;
        else         delta = length/2;
        
        for(int i = 0;i<length/2;i++) {
            compare(arr, begin + i , begin + delta - flip * i, dir);
        }
    }
}

void bitonicSort(int *arr, int length, bool dir) {
    
    for(int i = 2; i <= length ; i <<= 1) {
        for(int j = i ; j > 1; j>>= 1) {
            halfCleaner(arr, length/j, j, dir,(j==i?1:-1));
        }
    }
}

int main(int argc, const char * argv[]) {
    
    int num = 32;
    int input[num];
    
    cout<<"input: ";
    srand((unsigned)time(NULL));
    for(int i = 0;i<num;i++) {
        input[i] = rand() % 100 +1;
        cout<<input[i]<<' ';
    }
    cout<<endl;
    
    bitonicSort(input, num, 1);

    cout<<"output: ";
    for(int i = 0;i<num;i++) {
        cout<<input[i]<<' ';
    }
    cout<<endl;
    
    return 0;
}

