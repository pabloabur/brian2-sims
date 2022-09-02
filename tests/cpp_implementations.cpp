#include <iostream>
#include <bitset>

#include "fp8_add.h"
#include "fp8_multiply.h"

int test_multiply(){
    return 0;
}

int test_add(){
    return 0;
}

int main(){
    int a=8;
    int b=12;
    int res= fp8_add(a, b);

    std::cout << std::bitset<8>(res) << std::endl;

    return 0;
}
