#define EXP_WIDTH 4
#define FRAC_WIDTH 3
#define FRAC_MASK (1<<FRAC_WIDTH) - 1
#define SIGN_WIDTH 1
#define N_BITS SIGN_WIDTH + EXP_WIDTH + FRAC_WIDTH
#define BIAS 7
#define GUARD_WIDTH 3
#define REPR_MASK (1<<N_BITS) - 1
// Smallest normal: 2^-(-BIAS+1)
// Smallest subnormal: 2^(-BIAS+1)/2^(FRAC_WIDTH-1)
// Biased representation of exponents, i.e. what is actually stored in hardware
#define EMIN 0

bool fp8_smaller_than(int num1, int num2, int _vectorisation_idx){
    unsigned char sign1, exp1, abs_val1, sign2, exp2, abs_val2;
    bool is_normal1, is_normal2, result;

    // Code to extract relevant fields of the bitstring
    sign1 = num1 >> (EXP_WIDTH+FRAC_WIDTH);
    abs_val1 = num1 & 0x7F;
    exp1 = abs_val1 >> FRAC_WIDTH;
    is_normal1 = abs_val1 >= (1 << FRAC_WIDTH);

    // Code to extract relevant fields of the bitstring
    sign2 = num2 >> (EXP_WIDTH+FRAC_WIDTH);
    abs_val2 = num2 & 0x7F;
    exp2 = abs_val2 >> FRAC_WIDTH;
    is_normal2 = abs_val2 >= (1 << FRAC_WIDTH);

    if (sign1 > sign2)
        result = true;
    else if (sign1 < sign2)
        result = false;
    else if (sign1 == sign2){
        if (abs_val1 < abs_val2)
            result = true;
        else
            result = false;
    }

    return result;
}

