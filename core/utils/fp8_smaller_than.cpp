/* This is a copy-paste of the string used to define cpp implementation.
 * Terrible in terms of software maintenance, but including external files
 * directly on python code was not working and now at least I can test C++
 * code separately (just need to copy paste any changes in here...)*/
bool fp8_smaller_than(int num1, int num2, int _vectorisation_idx){
    unsigned char sign1, exp1, abs_val1, sign2, exp2, abs_val2;
    bool is_normal1, is_normal2, result;

    const char EXP_WIDTH = 4;
    const char FRAC_WIDTH = 3;

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
