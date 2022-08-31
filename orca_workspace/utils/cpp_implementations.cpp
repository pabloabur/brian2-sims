#include <iostream>
#include <bitset>

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

int fp8_multiply(int num1, int num2){
    unsigned char sign1, exp1, abs_val1, sign2, exp2, abs_val2;
    unsigned char int_repr1_abs, int_repr2_abs, aux_int_repr, carry;
    unsigned char num_leading_zero1=0, num_leading_zero2=0;
    unsigned char result, result_sign, result_exp, trunc_result, result_abs;
    unsigned char guard_bit, sticky_bit, round_factor;
    bool is_normal1, is_normal2;

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

    result_sign = sign1 ^ sign2;

    // Multiplication with FRAC_WIDTH+1 LSBs. In hardware, aligning after result
    // is obtained when subnormal values are involved gives us time to calculate
    // multiplication and number of leading zeros in parallel.
    int_repr1_abs = (is_normal1 << FRAC_WIDTH) | (abs_val1 & FRAC_MASK);
    int_repr2_abs = (is_normal2 << FRAC_WIDTH) | (abs_val2 & FRAC_MASK);
    // result occupies N_BITS bits in the form of cnr1r2...r6
    result = int_repr1_abs * int_repr2_abs;

    // Code to extract number of leading bits
    if (int_repr1_abs==0){
        num_leading_zero1 = N_BITS;
    }else{
        aux_int_repr = int_repr1_abs;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero1+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero1+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero1+=1;}
    }

    // Normalization of result is done after multiplication, according
    // to leading zeros of multiplicands. Note that if multiplicand is normal,
    // no need to shift result e.g. 1.000*0.001=1000 => 1000>>2 = 100000
    num_leading_zero1 = num_leading_zero1 - (SIGN_WIDTH + EXP_WIDTH - is_normal1);
    if (!is_normal1) result <<= num_leading_zero1;

    // Code to extract number of leading bits
    if (int_repr2_abs==0){
        num_leading_zero2 = N_BITS;
    }else{
        aux_int_repr = int_repr2_abs;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero2+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero2+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero2+=1;}
    }

    num_leading_zero2 = num_leading_zero2 - (SIGN_WIDTH + EXP_WIDTH - is_normal2);
    if (!is_normal2) result <<= num_leading_zero2;
    carry = result >> (N_BITS-1);

    // This represents biased exponent - 1. Implicit normal bit is added
    // afterwards to correctly pack the result
    result_exp = (exp1 - is_normal1 
                  + exp2 - is_normal2 
                  - num_leading_zero1 - num_leading_zero2
                  + (1 - BIAS) + carry);

    // In hardware, a slightly larger exponent range (herein 2 extra bits) to
    // identify underflows is needed.
    if((result_exp >> (EXP_WIDTH + 1)) != 0){
        result >>= -result_exp;
        // Note that no sticky bits are computed from eliminated bits
        result_exp = EMIN;
    }

    guard_bit = (result >> (GUARD_WIDTH-1+carry)) & 1;
    sticky_bit = ((result << (N_BITS-GUARD_WIDTH+1-carry)) & REPR_MASK) != 0;

    // MSB of truncated result is included so that it is added to exponent,
    // which was calculated as the final value minus 1
    trunc_result = result >> (GUARD_WIDTH+carry);
    round_factor = guard_bit & (trunc_result|sticky_bit);
    trunc_result = trunc_result + round_factor;

    result_abs = (result_exp<<FRAC_WIDTH) + trunc_result;
    // Dealing with overflow. Note that sign bit is used for comparison, so no
    // extra space would be needed in hardware
    if (result_abs > (1 << (N_BITS-1)) - 1) result_abs = (1 << (N_BITS-1)) - 1;

    // Note that negative zeros are not return from operations
    if (result_abs==0) result_sign = 0;

    return (result_sign << EXP_WIDTH+FRAC_WIDTH) + result_abs;
}

int fp8_add(int num1, int num2){
    unsigned char num1_sign, num1_exp, num1_abs, num1_normal;
    unsigned char num2_sign, num2_exp, num2_abs, num2_normal;
    unsigned char sign1, exp1, abs_val1, is_normal1;
    unsigned char sign2, exp2, abs_val2, is_normal2;
    unsigned char magnitude_factor, aux_val;
    unsigned char int_repr1_abs, int_repr2_abs, aligned_repr2, aux_int_repr;
    unsigned char sticky_bit, guard_bit, discarded_bits, round_factor;
    unsigned char result, result_sign, result_exp, trunc_result, result_abs;
    unsigned char carry, num_leading_zero=0, num_shifts, aux_shift;
    bool opposite_signs;

    // Code to extract relevant fields of the bitstring
    num1_sign = num1 >> (EXP_WIDTH+FRAC_WIDTH);
    num1_abs = num1 & 0x7F;
    num1_exp = num1_abs >> FRAC_WIDTH;
    num1_normal = num1_abs >= (1 << FRAC_WIDTH);

    // Code to extract relevant fields of the bitstring
    num2_sign = num2 >> (EXP_WIDTH+FRAC_WIDTH);
    num2_abs = num2 & 0x7F;
    num2_exp = num2_abs >> FRAC_WIDTH;
    num2_normal = num2_abs >= (1 << FRAC_WIDTH);

    // Use largest numbers as reference to simplify code
    if (num1_abs>num2_abs){
        sign1 = num1_sign; sign2 = num2_sign;
        exp1 = num1_exp; exp2 = num2_exp;
        abs_val1 = num1_abs; abs_val2 = num2_abs;
        is_normal1 = num1_normal; is_normal2 = num2_normal;
        result_sign = num1_sign;
    }else{
        sign1 = num2_sign; sign2 = num1_sign;
        exp1 = num2_exp; exp2 = num1_exp;
        abs_val1 = num2_abs; abs_val2 = num1_abs;
        is_normal1 = num2_normal; is_normal2 = num1_normal;
        result_sign = num2_sign;
    }

    opposite_signs = sign1 ^ sign2;
    // Note magnitude difference of normal and subnormal values, e.g. 1.0*2^-6
    // and 0.1*2^-6 have the same magnitude
    magnitude_factor = (exp1 - is_normal1
                        - (exp2 - is_normal2));

    // Get integer representation in the form of c,n,f1,f2...f6
    aux_val = ((abs_val1 << (EXP_WIDTH+SIGN_WIDTH))
               & REPR_MASK);
    int_repr1_abs = ((is_normal1 << (FRAC_WIDTH+GUARD_WIDTH))
                     | (aux_val >> (EXP_WIDTH+SIGN_WIDTH-GUARD_WIDTH)));
    aux_val = ((abs_val2 << (EXP_WIDTH+SIGN_WIDTH))
               & REPR_MASK);
    int_repr2_abs = ((is_normal2 << (FRAC_WIDTH+GUARD_WIDTH))
                     | (aux_val >> (EXP_WIDTH+SIGN_WIDTH-GUARD_WIDTH)));

    // Align smallest to largest operand's magnitude. Now smallest exponent
    // equals largest exponent
    aligned_repr2 = int_repr2_abs >> magnitude_factor;

    // Uses sticky bit to avoid losing all bits when aligning.
    // Guard bits are initialized as zero, so no bits are lost
    if (magnitude_factor <= GUARD_WIDTH){
        sticky_bit = 0;
    } else if (magnitude_factor >= GUARD_WIDTH+FRAC_WIDTH){
        // All relevant bits were discarded, so worst case scenario is considered
        sticky_bit = 1;
    } else if ((magnitude_factor > GUARD_WIDTH)
               && (magnitude_factor < GUARD_WIDTH+FRAC_WIDTH)){
        discarded_bits = int_repr2_abs << (N_BITS - magnitude_factor);
        sticky_bit = (discarded_bits & REPR_MASK) != 0;
    }
    
    if (opposite_signs)
        result = int_repr1_abs - (aligned_repr2|sticky_bit);
    else
        result = int_repr1_abs + (aligned_repr2|sticky_bit);

    // Code to extract number of leading bits
    if (result==0){
        num_leading_zero = N_BITS;
    }else{
        aux_int_repr = result;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero+=1;}
    }
    carry = result >> (N_BITS-1);

    // Note that bias exponent -1 is calculated so later on result
    // containing the implicit bit is added to it.
    result_exp = exp1 - is_normal1 + carry;

    // Subnormal result or requiring renormalization. Particularly
    // useful when dealing with underflows
    if (num_leading_zero >= 2){
        if (exp1-is_normal1 < num_leading_zero)
            num_shifts = exp1 - is_normal1;
        else
            // Shift subtracted by -1 to align MSB to normal bit
            num_shifts = num_leading_zero - 1;
        result_exp -= num_shifts;
        result <<= num_shifts;
    }

    aux_shift = GUARD_WIDTH + carry;
    trunc_result = result >> aux_shift;
    discarded_bits = (((result << (N_BITS-aux_shift)) & REPR_MASK)
                      >> (N_BITS - aux_shift));
    sticky_bit = discarded_bits & ((1 << (aux_shift-1)) - 1) != 0;
    guard_bit = discarded_bits & (1 << (aux_shift-1)) != 0;

    // Note that negative zeros are not return from operations
    if (abs_val1==abs_val2 && opposite_signs){
        result_exp = 0;
        result_sign = 0;
    }

    // Using LSB of trunc_result only. Guard bit together with LSB define
    // previous and next values (for round down and up, respectively)
    round_factor = guard_bit & (trunc_result | sticky_bit);
    trunc_result = trunc_result + round_factor;
    result_abs = (result_exp<<FRAC_WIDTH) + trunc_result;
    // Dealing with overflow. Note that sign bit is used for comparison, so no
    // extra space would be needed in hardware
    if (result_abs > (1<<(N_BITS-1)) - 1)
        result_abs = (1<<(N_BITS-1)) - 1;

    return (result_sign << EXP_WIDTH+FRAC_WIDTH) + result_abs;
}

bool fp8_smaller_than(int num1, int num2){
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
