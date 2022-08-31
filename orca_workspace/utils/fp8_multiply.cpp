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

int fp8_multiply(int num1, int num2, int _vectorisation_idx){
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
