/* This is a copy-paste of the string used to define cpp implementation.
 * Terrible in terms of software maintenance, but including external files
 * directly on python code was not working and now at least I can test C++
 * code separately (just need to copy paste any changes in here...)*/
int fp8_add(int num1, int num2, int _vectorisation_idx){
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

    const char EXP_WIDTH = 4;
    const char FRAC_WIDTH = 3;
    const char SIGN_WIDTH = 1;
    const char N_BITS = SIGN_WIDTH + EXP_WIDTH + FRAC_WIDTH;
    const char GUARD_WIDTH = 3;
    const char REPR_MASK =  (1<<N_BITS) - 1;
    // Smallest normal: 2^-(-BIAS+1)
    // Smallest subnormal: 2^(-BIAS+1)/2^(FRAC_WIDTH-1)
    // Biased representation of exponents, i.e. what is actually stored in hardware

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
