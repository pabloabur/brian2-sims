"""
Created on Thu Feb 3 7:28:46 2022

@author: pablo
"""
import unittest
from teili.tools.misc import fp8_multiply, fp8_add
import numpy as np

class TestOrca(unittest.TestCase):

    def test_addition(self):
        num1 = np.array([8]) #0 0001 000 => (1.0)2 * 2^-6 => (0.015625)10
        num2 = np.array([12]) #0 0001 100 => (1.100)2 * 2^-6 => (0.0234375)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (0.0390625)10
        # Final result: 0 0010 010 => (1.010)2 * 2^-5 => (0.0390625)10
        self.assertEqual(res, 18, 'failed summation with carry on exponent')

        num1 = np.array([28]) #0 0011 100 => (1.100)2 * 2^-4 => (0.09375)10
        num2 = np.array([12]) #0 0001 100 => (1.100)2 * 2^-6 => (0.0234375)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (0.1171875)10
        # Final result: 0 0011 111 => (1.111)2 * 2^-4 => (0.1171875)10
        self.assertEqual(res, 31, 'failed summation with different exponent')

        num1 = np.array([8, 28])
        num2 = np.array([12, 12])
        res = fp8_add(num1, num2, np.array([0, 1]))
        self.assertEqual(res.tolist(), [18, 31], 'failed array summation')

        num1 = np.array([20]) #0 0010 100 => (1.100)2 * 2^-5 => (0.046875)10
        num2 = np.array([139]) #1 0001 011 => -(1.011)2 * 2^-6 => -(0.021484375)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (0.025390625)10
        # Final result: 0 0001 101 => (1.101)2 * 2^-6 => (0.025390625)10
        self.assertEqual(res, 13, 'failed subtraction without rounding')

        num1 = np.array([232]) #1 1101 000 => -(1.000)2 * 2^6 => -(64)10
        num2 = np.array([239]) #1 1101 111 => -(1.111)2 * 2^6 => -(120)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (184)10
        # Final result: 1 1110 100 => (1.100)2 * 2^7 => -(192)10
        self.assertEqual(res, 244, 'failed addition with rounding')

        num1 = np.array([238]) #1 1101 110 => -(1.110)2 * 2^6 => -(112)10
        num2 = np.array([239]) #1 1101 111 => -(1.111)2 * 2^6 => -(120)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (232)10
        # Final result: 1 1110 110 => (1.110)2 * 2^7 => -(224)10
        self.assertEqual(res, 246, 'failed addition with rounding')

        num1 = np.array([10]) #0 0001 010 => (1.010)2 * 2^-6 => (0.01953125)10
        num2 = np.array([138]) #1 0001 010 => -(1.010)2 * 2^-6 => -(0.01953125)10
        res = fp8_add(num1, num2, np.array([0]))
        self.assertEqual(res, 0, 'failed cancellation of normal values')

        num1 = np.array([10]) #0 0001 010 => (1.010)2 * 2^-6 => (0.01953125)10
        num2 = np.array([1]) #0 0000 001 => (0.001)2 * 2^-6 => (0.001953125)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (0.021484375)10
        # Result is exact, i.e. (1.011)2 * 2^-6 => (0.021484375)10
        self.assertEqual(res, 11,
            'failed summation of normal with subnormal value')

        num1 = np.array([8]) #0 0001 000 => (1.000)2 * 2^-6 => (0.015625)10
        num2 = np.array([135]) #1 0000 111 => -(0.111)2 * 2^-6 => -(0.013671875)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (0.001953125)10
        # Result is exact, i.e. (0.001)2 * 2^-6 => (0.001953125)10
        self.assertEqual(res, 1, 'failed subtraction of normal with subnormal value')

        num1 = np.array([3]) #0 0000 011 => (0.011)2 * 2^-6 => (0.005859375)10
        num2 = np.array([1]) #0 0000 001 => (0.001)2 * 2^-6 => (0.001953125)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (0.0078125)10
        # Result is exact, i.e. (0.100)2 * 2^-6 => (0.0078125)10
        self.assertEqual(res, 4, 'failed summation of subnormal values')

        num1 = np.array([16]) #0 0010 000 => (1.000)2 * 2^-5 => (0.03125)10
        num2 = np.array([143]) #1 0001 111 => -(1.111)2 * 2^-6 => -(0.029296875)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (0.001953125)10
        # Result is exact, i.e. (0.001)2 * 2^-6 => (0.001953125)10
        self.assertEqual(res, 1,
            'failed to return subnormal value with renormalization')

        num1 = np.array([68]) #0 1000 100 => (1.100)2 * 2^1 => (3)10
        num2 = np.array([44]) #0 0101 100 => (1.100)2 * 2^-2 => (0.375)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (3.375)10
        # Result is (1.110)2 * 2^1 => (3.5)10
        self.assertEqual(res, 70, 'failed summation with round to nearest')

        num1 = np.array([145]) #1 0010 001 => -(1.001)2 * 2^-5 => -(0.03515625)10
        num2 = np.array([56]) #0 0111 000 => (1.000)2 * 2^0 => (1)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (0.96484375)10
        # Final result => (1.111)2 * 2^-1 => (0.9375)10
        self.assertEqual(res, 55, 'failed multiple roudings')

        num1 = np.array([0x77]) #0 1110 111 => (1.111)2 * 2^7 => (240)10
        num2 = np.array([0x71]) #0 1110 001 => (1.001)2 * 2^7 => (144)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (384)10
        # Final result is 0 1111 100 => (1.100)2 * 2^8 => (384)10
        self.assertEqual(res, 124, 'failed to return value close to overflow')

        num1 = np.array([0x78]) #0 1111 000 => (1.000)2 * 2^8 => (256)10
        num2 = np.array([0x77]) #0 1110 111 => (1.111)2 * 2^7 => (240)10
        res = fp8_add(num1, num2, np.array([0]))
        # Exact result is (496)10
        # Final result is 0 1111 111 => (1.111)2 * 2^8 => (480)10
        self.assertEqual(res, 127, 'failed to return overflow')

    def test_multiplication(self):
        num1 = np.array([62]) #0 0111 110 => (1.110)2 * 2^0 => (1.75)10
        num2 = np.array([74]) #0 1001 010 => (1.010)2 * 2^2 => (5)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (8.75)10
        # Final result => (1.001)2 * 2^3 => (9)10
        self.assertEqual(res, 81, 'failed to round result')

        num1 = np.array([53]) #0 0110 101 => (1.101)2 * 2^-1 => (0.8125)10
        num2 = np.array([18]) #0 0010 010 => (1.010)2 * 2^-5 => (0.0390625)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (0.03173828125)10
        # Final result => (1.000)2 * 2^-5 => (0.03125)10
        self.assertEqual(res, 16, 'failed rounding subnormal to normal value')

        num1 = np.array([62, 53])
        num2 = np.array([74, 18])
        res = fp8_multiply(num1, num2, np.array([0, 1]))
        self.assertEqual(res.tolist(), [81, 16], 'failed array multiplication')

        num1 = np.array([48]) #0 0110 000 => (1.000)2 * 2^-1 => (0.5)10
        num2 = np.array([52]) #0 0110 100 => (1.100)2 * 2^-1 => (0.75)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (0.375)10
        # Final result => (1.100)2 * 2^-2 => (0.375)10
        self.assertEqual(res, 44, 'failed simple multiplication')

        num1 = np.array([176]) #1 0110 000 => -(1.000)2 * 2^-1 => -(0.5)10
        num2 = np.array([52]) #0 0110 100 => (1.100)2 * 2^-1 => (0.75)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is -(0.375)10
        # Final result => -(1.100)2 * 2^-2 => -(0.375)10
        self.assertEqual(res, 172, 'failed multiplication with negative value')

        num1 = np.array([176]) #1 0110 000 => -(1.000)2 * 2^-1 => -(0.5)10
        num2 = np.array([180]) #1 0110 100 => -(1.100)2 * 2^-1 => -(0.75)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (0.375)10
        # Final result => (1.100)2 * 2^-2 => (0.375)10
        self.assertEqual(res, 44, 'failed multiplication with negative value')

        num1 = np.array([63]) #0 0111 111 => (1.111)2 * 2^0 => (1.875)10
        num2 = np.array([63]) #0 0111 111 => (1.111)2 * 2^0 => (1.875)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (3.515625)10
        # Final result => (1.110)2 * 2^1 => (3.5)10
        self.assertEqual(res, 70, 'failed multiplication of same values')

        num1 = np.array([16]) #0 0010 000 => (1.000)2 * 2^-5 => (0.03125)10
        num2 = np.array([1]) #0 0000 001 => (0.001)2 * 2^-6 => (0.001953125)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (6.103515625e-05)10
        # Final result is 0
        self.assertEqual(res, 0, 'failed to return small result as zero')

        num1 = np.array([135]) #1 0000 111 => -(0.111)2 * 2^-6 => -(0.013671875)10
        num2 = np.array([7]) #0 0000 111 => (0.111)2 * 2^-6 => (0.013671875)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is -(0.000186920166015625)10
        # Final result is 0
        self.assertEqual(res, 0, 'failed to multiply subnormal values')

        num1 = np.array([63]) # => (1.111)2 * 2^0 => (1.875)10
        num2 = np.array([7]) # => (0.111)2 * 2^-6 => (0.013671875)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (0.025634765625)10
        # Final result => (1.101)2 * 2^-6 => (0.025390625)10
        self.assertEqual(res, 13,
            'failed to return normal value after subnormal operation')

        num1 = np.array([63]) #0 0111 111 => (1.111)2 * 2^0 => (1.875)10
        num2 = np.array([1]) #0 0000 001 => (0.001)2 * 2^-6 => (0.001953125)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (0.003662109375)10
        # Final result => (0.010)2 * 2^-6 => (0.00390625)10
        self.assertEqual(res, 2,
            'failed to return subnormal value after subnormal operation')

        num1 = np.array([0x77]) #0 1110 111 => (1.111)2 * 2^7 => (240)10
        num2 = np.array([0x71]) #0 1110 001 => (1.001)2 * 2^7 => (144)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (34560)10
        # Final result is 0 1111 111 => (1.111)2 * 2^8 => (480)10
        self.assertEqual(res, 127, 'failed to return overflow')

        num1 = np.array([0x5F]) # 0 1011 111 => (1.111)2 * 2^4 => (30)10
        num2 = np.array([0x51]) # 0 1010 001 => (1.001)2 * 2^3 => (9)10
        res = fp8_multiply(num1, num2, np.array([0]))
        # Exact result is (270)10
        # Final result => 0 1111 000 => (1.000)2 * 2^8 => (256)10
        self.assertEqual(res, 120, 'failed to return value close to overflow')


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
