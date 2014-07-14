using System.Collections.Generic;

namespace icp
{
    struct Matrix3x3
    {
        public float
            a, b, c,
            d, e, f,
            g, h, i;

        public Matrix3x3
        (
            float a, float b, float c,
            float d, float e, float f,
            float g, float h, float i
        )
        {
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
            this.e = e;
            this.f = f;
            this.g = g;
            this.h = h;
            this.i = i;
        }

        public static Matrix3x3 operator -(Matrix3x3 rhs)
        {
            return new Matrix3x3
            (
                -rhs.a,
                -rhs.b,
                -rhs.c,
                -rhs.d,
                -rhs.e,
                -rhs.f,
                -rhs.g,
                -rhs.h,
                -rhs.i
            );
        }

        public static Matrix3x3 operator +(Matrix3x3 lhs, Matrix3x3 rhs)
        {
            return new Matrix3x3
            (
                lhs.a + rhs.a,
                lhs.b + rhs.b,
                lhs.c + rhs.c,
                lhs.d + rhs.d,
                lhs.e + rhs.e,
                lhs.f + rhs.f,
                lhs.g + rhs.g,
                lhs.h + rhs.h,
                lhs.i + rhs.i
            );
        }

        public static Matrix3x3 operator -(Matrix3x3 lhs, Matrix3x3 rhs)
        {
            return new Matrix3x3
            (
                lhs.a - rhs.a,
                lhs.b - rhs.b,
                lhs.c - rhs.c,
                lhs.d - rhs.d,
                lhs.e - rhs.e,
                lhs.f - rhs.f,
                lhs.g - rhs.g,
                lhs.h - rhs.h,
                lhs.i - rhs.i
            );
        }

        public static Matrix3x3 operator *(float lhs, Matrix3x3 rhs)
        {
            return new Matrix3x3
            (
                lhs * rhs.a,
                lhs * rhs.b,
                lhs * rhs.c,
                lhs * rhs.d,
                lhs * rhs.e,
                lhs * rhs.f,
                lhs * rhs.g,
                lhs * rhs.h,
                lhs * rhs.i
            );            
        }

        public static Matrix3x3 operator *(Matrix3x3 lhs, float rhs)
        {
            return rhs * lhs;
        }

        public static Matrix3x3 operator *(Matrix3x3 lhs, Matrix3x3 rhs)
        {
            // a b c
            // d e f
            // g h i

            return new Matrix3x3
            (
                lhs.a * rhs.a + lhs.b * rhs.d + lhs.c * rhs.g,
                lhs.a * rhs.b + lhs.b * rhs.e + lhs.c * rhs.h,
                lhs.a * rhs.c + lhs.b * rhs.f + lhs.c * rhs.i,

                lhs.d * rhs.a + lhs.e * rhs.d + lhs.f * rhs.g,
                lhs.d * rhs.b + lhs.e * rhs.e + lhs.f * rhs.h,
                lhs.d * rhs.c + lhs.e * rhs.f + lhs.f * rhs.i,

                lhs.g * rhs.a + lhs.h * rhs.d + lhs.i * rhs.g,
                lhs.g * rhs.b + lhs.h * rhs.e + lhs.i * rhs.h,
                lhs.g * rhs.c + lhs.h * rhs.f + lhs.i * rhs.i
            );
        }

        public Matrix3x3 Invert()
        {
            float A = e * i - f * h;
            float B = f * g - d * i;
            float C = d * h - e * g;
            float D = c * h - b * i;
            float E = a * i - c * g;
            float F = b * g - a * h;
            float G = b * f - c * e;
            float H = c * d - a * f;
            float I = a * e - b * d;

            float det = a * A + b * B + c * C;

            return (1.0f / det) * new Matrix3x3
            (
                A, D, G,
                B, E, H,
                C, F, I
            );
        }

        public static void Invert6x6(float[] x)
        {
            // x is a 6x6 matrix in row major layout

            //  0  1  2  3  4  5
            //  6  7  8  9 10 11
            // 12 13 14 15 16 17
            // 18 19 20 21 22 23
            // 24 25 26 27 28 29
            // 30 31 32 33 34 35

            Matrix3x3 A = new Matrix3x3
            (
                x[ 0], x[ 1], x[ 2],
                x[ 6], x[ 7], x[ 8],
                x[12], x[13], x[14]
            );

            Matrix3x3 B = new Matrix3x3
            (
                x[ 3], x[ 4], x[ 5],
                x[ 9], x[10], x[11],
                x[15], x[16], x[17]
            );

            Matrix3x3 C = new Matrix3x3
            (
                x[18], x[19], x[20],
                x[24], x[25], x[26],
                x[30], x[31], x[32]
            );

            Matrix3x3 D = new Matrix3x3
            (
                x[21], x[22], x[23],
                x[27], x[28], x[29],
                x[33], x[34], x[35]
            );

            Matrix3x3 X = A.Invert();           // A^-1
            Matrix3x3 Y = X * B;                // A^-1 * B
            Matrix3x3 Z = C * X;                // C * A^-1
            Matrix3x3 W = (D - Z * B).Invert(); // (D - C * A^-1 * B)^-1

            Matrix3x3 A1 =  X + Y * W * Z;
            Matrix3x3 B1 = -Y * W;
            Matrix3x3 C1 = -W * Z;
            Matrix3x3 D1 =  W;

            x[ 0] = A1.a; x[ 1] = A1.b; x[ 2] = A1.c;
            x[ 6] = A1.d; x[ 7] = A1.e; x[ 8] = A1.f;
            x[12] = A1.g; x[13] = A1.h; x[14] = A1.i;
            
            x[ 3] = B1.a; x[ 4] = B1.b; x[ 5] = B1.c;
            x[ 9] = B1.d; x[10] = B1.e; x[11] = B1.f;
            x[15] = B1.g; x[16] = B1.h; x[17] = B1.i;

            x[18] = C1.a; x[19] = C1.b; x[20] = C1.c;
            x[24] = C1.d; x[25] = C1.e; x[26] = C1.f;
            x[30] = C1.g; x[31] = C1.h; x[32] = C1.i;

            x[21] = D1.a; x[22] = D1.b; x[23] = D1.c;
            x[27] = D1.d; x[28] = D1.e; x[29] = D1.f;
            x[33] = D1.g; x[34] = D1.h; x[35] = D1.i;
        }
    }
}
