﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace icp
{
    class Eigen
    {
        private static int MAX_ITERS = 30;

        private static double SIGN(double a, double b)
        {
            return b < 0.0 ? -Math.Abs(a) : Math.Abs(a);
        }

        /*
         * Computes the eigenvalues and eigenvectors of a symmetric matrix 'matrix'.
         * The eigenvalues are retunred via 'eigenValues', while the eigenvectors are written
         * to the columns of 'matrix'.
         */
        public static void eigen(double[,] matrix, out double[] eigenValues)
        {
            eigenValues = new double[matrix.GetLength(0)];
            double[] tmp = new double[matrix.GetLength(0)];

            G_tred2(matrix, eigenValues, tmp);
            G_tqli(eigenValues, tmp, matrix);
        }

        private static void G_tred2(double[,] a, double[] d, double[] e)
        {
            int l, k, j, i;
            double scale, hh, h, g, f;

            for (i = a.GetLength(0) - 1; i >= 1; i--)
            {
                l = i - 1;
                h = scale = 0.0;
                if (l > 0)
                {
                    for (k = 0; k <= l; k++)
                        scale += Math.Abs(a[i,k]);
                    if (scale == 0.0)
                        e[i] = a[i,l];
                    else
                    {
                        for (k = 0; k <= l; k++)
                        {
                            a[i,k] /= scale;
                            h += a[i,k] * a[i,k];
                        }
                        f = a[i,l];
                        g = f > 0 ? -Math.Sqrt(h) : Math.Sqrt(h);
                        e[i] = scale * g;
                        h -= f * g;
                        a[i,l] = f - g;
                        f = 0.0;
                        for (j = 0; j <= l; j++)
                        {
                            /* Next statement can be omitted if eigenvectors not wanted */
                            a[j,i] = a[i,j] / h;
                            g = 0.0;
                            for (k = 0; k <= j; k++)
                                g += a[j,k] * a[i,k];
                            for (k = j + 1; k <= l; k++)
                                g += a[k,j] * a[i,k];
                            e[j] = g / h;
                            f += e[j] * a[i,j];
                        }
                        hh = f / (h + h);
                        for (j = 0; j <= l; j++)
                        {
                            f = a[i,j];
                            e[j] = g = e[j] - hh * f;
                            for (k = 0; k <= j; k++)
                                a[j,k] -= (f * e[k] + g * a[i,k]);
                        }
                    }
                }
                else
                    e[i] = a[i,l];
                d[i] = h;
            }
            /* Next statement can be omitted if eigenvectors not wanted */
            d[0] = 0.0;
            e[0] = 0.0;
            /* Contents of this loop can be omitted if eigenvectors not 
                    wanted except for statement d[i]=a[i,i]; */
            for (i = 0; i < a.GetLength(0); i++)
            {
                l = i - 1;
                if (d[i] != 0.0)
                {
                    for (j = 0; j <= l; j++)
                    {
                        g = 0.0;
                        for (k = 0; k <= l; k++)
                            g += a[i,k] * a[k,j];
                        for (k = 0; k <= l; k++)
                            a[k,j] -= g * a[k,i];
                    }
                }
                d[i] = a[i,i];
                a[i,i] = 1.0;
                for (j = 0; j <= l; j++) a[j,i] = a[i,j] = 0.0;
            }
        }

        /* From Numerical Recipies in C */  
      
        static int G_tqli(double[] d, double[] e, double[,] z)  
        {  
            int m,l,iter,i,k;  
            double s,r,p,g,f,dd,c,b;  
      
            for (i=1;i<z.GetLength(0);i++) e[i-1]=e[i];
            e[z.GetLength(0) - 1] = 0.0;
            for (l = 0; l < z.GetLength(0); l++)
            {  
                iter=0;  
                do {
                    for (m = l; m < z.GetLength(0) - 1; m++)
                    {  
                        dd=Math.Abs(d[m])+Math.Abs(d[m+1]);  
                        if (Math.Abs(e[m])+dd == dd) break;  
                    }  
                    if (m != l) {  
                        if (iter++ == MAX_ITERS) return 0; /* Too many iterations in TQLI */  
                        g=(d[l+1]-d[l])/(2.0*e[l]);  
                        r=Math.Sqrt((g*g)+1.0);  
                        g=d[m]-d[l]+e[l]/(g+SIGN(r,g));  
                        s=c=1.0;  
                        p=0.0;  
                        for (i=m-1;i>=l;i--) {  
                            f=s*e[i];  
                            b=c*e[i];  
                            if (Math.Abs(f) >= Math.Abs(g)) {  
                                c=g/f;  
                                r=Math.Sqrt((c*c)+1.0);  
                                e[i+1]=f*r;  
                                c *= (s=1.0/r);  
                            } else {  
                                s=f/g;  
                                r=Math.Sqrt((s*s)+1.0);  
                                e[i+1]=g*r;  
                                s *= (c=1.0/r);  
                            }  
                            g=d[i+1]-p;  
                            r=(d[i]-g)*s+2.0*c*b;  
                            p=s*r;  
                            d[i+1]=g+p;  
                            g=c*r-b;  
                            /* Next loop can be omitted if eigenvectors not wanted */
                            for (k = 0; k < z.GetLength(0); k++)
                            {  
                                f=z[k,i+1];  
                                z[k,i+1]=s*z[k,i]+c*f;  
                                z[k,i]=c*z[k,i]-s*f;  
                            }  
                        }  
                        d[l]=d[l]-p;  
                        e[l]=g;  
                        e[m]=0.0;  
                    }  
                } while (m != l);  
            }  
            return 1;  
        }  
    }
}
