using System.Collections.Generic;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;

namespace icp
{
    class PointCloud
    {
        Vector3[] points;
        int[] indices;
        VertexBuffer vb;
        IndexBuffer ib;
        VertexDeclaration vd;
        Effect effect;

        public PointCloud(int n)
        {
            points = new Vector3[(n + 1) * (n + 1)];
            indices = new int[2 * 3 * n * n];

            System.Random rnd = new System.Random();
            float step = 2.0f / n;

            for (int z = 0; z <= n; z++)
                for (int x = 0; x <= n; x++)
                    points[x + z * (n + 1)] = new Vector3(
                        -1.0f + x * 2.0f / n,
                        rnd.Next(0, 10) * 0.02f,
                        -1.0f + z * 2.0f / n
                    );

            int outindex = 0;
            for (int row = 0; row < n; row++)
                for (int col = 0; col < n; col++)
                {
                    int topleft = (n + 1) * row + col;
                    int topright = topleft + 1;
                    int bottomleft = topleft + n + 1;
                    int bottomright = bottomleft + 1;

                    indices[outindex++] = topleft;
                    indices[outindex++] = topright;
                    indices[outindex++] = bottomleft;

                    indices[outindex++] = bottomleft;
                    indices[outindex++] = topright;
                    indices[outindex++] = bottomright;
                }

            vd = new VertexDeclaration(new VertexElement[] { 
                new VertexElement(0, VertexElementFormat.Vector3, VertexElementUsage.Position, 0)
            });
        }

        public void Initialize(GraphicsDevice dev)
        {
            vb = new VertexBuffer(dev, vd, points.Length, BufferUsage.WriteOnly);
            vb.SetData<Vector3>(points);

            ib = new IndexBuffer(dev, IndexElementSize.ThirtyTwoBits, indices.Length, BufferUsage.WriteOnly);
            ib.SetData<int>(indices);
        }

        public void LoadContent(ContentManager cm)
        {
            effect = cm.Load<Effect>("simple");
        }

        public void Draw(GraphicsDevice dev, Matrix world, Matrix view, Matrix proj)
        {
            RasterizerState rs = new RasterizerState();
            rs.FillMode = FillMode.WireFrame;

            dev.RasterizerState = rs;
            dev.SetVertexBuffer(vb);
            dev.Indices = ib;

            effect.Parameters["World"].SetValue(world);
            effect.Parameters["View"].SetValue(view);
            effect.Parameters["Projection"].SetValue(proj);
            effect.Techniques[0].Passes[0].Apply();

            dev.DrawIndexedPrimitives(PrimitiveType.TriangleList, 0, 0, points.Length, 0, indices.Length / 3);
        }

        // TODO: Extract class
        //private static 

        public static Matrix Align(PointCloud src, Matrix srcWorld, PointCloud dst, Matrix dstWorld)
        {
            Vector3 centerSrc = new Vector3(0.0f);
            Vector3 centerDst = new Vector3(0.0f);

            // Compute centroids of both point clouds
            for (int ii = 0; ii < src.points.Length; ii++)
                centerSrc += Vector3.Transform(src.points[ii], srcWorld);

            centerSrc /= src.points.Length;

            for (int ii = 0; ii < dst.points.Length; ii++)
                centerDst += Vector3.Transform(dst.points[ii], dstWorld);

            centerDst /= dst.points.Length;

            double Sxx = 0.0f, Sxy = 0.0f, Sxz = 0.0f,
                   Syx = 0.0f, Syy = 0.0f, Syz = 0.0f,
                   Szx = 0.0f, Szy = 0.0f, Szz = 0.0f;

            for (int ii = 0; ii < src.points.Length; ii++)
            {
                Vector3 vdst = Vector3.Transform(dst.points[ii], dstWorld) - centerDst;
                Vector3 vsrc = Vector3.Transform(src.points[ii], srcWorld) - centerSrc;

                Sxx += vdst.X * vsrc.X;
                Sxy += vdst.X * vsrc.Y;
                Sxz += vdst.X * vsrc.Z;

                Syx += vdst.Y * vsrc.X;
                Syy += vdst.Y * vsrc.Y;
                Syz += vdst.Y * vsrc.Z;

                Szx += vdst.Z * vsrc.X;
                Szy += vdst.Z * vsrc.Y;
                Szz += vdst.Z * vsrc.Z;
            }

            double[,] N = { 
                { Sxx + Syy + Szz, Syz - Szy      , Szx - Sxz      , Sxy - Syx       },
                { 0.0            , Sxx - Syy - Szz, Sxy + Syx      , Szx + Sxz       },
                { 0.0            , 0.0            , Syy - Sxx - Szz, Syz + Szy       },
                { 0.0            , 0.0            , 0.0            , Szz - Sxx - Syy }
            };

            // N is symmetric by construction.
            N[1, 0] = N[0, 1];
            N[2, 0] = N[0, 2];
            N[2, 1] = N[1, 2];
            N[3, 0] = N[0, 3];
            N[3, 1] = N[1, 3];
            N[3, 2] = N[2, 3];

            // compute my own ev
            double a = N[0, 0];
            double b = N[1, 1];
            double c = N[2, 2];
            double d = N[3, 3];
            double e = N[1, 0];
            double f = N[2, 1];
            double g = N[3, 2];
            double h = N[2, 0];
            double i = N[3, 1];
            double j = N[3, 0];

            // todo: double check c0..c4
            //double c4 = 1.0;
            //double c3 = 0.0;
            double c2 = a * c - h * h + b * c - f * f + a * d - j * j + b * d - i * i;
            double c1 = -b * (c * d - g * g) + f * (d * f - g * i) - i * (f * g - c * i)
                - a * (c * d - g * g) + h * (d * h - g * j) - j * (g * h - c * j)
                - a * (b * d - i * i) + e * (d * e - i * j) - j * (e * i - b * j)
                - a * (b * c - f * f) + e * (c * e - f * h) - h * (e * f - b * h);
            double c0 = (a * b - e * e) * (c * d - g * g) + (e * h - a * f) * (f * d - g * i)
                + (a * i - e * j) * (f * g - c * i) + (e * f - b * h) * (h * d - g * j)
                + (b * j - e * i) * (h * g - c * j) + (h * i - f * j) * (h * i - f * j);

            // a = c4
            // b = c3
            // c = c2
            // d = c1
            // e = c0
            double delta =
                /*256 * c4 * c4 * c4 * c0 * c0 * c0 
                - 192 * c4 * c4 * c3 * c1 * c0 * c0 
                - 128 * c4 * c4 * c2 * c2 * c0 * c0
                + 144 * c4 * c4 * c2 * c1 * c1 * c0 
                - 27 * c4 * c4 * c1 * c1 * c1 * c1
                + 144 * c4 * c3 * c3 * c2 * c0 * c0 
                - 6 * c4 * c3 * c3 * c1 * c1 * c0 
                - 80 * c4 * c3 * c2 * c2 * c1 * c0
                + 18 * c4 * c3 * c2 * c1 * c1 * c1 
                + 16 * c4 * c2 * c2 * c2 * c2 * c0 
                - 4 * c4 * c2 * c2 * c2 * c1 * c1
                - 27 * c3 * c3 * c3 * c3 * c0 * c0 
                + 18 * c3 * c3 * c3 * c2 * c1 * c0 
                - 4 * c3 * c3 * c3 * c1 * c1 * c1 
                - 4 * c3 * c3 * c2 * c2 * c2 * c0
                + c3 * c3 * c2 * c2 * c1 * c1;*/
                256 * c0 * c0 * c0
                - 128 * c2 * c2 * c0 * c0
                + 144 * c2 * c1 * c1 * c0
                - 27 * c1 * c1 * c1 * c1
                + 16 * c2 * c2 * c2 * c2 * c0
                - 4 * c2 * c2 * c2 * c1 * c1;

            double[,] N2 = new double[4, 4];
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 4; col++)
                    N2[row, col] = N[row, col];

            double[] ev;
            Eigen.eigen(N, out ev);

            int imaxev = 0;
            for (int ii = 1; ii < ev.Length; ii++)
                if (ev[ii] > ev[imaxev])
                    imaxev = ii;

            // ev can be computed with Ferrari's rule.
            N2[0, 0] -= ev[imaxev];
            N2[1, 1] -= ev[imaxev];
            N2[2, 2] -= ev[imaxev];
            N2[3, 3] -= ev[imaxev];

            float[,] result = new float[4, 4];
            for (int ii = 0; ii < 4; ii++)
                for (int jj = 0; jj < 4; jj++)
                {
                    float[,] tmp = new float[3, 3];
                    int aa = 0;
                    for (int row = 0; row < 4; row++)
                        if (row != ii)
                        {
                            int bb = 0;
                            for (int col = 0; col < 4; col++)
                                if (col != jj)
                                    tmp[aa, bb++] = (float)N2[row, col];
                            aa++;
                        }

                    // output co-factor
                    result[ii, jj] = new Matrix3x3
                    (
                        tmp[0, 0], tmp[0, 1], tmp[0, 2],
                        tmp[1, 0], tmp[1, 1], tmp[1, 2],
                        tmp[2, 0], tmp[2, 1], tmp[2, 2]
                    ).det();

                    if ((ii + jj) % 2 == 1)
                        result[ii, jj] = -result[ii, jj];
                }

            // todo: use biggest value rather than adding all of them for greater accuracy
            for (int ii = 1; ii < 4; ii++)
                for (int jj = 0; jj < 4; jj++)
                    result[0, jj] += result[ii, jj];

            Vector4 q = new Vector4((float)N[0, imaxev], (float)N[1, imaxev], (float)N[2, imaxev], (float)N[3, imaxev]);
            q.Normalize();

            Vector4 q2 = new Vector4(result[0, 0], result[0, 1], result[0, 2], result[0, 3]);
            q2.Normalize();

            Matrix R = Matrix.Identity;
            float xy = q.X * q.Y;
            float xz = q.X * q.Z;
            float xw = q.X * q.W;
            float yy = q.Y * q.Y;
            float yz = q.Y * q.Z;
            float yw = q.Y * q.W;
            float zz = q.Z * q.Z;
            float zw = q.Z * q.W;
            float ww = q.W * q.W;

            R.M11 = 1.0f - 2.0f * (zz + ww);
            R.M12 = 2.0f * (yz - xw);
            R.M13 = 2.0f * (yw + xz);

            R.M21 = 2.0f * (yz + xw);
            R.M22 = 1.0f - 2.0f * (yy + ww);
            R.M23 = 2.0f * (zw - xy);

            R.M31 = 2.0f * (yw - xz);
            R.M32 = 2.0f * (zw + xy);
            R.M33 = 1.0f - 2.0f * (yy + zz);

            Vector3 t = centerDst - Vector3.Transform(centerSrc, R);

            return R * Matrix.CreateTranslation(t);
        }
    }
}
