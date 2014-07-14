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
            for (int i = 0; i < src.points.Length; i++)
                centerSrc += Vector3.Transform(src.points[i], srcWorld);

            centerSrc /= src.points.Length;

            for (int i = 0; i < dst.points.Length; i++)
                centerDst += Vector3.Transform(dst.points[i], dstWorld);

            centerDst /= dst.points.Length;

           double Sxx = 0.0f, Sxy = 0.0f, Sxz = 0.0f,
                  Syx = 0.0f, Syy = 0.0f, Syz = 0.0f,
                  Szx = 0.0f, Szy = 0.0f, Szz = 0.0f;

            for (int i = 0; i < src.points.Length; i++ )
            {
                Vector3 vdst = Vector3.Transform(dst.points[i], dstWorld) - centerDst;
                Vector3 vsrc = Vector3.Transform(src.points[i], srcWorld) - centerSrc;

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
                { Syz - Szy      , Sxx - Syy - Szz, Sxy + Syx      , Szx + Sxz       },
                { Szx - Sxz      , Sxy + Syx      , Syy - Sxx - Szz, Syz + Szy       },
                { Sxy - Syx      , Szx + Sxz      , Syz + Szy      , Szz - Sxx - Syy }
            };

            double[] ev;
            Eigen.eigen(N, out ev);

            int imaxev = 0;
            for (int i = 1; i < ev.Length; i++)
                if (ev[i] > ev[imaxev])
                    imaxev = i;
            
            Vector4 q = new Vector4((float)N[0, imaxev], (float)N[1, imaxev], (float)N[2, imaxev], (float)N[3, imaxev]);
            q.Normalize();

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
