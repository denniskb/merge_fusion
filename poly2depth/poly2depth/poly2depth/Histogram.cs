using System.Diagnostics;
using Microsoft.Xna.Framework;

namespace poly2depth
{
    class Histogram
    {
        int[,,] data;
        Vector3 res;
        Vector3 maxidx;
        Vector3 maxidxtrilerp;
        Vector3 min;
        Vector3 max;
        Vector3 size;

        public Histogram(int res, Vector3 min, Vector3 max)
        {
            Debug.Assert(
                min.X < max.X &&
                min.Y < max.Y &&
                min.Z < max.Z
            );

            data = new int[res, res, res];
            this.res = new Vector3(res, res, res);
            maxidx = this.res - new Vector3(1.0f);
            maxidxtrilerp = this.res - new Vector3(2.0f);
            this.min = min;
            this.max = max;
            size = max - min;
        }

        public void Clear()
        {
            for (int z = 0; z < data.GetLength(2); z++)
                for (int y = 0; y < data.GetLength(1); y++)
                    for (int x = 0; x < data.GetLength(0); x++ )
                        data[x, y, z] = 0;            
        }

        public void Increment(Vector3 p)
        {
            var idx = PosToIdx(p);

            data[(int) idx.X, (int) idx.Y, (int) idx.Z]++;
        }

        public void Decrement(Vector3 p)
        {
            var idx = PosToIdx(p);

            data[(int)idx.X, (int)idx.Y, (int)idx.Z]--;
        }

        public int Point(Vector3 p)
        {
            var idx = PosToIdx(p);

            return data[(int)idx.X, (int)idx.Y, (int)idx.Z];
        }

        public float Trilerp(Vector3 p)
        {
            Vector3 idx = (p - min) / size * res - new Vector3(0.5f);
            idx = Max(Vector3.Zero, Min(maxidxtrilerp, idx));

            int x0 = (int)idx.X;
            int y0 = (int)idx.Y;
            int z0 = (int)idx.Z;

            int x1 = x0 + 1;
            int y1 = y0 + 1;
            int z1 = z0 + 1;

            float f000 = data[x0, y0, z0];
            float f100 = data[x1, y0, z0];
            float f010 = data[x0, y1, z0];
            float f110 = data[x1, y1, z0];

            float f001 = data[x0, y0, z1];
            float f101 = data[x1, y0, z1];
            float f011 = data[x0, y1, z1];
            float f111 = data[x1, y1, z1];

            float fi00 = MathHelper.Lerp(f000, f100, idx.X - x0);
            float fi01 = MathHelper.Lerp(f001, f101, idx.X - x0);
            float fi10 = MathHelper.Lerp(f010, f110, idx.X - x0);
            float fi11 = MathHelper.Lerp(f011, f111, idx.X - x0);

            float fii0 = MathHelper.Lerp(fi00, fi10, idx.Y - y0);
            float fii1 = MathHelper.Lerp(fi01, fi11, idx.Y - y0);

            return MathHelper.Lerp(fii0, fii1, idx.Z - z0);
        }

        private static Vector3 Min(Vector3 a, Vector3 b)
        {
            return new Vector3(
                MathHelper.Min(a.X, b.X),
                MathHelper.Min(a.Y, b.Y),
                MathHelper.Min(a.Z, b.Z)
            );
        }

        private static Vector3 Max(Vector3 a, Vector3 b)
        {
            return new Vector3(
                MathHelper.Max(a.X, b.X),
                MathHelper.Max(a.Y, b.Y),
                MathHelper.Max(a.Z, b.Z)
            );
        }

        private Vector3 PosToIdx(Vector3 p)
        {
            return Max(Vector3.Zero, Min(maxidx, (p - min) / size * res));
        }
    }
}
