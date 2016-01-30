using Microsoft.Xna.Framework;
using System.Collections.Generic;

namespace poly2depth
{
    class PointCloud
    {
        private List<Vector4> points;

        public PointCloud()
        {
            points = new List<Vector4>(640 * 480);
        }

        public void Integrate(Vector4[] depthPointCloud)
        {
            points.Clear();
            points.AddRange(depthPointCloud);
        }

        public void Render(Matrix worldToClip, Vector3 eye, Vector3 forward, Vector4[] outRGBA)
        {
            foreach (Vector4 p in points)
            {
                Vector4 worldPos = new Vector4(p.Y, p.Z, p.W, 1.0f);
                Vector4 clipPos = Vector4.Transform(worldPos, worldToClip); clipPos /= clipPos.W;

                // TODO: Generalize
                int screenX = (int) (clipPos.X * 320.0f + 320.0f);
                int screenY = 480 - (int) (clipPos.Y * 240.0f + 240.0f);

                // TODO: Generalize
                if (screenX < 0 || screenX >= 640 ||
                    screenY < 0 || screenY >= 480)
                    continue;

                // TODO: Generalize
                int idx = 640 * screenY + screenX;
                float depth = Vector3.Dot(new Vector3(worldPos.X, worldPos.Y, worldPos.Z) - eye, forward);
                float gray = (depth - 0.4f) / 3.6f;

                outRGBA[idx] = new Vector4(gray, gray, gray, 0.0f);
            }
        }
    }
}