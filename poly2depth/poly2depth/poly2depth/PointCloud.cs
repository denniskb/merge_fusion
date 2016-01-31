using Microsoft.Xna.Framework;
using System.Collections.Generic;

namespace poly2depth
{
    class PointCloud
    {
        private List<Vector4> points;
        private float TR;

        public PointCloud()
        {
            points = new List<Vector4>(640 * 480);
            TR = 0.02f;
        }

        public void Integrate(Vector4[] depthPointCloud, Matrix worldToClip, Vector3 eye, Vector3 forward)
        {
            for (int i = 0; i < points.Count; i++)
            {
                Vector4 p = points[i];
                Vector3 worldPos = new Vector3(p.X, p.Y, p.Z);
                Vector4 clipPos = Vector4.Transform(new Vector4(worldPos.X, worldPos.Y, worldPos.Z, 1.0f), worldToClip); clipPos /= clipPos.W;

                // TODO: Generalize
                int screenX = (int)(clipPos.X * 320.0f + 320.0f);
                int screenY = 480 - (int)(clipPos.Y * 240.0f + 240.0f);

                // TODO: Generalize
                if (screenX < 0 || screenX >= 640 ||
                    screenY < 0 || screenY >= 480)
                    continue;

                // TODO: Generalize
                int idx = 640 * screenY + screenX;
                
                float depth = Vector3.Dot(worldPos - eye, forward);
                
                if (depth < depthPointCloud[idx].X + TR)
                {
                    Vector3 rawPos = (worldPos - eye) / depth * depthPointCloud[idx].X + eye;
                    Vector3 newPos = Vector3.Lerp(worldPos, rawPos, 1.0f / (p.W + 1.0f));

                    points[i] = new Vector4(newPos.X, newPos.Y, newPos.Z, p.W + 1.0f);
                }
            }

            // HACK
            if (points.Count < 300000)
                for (int i = 0; i < depthPointCloud.Length; i++)
                {
                    Vector4 p = depthPointCloud[i];
                    if (p.X > 0.0f)
                        points.Add(new Vector4(p.Y, p.Z, p.W, 1.0f));
                }

            // TODO:
            // - Create histogram density bins
            // - Count occurences during the two above iterations
            // - Prune points that exceed density (trilerp, ie cells have to store densities, not counts!)
            // - OR: They can store counts, but then trilerp the counts rather than sum them! etsi
            // - Show results, discuss further steps (hopefully just need to write down algorithm)
            // - Use them to cap both the sampling rate and to prune isolated points! then show how it performs
            //   without the pruning step as a comparison! afta
        }

        public List<Vector4> Points()
        {
            return points;
        }
    }
}