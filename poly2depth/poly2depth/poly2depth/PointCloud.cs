using Microsoft.Xna.Framework;
using System.Collections.Generic;

namespace poly2depth
{
    class PointCloud
    {
        private List<Vector4> points;
        private float TR;

        private Histogram densities;
        
        public PointCloud()
        {
            points = new List<Vector4>(640 * 480);
            TR = 0.02f;

            densities = new Histogram(128, new Vector3(-1.5f), new Vector3(1.5f));
        }

        public void Integrate(Vector4[] depthPointCloud, Matrix worldToClip, Vector3 eye, Vector3 forward)
        {
            // Update existing points
            for (int i = 0; i < points.Count; i++)
            {
                Vector4 p = points[i];
                Vector3 worldPos = new Vector3(p.X, p.Y, p.Z);
                Vector4 clipPos = Vector4.Transform(new Vector4(worldPos.X, worldPos.Y, worldPos.Z, 1.0f), worldToClip);

                float depth = clipPos.W;
                
                // TODO: Generalize
                int screenX = (int)(clipPos.X / clipPos.W * 320.0f + 320.0f);
                int screenY = 480 - (int)(clipPos.Y / clipPos.W * 240.0f + 240.0f);

                // TODO: Generalize
                if (screenX < 0 || screenX >= 640 ||
                    screenY < 0 || screenY >= 480)
                    continue;

                // TODO: Generalize
                int idx = 640 * screenY + screenX;
                
                if (depth < depthPointCloud[idx].X + TR)
                {
                    Vector3 rawPos = (worldPos - eye) / depth * depthPointCloud[idx].X + eye;
                    Vector3 newPos = Vector3.Lerp(worldPos, rawPos, 1.0f / (p.W + 1.0f));

                    points[i] = new Vector4(newPos.X, newPos.Y, newPos.Z, p.W + 1.0f);
                }
            }

            // Add new valid points from current depth map
            if (points.Count < 500000) // HACK
                for (int i = 0; i < depthPointCloud.Length; i++)
                {
                    Vector4 p = depthPointCloud[i];
                    if (p.X > 0.0f)
                    {
                        points.Add(new Vector4(p.Y, p.Z, p.W, 1.0f));
                    }
                }

            // Reset histogram
            densities.Clear();

            // Update histogram
            for (int i = 0; i < points.Count; i++)
            {
                Vector4 p = points[i];
                Vector3 pos = new Vector3(p.X, p.Y, p.Z);

                densities.Increment(pos);
            }

            // Delete points based on histogram
            for (int i = 0; i < points.Count; i++)
            {
                Vector4 p = points[i];
                Vector3 pos = new Vector3(p.X, p.Y, p.Z);
            
                float count = densities.Trilerp(pos);
            
                if (count < 5.0f || count > 200.0f)
                {
                    p.W = 0.0f;
                    points[i] = p;
                    densities.Decrement(pos);
                }
            }

            // Compact array (remove 'mark-deleted' points)
            {
                int idst = 0;
                int isrc = 0;

                for (; isrc < points.Count; isrc++)
                    if (points[isrc].W > 0.0f)
                        points[idst++] = points[isrc];

                points.RemoveRange(idst, points.Count - idst);
            }
        }

        public List<Vector4> Points()
        {
            return points;
        }
    }
}