using Microsoft.Xna.Framework;

namespace poly2depth
{
    class Renderer
    {
        private float[] depthBuffer;

        public Renderer()
        {
            depthBuffer = new float[640 * 480];
        }

        public void Render(PointCloud pcl, Matrix worldToClip, Vector3 eye, Vector3 forward, Vector4[] outRGBA)
        {
            float MAX_DEPTH = 10.0f;

            for (int i = 0; i < depthBuffer.Length; i++)
            {
                depthBuffer[i] = MAX_DEPTH;
                outRGBA[i] = new Vector4();
            }

            foreach (Vector4 p in pcl.Points())
            {
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

                if (depth < depthBuffer[idx])
                    depthBuffer[idx] = depth;
            }

            for (int i = 0; i < depthBuffer.Length; i++)
            {
                if (depthBuffer[i] < MAX_DEPTH)
                {
                    float depth_norm = (depthBuffer[i] - 0.4f) / 3.6f;

                    float r = MathHelper.Min(1.0f, (MathHelper.Max(0.5f, depth_norm) - 0.5f) * 6.0f);
                    float g = MathHelper.Min(1.0f, depth_norm * 3.0f) - MathHelper.Min(1.0f, MathHelper.Max(0.0f, depth_norm - 0.666f) * 3.0f);
                    float b = MathHelper.Max(0.0f, 1.0f - MathHelper.Max(0.0f, depth_norm - 0.333f) * 6.0f);

                    outRGBA[i] = new Vector4(r, g, b, 1.0f);
                }
            }
        }
    }
}
