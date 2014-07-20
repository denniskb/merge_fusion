using Microsoft.Xna.Framework;

using Microsoft.Xna.Framework.Input;

namespace poly2depth
{
    /// <summary>
    /// Simmulates a Kinect camera in far mode (0.8m to 4.0m range)
    /// </summary>
    class KinectCamera
    {
        private Vector3 eye;
        private Vector3 forward;

        public KinectCamera()
        {
            eye = new Vector3(0.0f, 0.0f, 2.0f);
            forward = new Vector3(0.0f, 0.0f, -1.0f);
        }

        public Vector3 GetEye()
        {
            return eye;
        }

        public Vector3 GetForward()
        {
            return forward;
        }

        public Matrix GetView()
        {
            return Matrix.CreateLookAt(GetEye(), GetEye() + GetForward(), new Vector3(0, 1, 0));
        }

        public Matrix GetViewProjection()
        {
            const float radiansKinectFovY = 0.777169865f;

            return
                GetView() *
                Matrix.CreatePerspectiveFieldOfView(radiansKinectFovY, 639.0f / 479.0f, 0.8f, 4.0f);
        }

        public void Update()
        {
            // first translate, then rotate

            float maxMoveDelta = 2.0f / 60.0f;
            float maxRotDelta = 0.5f * MathHelper.Pi / 60.0f;

            Vector3 up = new Vector3(0.0f, 1.0f, 0.0f);
            Vector3 right = Vector3.Normalize(Vector3.Cross(GetForward(), up));
            Vector3 forwardXY = Vector3.Cross(up, right);

            eye += new Vector3(0.0f, GamePad.GetState(PlayerIndex.One).Triggers.Right * maxMoveDelta, 0.0f);
            eye -= new Vector3(0.0f, GamePad.GetState(PlayerIndex.One).Triggers.Left * maxMoveDelta, 0.0f);

            eye += forwardXY * GamePad.GetState(PlayerIndex.One).ThumbSticks.Left.Y * maxMoveDelta;
            eye += right * GamePad.GetState(PlayerIndex.One).ThumbSticks.Left.X * maxMoveDelta;

            forward = Vector3.TransformNormal(forward, Matrix.CreateFromAxisAngle(right, GamePad.GetState(PlayerIndex.One).ThumbSticks.Right.Y * maxRotDelta));
            forward = Vector3.TransformNormal(forward, Matrix.CreateRotationY(GamePad.GetState(PlayerIndex.One).ThumbSticks.Right.X * -maxRotDelta));
            forward.Normalize();
        }
    }
}
