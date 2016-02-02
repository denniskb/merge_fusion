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
            eye = new Vector3(-0.5f, 0.0f, 2.0f);
            forward = Vector3.Normalize(-eye);
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
            // ppY = 240, fl = 571.26
            // 2 * atan( ppY / fl ) = 0.795466679
            const float radiansKinectFovY = 0.795466679f;
            
            return
                GetView() *
                Matrix.CreatePerspectiveFieldOfView(radiansKinectFovY, 640.0f / 480.0f, 0.4f, 4.0f);
        }

        public void Update()
        {
            // first translate, then rotate

            var gamepad = GamePad.GetState(PlayerIndex.One);
            var keyboard = Keyboard.GetState();

            float triggerR = gamepad.Triggers.Right;
            float triggerL = gamepad.Triggers.Left;
            float thumbLX = gamepad.ThumbSticks.Left.X;
            float thumbLY = gamepad.ThumbSticks.Left.Y;
            float thumbRX = gamepad.ThumbSticks.Right.X;
            float thumbRY = gamepad.ThumbSticks.Right.Y;

            if (keyboard.IsKeyDown(Keys.Left))
                thumbLX = -1f;
            if (keyboard.IsKeyDown(Keys.Right))
                thumbLX = 1f;
            if (Keyboard.GetState().IsKeyDown(Keys.Up))
                thumbLY = 1f;
            if (Keyboard.GetState().IsKeyDown(Keys.Down))
                thumbLY = -1f;

            if (keyboard.IsKeyDown(Keys.A))
                thumbRX = -1f;
            if (keyboard.IsKeyDown(Keys.D))
                thumbRX = 1f;
            if (keyboard.IsKeyDown(Keys.W))
                thumbRY = 1f;
            if (keyboard.IsKeyDown(Keys.S))
                thumbRY = -1f;

            float maxMoveDelta = 0.5f / 60.0f;
            float maxRotDelta = 0.25f * MathHelper.Pi / 60.0f;

            Vector3 up = new Vector3(0.0f, 1.0f, 0.0f);
            Vector3 right = Vector3.Normalize(Vector3.Cross(GetForward(), up));
            Vector3 forwardXY = Vector3.Cross(up, right);

            eye += new Vector3(0.0f, triggerR * maxMoveDelta, 0.0f);
            eye -= new Vector3(0.0f, triggerL * maxMoveDelta, 0.0f);

            eye += forwardXY * thumbLY * maxMoveDelta;
            eye += right * thumbLX * maxMoveDelta;

            forward = Vector3.TransformNormal(forward, Matrix.CreateFromAxisAngle(right, thumbRY * maxRotDelta));
            forward = Vector3.TransformNormal(forward, Matrix.CreateRotationY(thumbRX * -maxRotDelta));
            forward.Normalize();
        }
    }
}
