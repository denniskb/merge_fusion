using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

namespace icp
{
    class Rt
    {
        Vector3 t;
        Vector3 R;

        public Rt()
        {
            t = new Vector3(0);
            R = new Vector3(0);
        }

        public void Update(GameTime gt)
        {
            GamePadState state = GamePad.GetState(PlayerIndex.One);
            float tdelta = 1.0f / 60.0f;

            t.Y += state.Triggers.Right * tdelta;
            t.Y -= state.Triggers.Left  * tdelta;

            t.X += state.ThumbSticks.Right.X * tdelta;
            t.Z -= state.ThumbSticks.Right.Y * tdelta;

            R.X -= state.ThumbSticks.Left.Y * tdelta;
            R.Z -= state.ThumbSticks.Left.X * tdelta;
        }

        public Matrix GetMatrx()
        {
            return
                Matrix.CreateRotationX(R.X) *
                Matrix.CreateRotationY(R.Y) *
                Matrix.CreateRotationZ(R.Z) *
                Matrix.CreateTranslation(t);
        }
    }
}
