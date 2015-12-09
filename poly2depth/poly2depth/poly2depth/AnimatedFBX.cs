using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;

using SkinnedModel;

namespace poly2depth
{
    class AnimatedFBX
    {
        private Model model;

        private AnimationPlayer player;

        public AnimatedFBX(ContentManager cm, string name)
        {
            model = cm.Load<Model>(name);

            player = null;

            SkinningData skinningData = model.Tag as SkinningData;
            if (skinningData == null)
                return;

            if (skinningData.AnimationClips.Values.Count == 0)
                return;

            player = new AnimationPlayer(skinningData);
            foreach (AnimationClip clip in skinningData.AnimationClips.Values)
            {
                player.StartClip(clip);
                break;
            }
        }

        public void Update(GameTime gt)
        {
            player.Update(gt.ElapsedGameTime, true, Matrix.Identity);
        }

        public void Draw(Effect effect)
        {
            Matrix[] bones = null;

            if (player != null)
                bones = player.GetSkinTransforms();

            foreach (ModelMesh mesh in model.Meshes)
            {
                foreach (ModelMeshPart part in mesh.MeshParts)
                {
                    part.Effect = effect;
                }
                
                foreach (Effect meshEffect in mesh.Effects)
                {
                    mesh.Draw();
                }
            }
        }
    }
}
