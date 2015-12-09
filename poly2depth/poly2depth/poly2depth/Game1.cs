using System.Collections.Generic;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

namespace poly2depth
{
    /// <summary>
    /// A simple tool to record depth streams from animated characters.
    /// (More info in readme.txt)
    /// 1 world unit = 1 meter
    /// </summary>
    public class Game1 : Microsoft.Xna.Framework.Game
    {
        private GraphicsDeviceManager graphics;
        private KinectCamera cam;
        private RenderTarget2D depthOut;
        private RenderTarget2D noiseOut;

        private Effect depthEffect;
        private Effect billboardEffect;

        private AnimatedFBX temple;
        private AnimatedFBX billboard;

        private Recorder recorder;

        private int frame;

        public Game1()
        {
            graphics = new GraphicsDeviceManager(this);
            graphics.PreferredBackBufferWidth = 640;
            graphics.PreferredBackBufferHeight = 480;

            Content.RootDirectory = "Content";

            cam = new KinectCamera();

            recorder = new Recorder();

            frame = 0;
        }

        /// <summary>
        /// Allows the game to perform any initialization it needs to before starting to run.
        /// This is where it can query for any required services and load any non-graphic
        /// related content.  Calling base.Initialize will enumerate through any components
        /// and initialize them as well.
        /// </summary>
        protected override void Initialize()
        {
            base.Initialize();

            depthOut = new RenderTarget2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4, DepthFormat.Depth24);
            noiseOut = new RenderTarget2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4, DepthFormat.Depth24);
        }

        /// <summary>
        /// LoadContent will be called once per game and is the place to load
        /// all of your content.
        /// </summary>
        protected override void LoadContent()
        {
            depthEffect = Content.Load<Effect>("Depth");
            billboardEffect = Content.Load<Effect>("billboardViz");
            
            temple = new AnimatedFBX
            (
                Content,
                "house"
            );

            billboard = new AnimatedFBX
            (
                Content,
                "billboard"
            );
        }

        /// <summary>
        /// UnloadContent will be called once per game and is the place to unload
        /// all content.
        /// </summary>
        protected override void UnloadContent()
        {
        }

        /// <summary>
        /// Allows the game to run logic such as updating the world,
        /// checking for collisions, gathering input, and playing audio.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed ||
                Keyboard.GetState().IsKeyDown(Keys.Escape))
                this.Exit();

            cam.Update();
            recorder.Update();

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.SetRenderTarget(depthOut);

            GraphicsDevice.Clear(Color.Black);
            depthEffect.Parameters["modelToWorld"].SetValue(Matrix.CreateTranslation(0.0f, -10.0f, 20.0f) * Matrix.CreateScale(0.1f));
            depthEffect.Parameters["eyeToClip"].SetValue(cam.GetViewProjection());
            depthEffect.Parameters["eye"].SetValue(cam.GetEye());
            depthEffect.Parameters["forward"].SetValue(cam.GetForward());
            temple.Draw(depthEffect);

            if (recorder.IsRecording() && frame % 3 == 0)
                recorder.RecordFrame(depthOut, cam.GetView());

            GraphicsDevice.SetRenderTarget(noiseOut);
            
            GraphicsDevice.Clear(Color.CornflowerBlue);
            billboardEffect.CurrentTechnique = billboardEffect.Techniques["AddNoise"];
            billboardEffect.Parameters["depth"].SetValue(depthOut);
            billboard.Draw(billboardEffect);

            GraphicsDevice.SetRenderTarget(null);

            GraphicsDevice.Clear(Color.CornflowerBlue);
            billboardEffect.CurrentTechnique = billboardEffect.Techniques["Depth2Color"];
            billboardEffect.Parameters["depth"].SetValue(noiseOut);
            billboard.Draw(billboardEffect);
            
            base.Draw(gameTime);

            frame++;
        }
    }
}
