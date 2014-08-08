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
        private RenderTarget2D rt;

        private Effect vizEffect;
        private Effect depthEffect;
        private Effect viz;
        private Effect depth;
        private Effect skinnedViz;
        private Effect skinnedDepth;

        private AnimatedFBX model;
        private AnimatedFBX imrod;
        private AnimatedFBX temple;

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

            rt = new RenderTarget2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4, DepthFormat.Depth24);
        }

        /// <summary>
        /// LoadContent will be called once per game and is the place to load
        /// all of your content.
        /// </summary>
        protected override void LoadContent()
        {
            viz = Content.Load<Effect>("Viz");
            depth = Content.Load<Effect>("Depth");
            skinnedViz = Content.Load<Effect>("SkinnedViz");
            skinnedDepth = Content.Load<Effect>("SkinnedDepth");

            imrod = new AnimatedFBX
            (
                Content,
                "imrod",
                Matrix.CreateTranslation(0.0f, 0.0f, 42.7585f),
                Matrix.CreateScale(0.023f) * Matrix.CreateRotationX(MathHelper.PiOver2)
            );

            temple = new AnimatedFBX
            (
                Content,
                "house",
                Matrix.CreateTranslation(0.0f, -10.0f, 20.0f),
                Matrix.CreateScale(0.1f)
            );

            //*
            vizEffect = skinnedViz;
            depthEffect = skinnedDepth;
            model = imrod;
            /*/
            vizEffect = viz;
            depthEffect = depth;
            model = temple;
            //*/
        }

        /// <summary>
        /// UnloadContent will be called once per game and is the place to unload
        /// all content.
        /// </summary>
        protected override void UnloadContent()
        {
        }

        bool updated = false;

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
            if (!updated)
            {
                imrod.Update(gameTime);
                updated = true;
            }
            recorder.Update();

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            if (recorder.IsRecording() && frame % 3 == 0)
            {
                graphics.GraphicsDevice.SetRenderTarget(rt);
                GraphicsDevice.Clear(Color.Black);
                model.Draw(depthEffect, cam.GetViewProjection(), cam.GetEye(), cam.GetForward());
                graphics.GraphicsDevice.SetRenderTarget(null);

                recorder.RecordFrame(rt, cam.GetView());
            }

            GraphicsDevice.Clear(Color.CornflowerBlue);
            model.Draw(vizEffect, cam.GetViewProjection(), cam.GetEye(), cam.GetForward());
            
            base.Draw(gameTime);

            frame++;
        }
    }
}
