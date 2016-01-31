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
        private RenderTarget2D posOut;
        private RenderTarget2D noiseOut;
        private RenderTarget2D medianOut;

        private Effect depthEffect;
        private Effect billboardEffect;

        private AnimatedFBX temple;
        private AnimatedFBX billboard;

        private Recorder recorder;

        private int iFrame;

        private Vector4[] rgba;
        private Texture2D rgbaTex;
        private Vector4[] tmpPCL;
        private PointCloud pcl;

        private Renderer render;

        public Game1()
        {
            graphics = new GraphicsDeviceManager(this);
            graphics.PreferredBackBufferWidth = 640;
            graphics.PreferredBackBufferHeight = 480;

            Content.RootDirectory = "Content";

            cam = new KinectCamera();

            recorder = new Recorder();

            iFrame = 0;

            // TODO: Generalize
            rgba = new Vector4[640 * 480];
            tmpPCL = new Vector4[640 * 480];
            pcl = new PointCloud();

            render = new Renderer();
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

            // TODO: Generalize
            depthOut = new RenderTarget2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4, DepthFormat.Depth24);
            posOut = new RenderTarget2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4, DepthFormat.Depth24);
            noiseOut = new RenderTarget2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4, DepthFormat.Depth24);
            medianOut = new RenderTarget2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4, DepthFormat.Depth24);

            rgbaTex = new Texture2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4);
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
            GraphicsDevice.SetRenderTarget(posOut);

            GraphicsDevice.Clear(Color.Black);
            depthEffect.CurrentTechnique = depthEffect.Techniques["WorldPos"];
            depthEffect.Parameters["modelToWorld"].SetValue(Matrix.CreateScale(0.075f) * Matrix.CreateTranslation(0.0f, -0.75f, -0.37f));
            depthEffect.Parameters["eyeToClip"].SetValue(cam.GetViewProjection());
            depthEffect.Parameters["eye"].SetValue(cam.GetEye());
            depthEffect.Parameters["forward"].SetValue(cam.GetForward());
            temple.Draw(depthEffect);

            GraphicsDevice.SetRenderTarget(depthOut);
            
            GraphicsDevice.Clear(Color.Black);
            depthEffect.CurrentTechnique = depthEffect.Techniques["DepthPlusNormal"];
            depthEffect.Parameters["modelToWorld"].SetValue(Matrix.CreateScale(0.075f) * Matrix.CreateTranslation(0.0f, -0.75f, -0.37f));
            depthEffect.Parameters["eyeToClip"].SetValue(cam.GetViewProjection());
            depthEffect.Parameters["eye"].SetValue(cam.GetEye());
            depthEffect.Parameters["forward"].SetValue(cam.GetForward());
            temple.Draw(depthEffect);

            GraphicsDevice.SetRenderTarget(noiseOut);

            if (recorder.IsRecording() && iFrame % 3 == 0)
                recorder.RecordFrame(depthOut, cam.GetView());
            
            GraphicsDevice.Clear(Color.Black);
            billboardEffect.CurrentTechnique = billboardEffect.Techniques["AddNoise"];
            billboardEffect.Parameters["depth"].SetValue(depthOut);
            billboardEffect.Parameters["pos"].SetValue(posOut);
            billboardEffect.Parameters["iFrame"].SetValue(iFrame);
            billboardEffect.Parameters["eye"].SetValue(cam.GetEye());
            billboardEffect.Parameters["forward"].SetValue(cam.GetForward());
            billboard.Draw(billboardEffect);

            GraphicsDevice.SetRenderTarget(medianOut);

            GraphicsDevice.Clear(Color.Black);
            billboardEffect.CurrentTechnique = billboardEffect.Techniques["ComputeMedian"];
            billboardEffect.Parameters["depth"].SetValue(noiseOut);
            billboard.Draw(billboardEffect);

            GraphicsDevice.SetRenderTarget(null);
            
            GraphicsDevice.Clear(Color.Black);
            billboardEffect.CurrentTechnique = billboardEffect.Techniques["Depth2Color"];
            billboardEffect.Parameters["depth"].SetValue(medianOut);
            billboardEffect.Parameters["eye"].SetValue(cam.GetEye());
            billboardEffect.Parameters["forward"].SetValue(cam.GetForward());
            billboard.Draw(billboardEffect);
            
            //if (iFrame < 100)
            {
                // TODO: Change tmpPCL packing order
                medianOut.GetData<Vector4>(tmpPCL);

                pcl.Integrate(tmpPCL, cam.GetViewProjection(), cam.GetEye(), cam.GetForward());

                render.Render(pcl, cam.GetViewProjection(), cam.GetEye(), cam.GetForward(), rgba);
                
                rgbaTex.SetData<Vector4>(rgba);
            }

            GraphicsDevice.SetRenderTarget(null);
            
            GraphicsDevice.Clear(Color.Black);
            billboardEffect.CurrentTechnique = billboardEffect.Techniques["RenderTex"];
            billboardEffect.Parameters["color"].SetValue(rgbaTex);
            billboard.Draw(billboardEffect);

            base.Draw(gameTime);

            iFrame++;
        }
    }
}
