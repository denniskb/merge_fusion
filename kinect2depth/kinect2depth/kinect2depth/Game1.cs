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
        GraphicsDeviceManager graphics;
        KinectCamera cam;

        float[] cachedDepth;
        Vector4[] tmpTex;
        Texture2D renderSurface;
        SpriteBatch renderer;

        Recorder recorder;

        public Game1()
        {
            graphics = new GraphicsDeviceManager(this);
            graphics.PreferredBackBufferWidth = 640;
            graphics.PreferredBackBufferHeight = 480;

            Content.RootDirectory = "Content";

            cam = new KinectCamera();

            cachedDepth = new float[640 * 480];
            tmpTex = new Vector4[640 * 480];
            
            recorder = new Recorder();
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

            renderSurface = new Texture2D(graphics.GraphicsDevice, 640, 480, false, SurfaceFormat.Vector4);
            renderer = new SpriteBatch(graphics.GraphicsDevice);
        }

        /// <summary>
        /// LoadContent will be called once per game and is the place to load
        /// all of your content.
        /// </summary>
        protected override void LoadContent()
        {
        }

        /// <summary>
        /// UnloadContent will be called once per game and is the place to unload
        /// all content.
        /// </summary>
        protected override void UnloadContent()
        {
            cam.Stop();
            cam = null;
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

            recorder.Update();

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            graphics.GraphicsDevice.Clear(Color.Black);

            cam.NextFrame(cachedDepth);
            recorder.RecordFrame(cachedDepth, cam.GetView());

            graphics.GraphicsDevice.Textures[0] = null;
            for (int i = 0; i < 640 * 480; i++)
                tmpTex[i] = new Vector4(cachedDepth[i/640*640+639-i%640] * 0.5f);
            renderSurface.SetData<Vector4>(tmpTex);

            renderer.Begin(SpriteSortMode.Immediate, null, SamplerState.PointClamp, null, null);
            renderer.Draw(renderSurface, new Rectangle(0, 0, 640, 480), Color.White);
            renderer.End();

            base.Draw(gameTime);
        }
    }
}
