using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;

namespace icp
{
    /// <summary>
    /// This is the main type for your game
    /// </summary>
    public class Game1 : Microsoft.Xna.Framework.Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;

        bool aDown;
        Matrix icp;

        Rt world;
        Matrix view;
        Matrix proj;
        Matrix tmp;
        PointCloud pcl;

        public Game1()
        {
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";

            aDown = false;
            icp = Matrix.Identity;

            world = new Rt();
            view = Matrix.CreateLookAt(new Vector3(1.0f, 2.0f, 1.0f), new Vector3(0.0f, 0.0f, 0.0f), new Vector3(0.0f, 1.0f, 0.0f));
            proj = Matrix.CreatePerspectiveFieldOfView(0.93f, (float)graphics.PreferredBackBufferWidth / graphics.PreferredBackBufferHeight, 0.1f, 5.0f);
            tmp = Matrix.Identity;

            pcl = new PointCloud(8);
        }

        /// <summary>
        /// Allows the game to perform any initialization it needs to before starting to run.
        /// This is where it can query for any required services and load any non-graphic
        /// related content.  Calling base.Initialize will enumerate through any components
        /// and initialize them as well.
        /// </summary>
        protected override void Initialize()
        {
            // TODO: Add your initialization logic here

            base.Initialize();

            pcl.Initialize(graphics.GraphicsDevice);
        }

        /// <summary>
        /// LoadContent will be called once per game and is the place to load
        /// all of your content.
        /// </summary>
        protected override void LoadContent()
        {
            // Create a new SpriteBatch, which can be used to draw textures.
            spriteBatch = new SpriteBatch(GraphicsDevice);

            // TODO: use this.Content to load your game content here
            pcl.LoadContent(Content);
        }

        /// <summary>
        /// UnloadContent will be called once per game and is the place to unload
        /// all content.
        /// </summary>
        protected override void UnloadContent()
        {
            // TODO: Unload any non ContentManager content here
        }

        /// <summary>
        /// Allows the game to run logic such as updating the world,
        /// checking for collisions, gathering input, and playing audio.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Update(GameTime gameTime)
        {
            // Allows the game to exit
            if (Keyboard.GetState().IsKeyDown(Keys.Escape) ||
                GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed)
                this.Exit();

            // TODO: Add your update logic here
            world.Update(gameTime);

            if (aDown && GamePad.GetState(PlayerIndex.One).Buttons.A == ButtonState.Released)
            {
                tmp = PointCloud.Align(pcl, world.GetMatrx() * icp, pcl, Matrix.Identity);
                icp *= tmp;
            }

            aDown = (GamePad.GetState(PlayerIndex.One).Buttons.A == ButtonState.Pressed);

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.LightSeaGreen);

            // TODO: Add your drawing code here
            pcl.Draw(graphics.GraphicsDevice, Matrix.Identity, view, proj);
            pcl.Draw(graphics.GraphicsDevice, world.GetMatrx() * icp, view, proj);

            base.Draw(gameTime);
        }
    }
}
