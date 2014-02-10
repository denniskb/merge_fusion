﻿using System.IO;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

namespace poly2depth
{
    // TODO: Test two recordings in a row!
    class Recorder
    {
        private FileStream fs;
        private BinaryWriter bw;
        private int frameCount;

        private bool buttonAWasDown;

        private Vector4[] tmpBufferRT;
        private byte[] tmpBufferDepth;

        public Recorder()
        {
            fs = null;
            bw = null;
            frameCount = 0;

            buttonAWasDown = false;

            tmpBufferRT = new Vector4[640 * 480];
            tmpBufferDepth = new byte[640 * 480 * 2];
        }

        public void Update()
        {
            bool buttonAIsUp = GamePad.GetState(Microsoft.Xna.Framework.PlayerIndex.One).Buttons.A == ButtonState.Released;

            if (buttonAIsUp && buttonAWasDown)
                if (IsRecording())
                    StopRecording();
                else
                    StartRecording();

            buttonAWasDown = !buttonAIsUp;
        }

        public bool IsRecording()
        {
            return fs != null;
        }



        private void StartRecording()
        {
            fs = File.Open(Path.GetTempFileName(), FileMode.Truncate);
            bw = new BinaryWriter(fs);

            bw.Write("KPPL raw depth\n".ToCharArray()); // magic
            bw.Write((int)1);   // version
            bw.Write((int)-1);  // #frames (reserved for later)
        }

        // TODO: Record at certain framerate! (possibly every nth frame)
        public void RecordFrame(RenderTarget2D rt, Matrix view)
        {
            if (!IsRecording())
                return;

            rt.GetData<Vector4>(tmpBufferRT);
            for (int i = 0; i < 640 * 480; i++)
            {
                ushort depth = (ushort)tmpBufferRT[i].X;
                // little endian
                tmpBufferDepth[2*i] = (byte)(depth & 255);
                tmpBufferDepth[2 * i + 1] = (byte)(depth >> 8);
            }

            bw.Write(view.M11);
            bw.Write(view.M12);
            bw.Write(view.M13);
            bw.Write(view.M14);

            bw.Write(view.M21);
            bw.Write(view.M22);
            bw.Write(view.M23);
            bw.Write(view.M24);

            bw.Write(view.M31);
            bw.Write(view.M32);
            bw.Write(view.M33);
            bw.Write(view.M34);

            bw.Write(view.M41);
            bw.Write(view.M42);
            bw.Write(view.M43);
            bw.Write(view.M44);

            bw.Write(tmpBufferDepth);
         
            frameCount++;
        }

        private void StopRecording()
        {
            bw.Seek(19, SeekOrigin.Begin);
            bw.Write(frameCount);
            bw.Close();
            fs.Close();

            var dialog = new System.Windows.Forms.SaveFileDialog();
            dialog.Filter = "KPPL raw depth streams|*.depth|All files|*";
            dialog.RestoreDirectory = true;

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                File.Move(fs.Name, dialog.FileName);
            else
                File.Delete(fs.Name);

            bw = null;
            fs = null;
            frameCount = 0;
        }
    }
}
