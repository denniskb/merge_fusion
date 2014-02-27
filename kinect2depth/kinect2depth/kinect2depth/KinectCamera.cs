using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

using Microsoft.Kinect;

namespace poly2depth
{
    /// <summary>
    /// Simmulates a Kinect camera in far mode (0.8m to 4.0m range)
    /// </summary>
    class KinectCamera
    {
        short[] cache;

        public KinectCamera()
        {
            cache = new short[640 * 480];

            if (KinectSensor.KinectSensors.Count > 0)
                if (KinectSensor.KinectSensors[0].Status == KinectStatus.Connected)
                {
                    KinectSensor.KinectSensors[0].Start();
                    
                    try
                    {
                        KinectSensor.KinectSensors[0].DepthStream.Range = DepthRange.Near;
                    }
                    catch (System.Exception e) { } // we're ok with far mode if near mode isn't available

                    KinectSensor.KinectSensors[0].DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);
                }
        }

        public void NextFrame(float[] outFrame)
        {
            if (KinectSensor.KinectSensors.Count > 0)
                if (KinectSensor.KinectSensors[0].Status == KinectStatus.Connected)
                {
                    DepthImageFrame frame = KinectSensor.KinectSensors[0].DepthStream.OpenNextFrame(1000);
                    frame.CopyPixelDataTo(cache);
                    frame.Dispose();

                    for (int i = 0; i < cache.Length; i++)
                        outFrame[i] = (cache[i] >> 3) * 0.001f;
                }
        }

        public void Stop()
        {
            if (KinectSensor.KinectSensors.Count > 0)
                if (KinectSensor.KinectSensors[0].Status == KinectStatus.Connected)
                {
                    KinectSensor.KinectSensors[0].DepthStream.Disable();
                    KinectSensor.KinectSensors[0].Stop();
                }
        }

        public Matrix GetView()
        {
            return Matrix.CreateLookAt
            (
                new Vector3( 0.0f, 0.0f, 1.0f ),
                new Vector3( 0.0f, 0.0f, 0.0f ),
                new Vector3( 0.0f, 1.0f, 0.0f )
            );
        }
    }
}