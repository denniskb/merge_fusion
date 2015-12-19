void sort25(float arr[25])
{
	#define CMP_SWAP(i, j) if(arr[i] > arr[j]) { float tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; }

	CMP_SWAP(1,2) CMP_SWAP(4,5) CMP_SWAP(7,8) CMP_SWAP(10,11) CMP_SWAP(13,14) CMP_SWAP(16,17) CMP_SWAP(19,20) CMP_SWAP(21,22) CMP_SWAP(23,24)
	CMP_SWAP(0,2) CMP_SWAP(3,5) CMP_SWAP(6,8) CMP_SWAP(9,11) CMP_SWAP(12,14) CMP_SWAP(15,17) CMP_SWAP(18,20) CMP_SWAP(21,23) CMP_SWAP(22,24)
	CMP_SWAP(0,1) CMP_SWAP(3,4) CMP_SWAP(2,5) CMP_SWAP(6,7) CMP_SWAP(9,10) CMP_SWAP(8,11) CMP_SWAP(12,13) CMP_SWAP(15,16) CMP_SWAP(14,17) CMP_SWAP(18,19) CMP_SWAP(22,23) CMP_SWAP(20,24)
	CMP_SWAP(0,3) CMP_SWAP(1,4) CMP_SWAP(6,9) CMP_SWAP(7,10) CMP_SWAP(5,11) CMP_SWAP(12,15) CMP_SWAP(13,16) CMP_SWAP(18,22) CMP_SWAP(19,23) CMP_SWAP(17,24)
	CMP_SWAP(2,4) CMP_SWAP(1,3) CMP_SWAP(8,10) CMP_SWAP(7,9) CMP_SWAP(0,6) CMP_SWAP(14,16) CMP_SWAP(13,15) CMP_SWAP(18,21) CMP_SWAP(20,23) CMP_SWAP(11,24)
	CMP_SWAP(2,3) CMP_SWAP(8,9) CMP_SWAP(1,7) CMP_SWAP(4,10) CMP_SWAP(14,15) CMP_SWAP(19,21) CMP_SWAP(20,22) CMP_SWAP(16,23)
	CMP_SWAP(2,8) CMP_SWAP(1,6) CMP_SWAP(3,9) CMP_SWAP(5,10) CMP_SWAP(20,21) CMP_SWAP(12,19) CMP_SWAP(15,22) CMP_SWAP(17,23)
	CMP_SWAP(2,7) CMP_SWAP(4,9) CMP_SWAP(12,18) CMP_SWAP(13,20) CMP_SWAP(14,21) CMP_SWAP(16,22) CMP_SWAP(10,23)
	CMP_SWAP(2,6) CMP_SWAP(5,9) CMP_SWAP(4,7) CMP_SWAP(14,20) CMP_SWAP(13,18) CMP_SWAP(17,22) CMP_SWAP(11,23)
	CMP_SWAP(3,6) CMP_SWAP(5,8) CMP_SWAP(14,19) CMP_SWAP(16,20) CMP_SWAP(17,21) CMP_SWAP(0,13) CMP_SWAP(9,22)
	CMP_SWAP(5,7) CMP_SWAP(4,6) CMP_SWAP(14,18) CMP_SWAP(15,19) CMP_SWAP(17,20) CMP_SWAP(0,12) CMP_SWAP(8,21) CMP_SWAP(10,22)
	CMP_SWAP(5,6) CMP_SWAP(15,18) CMP_SWAP(17,19) CMP_SWAP(1,14) CMP_SWAP(7,20) CMP_SWAP(11,22)
	CMP_SWAP(16,18) CMP_SWAP(2,15) CMP_SWAP(1,12) CMP_SWAP(6,19) CMP_SWAP(8,20) CMP_SWAP(11,21)
	CMP_SWAP(17,18) CMP_SWAP(2,14) CMP_SWAP(3,16) CMP_SWAP(7,19) CMP_SWAP(10,20)
	CMP_SWAP(2,13) CMP_SWAP(4,17) CMP_SWAP(5,18) CMP_SWAP(8,19) CMP_SWAP(11,20)
	CMP_SWAP(2,12) CMP_SWAP(5,17) CMP_SWAP(4,16) CMP_SWAP(3,13) CMP_SWAP(9,19)
	CMP_SWAP(5,16) CMP_SWAP(3,12) CMP_SWAP(4,14) CMP_SWAP(10,19)
	CMP_SWAP(5,15) CMP_SWAP(4,12) CMP_SWAP(11,19) CMP_SWAP(9,16) CMP_SWAP(10,17)
	CMP_SWAP(5,14) CMP_SWAP(8,15) CMP_SWAP(11,18) CMP_SWAP(10,16)
	CMP_SWAP(5,13) CMP_SWAP(7,14) CMP_SWAP(11,17)
	CMP_SWAP(5,12) CMP_SWAP(6,13) CMP_SWAP(8,14) CMP_SWAP(11,16)
	CMP_SWAP(6,12) CMP_SWAP(8,13) CMP_SWAP(10,14) CMP_SWAP(11,15)
	CMP_SWAP(7,12) CMP_SWAP(9,13) CMP_SWAP(11,14)
	CMP_SWAP(8,12) CMP_SWAP(11,13)
	CMP_SWAP(9,12)
	CMP_SWAP(10,12)
	CMP_SWAP(11,12)
}

// returns uniformly distributed random number \in [0, 1]
float rnd(float2 texcoord)
{
	const float M_PI = 3.14f;

	const float4 a = float4((M_PI * M_PI) * (M_PI * M_PI), exp(4.0f), pow(13.0f, M_PI / 2.0f), sqrt(1997.0f));
	float4 result = float4(texcoord, texcoord);
	
	for(int i = 0; i < 2; i++) 
	{
		result.x = frac(dot(result, a));
		result.y = frac(dot(result, a));
		result.z = frac(dot(result, a));
		result.w = frac(dot(result, a));
	}

	return result.w;
}

/*
TODO: Implement Kinect noise model:

'''
Created on Nov 10, 2015
 
@author: kyriazis
'''
 
import numpy as np
import cv2 as cv
from abc import abstractmethod
from scipy.ndimage.filters import median_filter
 
def NormalZ2Angle(z):
    return np.arccos(-z)
    
class MicrosoftKinectNoiseMaker(object):
    """
    Noise formulas from
    
    Nguyen, Chuong V., Shahram Izadi, and David Lovell.
    "Modeling kinect sensor noise for improved 3d reconstruction and tracking."
    3D Imaging, Modeling, Processing, Visualization and Transmission (3DIMPVT), 2012
    Second International Conference on. IEEE, 2012.
    
    This class also adds:
    - missing measurements for surface angles greater than a threshold
    - noise diffusion through median filtering
    """
    def __init__(self, angleCuttoff=1.4, medianWindow=(5,5)):
        self.angleCutoff = angleCuttoff
        self.medianWindow = medianWindow
        
    def __noiseZ__(self, z, theta):
        return 0.0012 + 0.0019 * ((z - 0.4) **2) + (0.0001 / np.sqrt(z)) * (theta ** 2) / ((np.pi/2 - theta)**2)
    
    def __noiseL__(self, theta):
        return 0.8 + 0.035 * theta / (np.pi/2 - theta)
    
    def __generateNoise__(self, valid, z, theta):
        height, width = z.shape
        
        # the formula requires depth in meters
        nZ = self.__noiseZ__(z[valid]/1000, theta[valid])
        nL = self.__noiseL__(theta[valid])
        
        NZ = np.zeros(z.shape)
        NZ[valid] = nZ
        NL = np.zeros(z.shape)
        NL[valid] = nL
        
        [X,Y] = np.mgrid[0:width,0:height]
        X = X.transpose()
        Y = Y.transpose()
        
        X = X + np.random.randn(*NL.shape) * (NL)
        Y = Y + np.random.randn(*NL.shape) * (NL)
            
        z[theta > self.angleCutoff] = 0
        zz = cv.remap(z + np.random.randn(*NZ.shape) * (NZ),
                      X.astype(np.float32),
                      Y.astype(np.float32),
                      cv.INTER_NEAREST)
        zz = median_filter(zz, self.medianWindow)
        
        return zz
    
    def simulate(self, mask, positionMap, normalMap):
        # Assuming positionMap is in mm.
        return self.__generateNoise__(mask, positionMap[:,:,2], NormalZ2Angle(normalMap[:,:,2]))
*/



Texture2D depth;
int iFrame;

SamplerState depthSampler
{
	Filter = MIN_MAG_MIP_POINT;
};

struct VSOut
{
    float4 position : POSITION0;
	float2 texcoord : TEXCOORD0;
};



VSOut VS(float4 position : POSITION0)
{
	VSOut output;

	float2 texcoord = position;
	texcoord.y = -texcoord.y;
	texcoord = texcoord * 0.5f + 0.5f;

	output.position = position;
	output.texcoord = texcoord;

    return output;
}



float4 PSColor(VSOut input) : COLOR0
{
	float depthInMeters = depth.Sample(depthSampler, input.texcoord).r;
	
	float depthRGB = (depthInMeters - 0.8f) / 3.2f; // mapped to [0.8m, 4m]
	
	return depthRGB;
}

float4 PSNoise(VSOut input) : COLOR0
{
	float median[25];

	{
		int kernelSize = 2;
		float2 px = float2(1.0f / 640, 1.0f / 480); // TODO: Generalize
		int i = 0;
		for (int y = -kernelSize; y <=kernelSize; y++)
		{
			for (int x = -kernelSize; x <=kernelSize; x++)
			{
				median[i++] = depth.Sample(depthSampler, input.texcoord + float2(x, y) * px).x;
			}
		}
	}

	sort25(median);

	return median[12];
}



technique Depth2Color
{
    pass Pass1
    {
        VertexShader = compile vs_3_0 VS();
        PixelShader = compile ps_3_0 PSColor();
    }
}

technique AddNoise
{
	pass Pass1
	{
		VertexShader = compile vs_3_0 VS();
        PixelShader = compile ps_3_0 PSNoise();
	}
}