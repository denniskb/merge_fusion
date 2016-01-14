void sort25(inout float arr[25])
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
float rnd_uniform(float2 texcoord)
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

// returns a normally (mu=0, sigma=1) distributed random number \in [-3, 3]
float rnd_normal(float2 texcoord)
{
	float result = rnd_uniform(texcoord) - 0.5f;
	int negative = result < 0.0f;
	return (result * result) * (1 - 2 * negative) * 12;
}

// computes the maximally possible noise in meters depending on distance (in meters) z and surface angle theta
// Original code by Kyriazis
float noiseZ(float z, float theta)
{
	const float PI = 3.14f;

	return
	0.0012 +
	0.0019 * ((z - 0.4f) * (z - 0.4f)) + 
	(0.0001 / sqrt(z)) * (theta * theta) / ((PI/2 - theta) * (PI/2 - theta));
}



Texture2D depth;
Texture2D pos;
int iFrame;
float3 eye;
float3 forward;

sampler depthSampler = sampler_state
{
	texture = <depth>;
	Filter = MIN_MAG_MIP_POINT;
};

sampler posSampler = sampler_state
{
	texture = <pos>;
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



float4 PSNoise(VSOut input) : COLOR0
{
	float4 sample = tex2D(depthSampler, input.texcoord);
	float3 eyeVec = normalize(eye - tex2D(posSampler, input.texcoord).xyz);
	
	float theta = acos(dot(sample.yzw, eyeVec));

	float z = 
	sample.x
	+ 2 * noiseZ(sample.x, theta) 
	* rnd_normal(input.texcoord + iFrame) 
	* (sample.x > 0.0f);
	
	return float4(z, sample.yzw);
}

float4 PSMedian(VSOut input) : COLOR0
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
				median[i++] = tex2D(depthSampler, input.texcoord + float2(x, y) * px).x;
			}
		}
	}
	
	sort25(median);
	
	return median[12];
}

float4 PSColor(VSOut input) : COLOR0
{
	float depthInMeters = tex2D(depthSampler, input.texcoord).r;

	// map [0.4m, 4m] to [0, 1]
	float depth_norm = (depthInMeters - 0.4f) / 3.6f;

	float r = min( 1.0f, ( max( 0.5f, depth_norm ) - 0.5f ) * 6.0f );
    float g = min( 1.0f, depth_norm * 3.0f ) - min( 1.0f, max( 0.0f, depth_norm - 0.666f ) * 3.0f );
    float b = max( 0.0f, 1.0f - max( 0.0f, depth_norm - 0.333f ) * 6.0f );

	return float4(r, g, b, 1.0f) * (depthInMeters > 0);
}



technique AddNoise
{
	pass Pass1
	{
		VertexShader = compile vs_3_0 VS();
        PixelShader = compile ps_3_0 PSNoise();
	}
}

technique ComputeMedian
{
	pass Pass1
	{
		VertexShader = compile vs_3_0 VS();
		PixelShader = compile ps_3_0 PSMedian();
	}
}

technique Depth2Color
{
    pass Pass1
    {
        VertexShader = compile vs_3_0 VS();
        PixelShader = compile ps_3_0 PSColor();
    }
}