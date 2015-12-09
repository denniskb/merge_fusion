Texture2D depth;

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
	
	return float4
	(
		depthRGB,
		depthRGB,
		depthRGB,
		1
	);
}

float4 PSNoise(VSOut input) : COLOR0
{
	float4 depthInMeters = depth.Sample(depthSampler, input.texcoord);

	// median
	int kernelSize = 2;
	float2 px = 1.0f / float2(640, 480);
	float avg = 0;
	int i = 0;
	for (int y = -kernelSize; y <=kernelSize; y++)
	{
		for (int x = -kernelSize; x <=kernelSize; x++)
		{
			float d = depth.Sample(depthSampler, input.texcoord + float2(x, y) * px);
			avg += d;
		}
	}
	avg /= (2 * kernelSize + 1) * (2 * kernelSize + 1);

	float err = 10;
	float median;
	for (int y = -kernelSize; y <=kernelSize; y++)
	{
		for (int x = -kernelSize; x <=kernelSize; x++)
		{
			float d = depth.Sample(depthSampler, input.texcoord + float2(x, y) * px);
			if (abs(d - avg) < err)
			{
				median = d;
				err = abs(d - avg);
			}
		}
	}

	return median;
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