float4x4 local;
float4x4 world;
float4x4 viewProjection;
float3 eye;
float3 forward;

struct VSIn
{
	float4 position: POSITION0;
};

struct VSOut
{
    float4 position : POSITION0;
	float4 worldPosition : TEXCOORD0;
};

VSOut VS(VSIn input)
{
	VSOut output;

	float4 modelPos = mul(input.position, local);
	float4 worldPos = mul(modelPos, world);

	output.position = mul(worldPos, viewProjection);
	output.worldPosition = worldPos;

    return output;
}

float4 PS(VSOut input) : COLOR0
{
	float depthInMeters = dot(input.worldPosition - eye, forward);

    return float4
	(
		depthInMeters,
		depthInMeters,
		depthInMeters,
		1
	);
}

technique Technique1
{
    pass Pass1
    {
        VertexShader = compile vs_2_0 VS();
        PixelShader = compile ps_2_0 PS();
    }
}
