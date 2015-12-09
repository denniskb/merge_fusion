float4x4 modelToWorld;
float4x4 eyeToClip;
float3 eye;
float3 forward;

struct VSIn
{
	float4 position : POSITION0;
	float4 normal : NORMAL0;
};

struct VSOut
{
    float4 position : POSITION0;
	float4 worldPosition : TEXCOORD0;
	float4 normal : TEXCOORD1;
};

VSOut VS(VSIn input)
{
	VSOut output;

	float4 worldPos = mul(input.position, modelToWorld);

	output.position = mul(worldPos, eyeToClip);
	output.worldPosition = worldPos;
	output.normal = input.normal;

    return output;
}

float4 PS(VSOut input) : COLOR0
{
	float depthInMeters = dot(input.worldPosition - eye, forward);
	float3 normal = normalize(input.normal.xyz);

    return float4
	(
		depthInMeters,
		normal.x,
		normal.y,
		normal.z
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
