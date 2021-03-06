float4x4 local;
float4x4 world;
float4x4 viewProjection;
float3 eye;
float3 forward;

float4x4 joints[32];

struct VSIn
{
	float4 position: POSITION0;
	uint4 boneIndices : BLENDINDICES0;
	float4 boneWeights : BLENDWEIGHT0;
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

	float4 skinnedModelPos0 = mul(modelPos, joints[input.boneIndices.x]);
	float4 skinnedModelPos1 = mul(modelPos, joints[input.boneIndices.y]);
	float4 skinnedModelPos2 = mul(modelPos, joints[input.boneIndices.z]);
	float4 skinnedModelPos3 = mul(modelPos, joints[input.boneIndices.w]);

	float4 skinnedModelPos =
		skinnedModelPos0 * input.boneWeights.x +
		skinnedModelPos1 * input.boneWeights.y +
		skinnedModelPos2 * input.boneWeights.z +
		skinnedModelPos3 * input.boneWeights.w;

	float4 worldPos = mul(skinnedModelPos, world);

	output.position = mul(worldPos, viewProjection);
	output.worldPosition = worldPos;

    return output;
}

float4 PS(VSOut input) : COLOR0
{
	float depthInMeters = dot(input.worldPosition - eye, forward);
	float depthRGB = (depthInMeters - 0.8f) / 3.2f; // mapped to [0.8m, 4m]

    return float4
	(
		depthRGB,
		depthRGB,
		depthRGB,
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
