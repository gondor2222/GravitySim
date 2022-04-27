Shader "Unlit/StarSurfaceShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _BaseTemperature("BaseTemperature", Float) = 5000
    }
    SubShader
    {
        Tags {"RenderType" = "Opaque" "LightMode" = "ForwardBase"}
            Lighting Off
            ZWrite Off
            Blend SrcAlpha OneMinusSrcAlpha
            CGPROGRAM
                #pragma surface surf StarSurfaceShader alpha

                #include "UnityCG.cginc"
                #include "Lighting.cginc"

                half4 LightingStarSurfaceShader(SurfaceOutput s, half3 viewDir, UnityGI gi) {
                    return half4(s.Albedo, s.Alpha);
                }

                inline void LightingStarSurfaceShader_GI(SurfaceOutput s, UnityGIInput data, inout UnityGI gi) {
                    //
                }

                struct Input
                {
                    float2 uv_MainTex;
                };

                UNITY_INSTANCING_BUFFER_START(Props)
                UNITY_DEFINE_INSTANCED_PROP(half, _BaseTemperature)
                UNITY_INSTANCING_BUFFER_END(Props)
                sampler2D _MainTex;

                float3 colorFromTemperature(float temperature) {
                    float temp = temperature / 100;
                    float red, green, blue;
                    if (temp > 66) {
                        red = 329.6987 * pow(temp - 60, -0.1332);
                    }
                    else if (temp > 10)
                    {
                        red = 255;
                    } else {
                        red = 255 * temp * temp / 100;
                    }

                    if (temp <= 66)
                    {
                        green = 99.47 * log(temp) - 141.1196;
                    }
                    else if (temp > 10)
                    {
                        green = 288.122 * pow(temp - 60, -0.0755);
                    } else
                    {
                        green = 288.122 * pow(temp - 60, -0.0755) * temp * temp / 100;
                    }

                    if (temp >= 66)
                    {
                        blue = 255;
                    }
                    else if (temp > 19)
                    {
                        blue = 138.5177 * log(temp - 10) - 305.0448;
                    }
                    else if (temp > 10)
                    {
                        blue = 50 - temp * temp;
                    }
                    else
                    {
                        blue = 10 * temp - temp * temp * temp / 10;
                    }
                    

                    red = clamp(red, 0, 255);
                    green = clamp(green, 0, 255);
                    blue = clamp(blue, 0, 255);

                    return float3(red, green, blue);
                }

                void surf(Input IN, inout SurfaceOutput o) {
                    float temp = (tex2D(_MainTex, IN.uv_MainTex) + 0.5) * UNITY_ACCESS_INSTANCED_PROP(Props, _BaseTemperature);

                    float3 color = colorFromTemperature(temp) / 255;
                    o.Albedo = color;
                    o.Alpha = 1;
                }

                ENDCG
    }
}
