using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using static System.Math;

public class Particle : Object
{

    public static readonly double PI = 3.1415926535;
    public static readonly double L_Sun = 3.83E26;
    public static readonly double r_Jupiter = 6.9173E7;
    public static readonly double SB = 5.670373E-8;
    public static readonly double radiusScale = 1f; // display objects larger than they actually are by this factor

    public volatile GameObject gameObject;
    private Material material;
    private volatile Transform transform;
    private readonly Main main;

    public new string name;
    public int id;

    public double x;
    public double y;
    public double vx;
    public double vy;
    public double mass; // units of 1e27 kg

    public double temperature;
    public double lifeLeft;
    public double radius;
    public double luminosity;
    public string starClass;
    public bool isBlackHole;
    public Color color;
    
    public bool removed;
    public bool massDirty = true; // dirty flag for mass
    public double lastRenderX = double.PositiveInfinity;
    public double lastRenderY = double.PositiveInfinity;
    private static double lastRenderUpdateThreshold = 1e-4f;
    
    public Particle(string name, int id, double x, double y, double vx, double vy, double mass, Main main, bool isBlackHole = false)
    {
        this.name = name;
        this.id = id;
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.mass = mass;

        this.gameObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        DestroyImmediate(this.gameObject.GetComponent<SphereCollider>());
        this.gameObject.name = name;
        this.gameObject.transform.position = new Vector2((float)x,(float)y);
        this.gameObject.GetComponent<MeshRenderer>().sharedMaterial = main.yellowDwarfMaterial;
        this.material = this.gameObject.GetComponent<MeshRenderer>().sharedMaterial;

        this.transform = this.gameObject.transform;
        this.isBlackHole = isBlackHole;
        this.main = main;
        this.lifeLeft = 1;
    }

    private static double[] starClassMinimumTemperatures = {
        200, 700, 1300, 2000, 3700, 5200, 6000, 7500, 10000, 33000, 60000,
    };

    private static string[] starClasses = {
        "P", "Y", "T", "L", "M", "K", "G", "F", "A", "O", "WR"
    };

    public void updateNonPhysics(double stepSize) {
        if (massDirty)
        {
            string oldStarClass = starClass;
            updateRadius();
            updateLuminosity();
            updateTemperature();
            updateColor();
            updateStarClass();
            updateGameObject(oldStarClass != starClass);
            massDirty = false;
            float diameter = (float)(2 * radiusScale * radius * Main.displayScale);
            transform.localScale = new Vector3(diameter, diameter, diameter);
        }
        updateLife(stepSize);
        
    }

    public void updatePhysics(double stepSize)
    {
        x += stepSize * vx;
        y += stepSize * vy;
        double r2FromLastRender = (lastRenderX - x) * (lastRenderX - x) + (lastRenderY - y ) * (lastRenderY - y);
        if (r2FromLastRender > lastRenderUpdateThreshold)
        {
            transform.position = new Vector2((float)x, (float)y);
            lastRenderX = x;
            lastRenderY = y;
        }
        
    }

    private void updateGameObject(bool classChanged)
    {
        if (gameObject != null) {
            MaterialPropertyBlock newBlock = new MaterialPropertyBlock();
            newBlock.SetFloat("_BaseTemperature", (float)temperature);
            if (classChanged)
            {
                switch (starClass)
                {
                    case "O":
                    case "B":
                    case "A":
                        // newBlock.SetTexture("Texture", main.yellowDwarfMaterial.mainTexture);
                        break;
                    case "F":
                    case "G":
                    case "K":
                        // newBlock.SetTexture("Texture", main.yellowDwarfMaterial.mainTexture);
                        break;
                    case "M":
                        // newBlock.SetTexture("Texture", main.yellowDwarfMaterial.mainTexture);
                        break;
                    case "L":
                    case "T":
                    case "Y":
                        // newBlock.SetTexture("Texture", main.yellowDwarfMaterial.mainTexture);
                        break;
                    default:
                        // newBlock.SetTexture("Texture", main.yellowDwarfMaterial.mainTexture);
                        break;
                }
            }

            gameObject.GetComponent<MeshRenderer>().SetPropertyBlock(newBlock);
        }
    }

    private void updateStarClass()
    {
        if (isBlackHole)
        {
            starClass = "X";
        }
        for (int i = 0; i < starClassMinimumTemperatures.Length; i++)
        {
            if (temperature < starClassMinimumTemperatures[i])
            {
                starClass = starClasses[i];
                break;
            }
        }
    }

    private void updateTemperature()
    {
        /* L = SB * 4pi * r^2 * T^4
         * T^4 = L / ( SB * 4 * pi * r^2)
         * T = pow(L / SB / 4 / pi / r^2, 0.25)
         */
        double Rt = this.radius * r_Jupiter;
        this.temperature = Pow(this.luminosity / 4 / PI / Rt / Rt / SB, 0.25);
    }

    private void updateLuminosity() {
        double mFactor = this.mass / 1988;
        if (mFactor > 20)
        {
            this.luminosity = 58.8 * Pow(mFactor, 2.3) * L_Sun;
        }
        else if (mFactor > 2)
        {
            this.luminosity = 1.82 * Pow(mFactor, 3.46) * L_Sun;
            //returns 1.5*2^3.5 at 2
        }
        else if (mFactor > 1)
        {
            this.luminosity = Pow(mFactor, 4.33) * L_Sun;
        }
        else if (mFactor > 0.4)
        {
            this.luminosity = Pow(mFactor, 3.75) * L_Sun;
        }
        else
        {
            this.luminosity = Pow(mFactor, 4) * L_Sun;
        }
    }


    private void updateRadius()
    {
        if (this.isBlackHole)
        {
            this.radius = Sqrt(this.mass) / 5000;
        }
        // above about 150 solar masses i.e. 300 000 e27 kg, further growth becomes impossible as extreme temperatures expel the outer layers of a star.
        // But we don't model that here :)
        else if (this.mass > 159) // 84 jupiter masses i.e. 0.08 solar masses: stars, heat from fusion gradually dominates gravitational effects
        {
            this.radius = 0.0317 * Pow(this.mass, 0.88) * 0.43;
        }
        else if (this.mass > 0.778) // 130 earth masses i.e. 0.41 jupiter masses: jupiters and brown dwarfs, gravity slowly begins to shrink the object as mass increases
        {
            this.radius = 3.36 * Pow(this.mass, -0.04) * 0.43; // 3.01 at upper end before scaling
        }
        else if (this.mass > 0.012)  // 2 earth masses: gas/ice dwarfs such as neptune, runaway increase in size due to rapid accumulation of low-density volatiles
        {
            this.radius = 3.94 * Pow(mass, 0.59) * 0.43; // 3.40 at upper end before scaling
        }
        else // less than 2 earth masses; radius grows approximately with cube root of mass, with only minor corrections due to gravitational compression
        {
            this.radius = 1.0 * Pow(mass, 0.28) * 0.43; // 0.290 at upper end before scaling
        }
        // below about 1e-6 units, objects begin to deviate significantly from round, and mean radius can't be predicted from mass even with known composition
        // but we don't model that here either
    }

    private void updateColor()
    {
        double red, green, blue, temp;
        red = green = blue = 0;
        temp = this.temperature / 100;
        if (Main.colorscheme == ColorScheme.NATURAL)
        {
            if (this.isBlackHole)
            {
                this.color = new Color(0, 0, 0);
            }
            else
            {
                if (temp < 66)
                {
                    red = 255;
                }
                else
                {
                    red = temp - 60;
                    red = 329.6987 * Pow(red, -0.1332);
                    if (red < 0)
                    {
                        red = 0;
                    }
                    if (red > 255)
                    {
                        red = 255;
                    }
                }
                if (temp <= 66)
                {
                    green = temp;
                    green = 99.47 * Log(green) - 161.1196;
                }
                else
                {
                    green = temp - 60;
                    green = 288.122 * Pow(green, -0.0755);
                }
                if (green < 0)
                {
                    green = 0;
                }
                if (green > 255)
                {
                    green = 255;
                }
                if (temp >= 66)
                {
                    blue = 255;
                }
                if (temp <= 19 && temp > 10)
                {
                    blue = 50 - temp * temp;
                }
                if (temp < 10)
                {
                    blue = 100 - temp * temp;
                }
                if (temp < 66 && temp > 19)
                {
                    blue = temp - 10;
                    blue = 138.5177 * Log(blue) - 305.0448;
                }
                if (blue < 0)
                {
                    blue = 0;
                }
                if (blue > 255)
                {
                    blue = 255;
                }
                if (temp < 10)
                {
                    this.color = new Color((float)(red * temp * temp * temp / 1e3), (float)(green * temp * temp * temp / 1e3), (float)(blue));
                }
                else
                {
                    this.color = new Color((float)(red), (float)(green), (float)(blue));
                }
            }
        }
        else if (Main.colorscheme == ColorScheme.VELOCITY)
        {
            double v = Sqrt(vx * vx + vy * vy);
            red = green = blue = 1000 * v;
            if (red > 255)
            {
                red = 255;
            }
            if (green > 255)
            {
                green = 255;
            }
            if (blue > 255)
            {
                blue = 255;
            }
            this.color = new Color((float)red, (float)green, (float)blue);
        }
        else if (Main.colorscheme == ColorScheme.LIFETIME)
        {
            red = green = blue = lifeLeft / 4000000;
            if (red > 255)
            {
                red = 255;
            }
            if (green > 255)
            {
                green = 255;
            }
            if (blue > 255)
            {
                blue = 255;
            }
            this.color = new Color((float)red, (float)green, (float)blue);
        }
        else
        {
            this.color = new Color(0, 255, 0);
        }
    }

    private void updateLife(double stepSize)
    {
        if (this.isBlackHole)
        {
            return;
        }
        double penalty = mass * mass * mass / 1.5e12 * stepSize;
        this.lifeLeft -= penalty;
        if (lifeLeft < 0)
        {
            splinter();
        }
    }

    public void remove()
    {
        this.removed = true;
        DestroyImmediate(this.gameObject);
        this.gameObject = null;
    }

    private void splinter()
    {
    }
}
