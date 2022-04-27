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
    private static double lastRenderUpdateThreshold = 1e-2f;
    
    
    
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
        200, 700, 1300, 2000, 3700, 5200, 6000, 7500, 10000, 33000, 1e6,
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
        else if (this.mass > 119) // 62.7 jupiter masses / 0.06 solar masses, dominated by fusion / gravity pressure
        {
            this.radius = 0.00089 * Pow(this.mass, 0.88) * 17;
        }
        else if (this.mass > 0.6) // 100 earth masses / 0.31 jupiter masses -> jupiter-class, dominated by degeneracy pressure
        {
            this.radius = 0.0725 * Pow(this.mass, -0.04) * 17; // 0.0600 at upper end
        }
        else  // less than 100 earth masses, assume neptune class -> dominated by gas envelopes
        {
            this.radius = 0.1 * Pow(mass, 0.59) * 17; // 0.0740 at upper end
        }
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
