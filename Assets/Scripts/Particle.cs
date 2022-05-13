using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using static System.Math;

public class Particle
{

    public static readonly double PI = 3.1415926535;
    public static readonly double L_Sun = 3.83e26;
    public static readonly double r_Jupiter = 6.9173e7; // meters
    public static readonly double r_Sun = 6.957e8;
    public static readonly double SB = 5.670373e-8;
    public static bool exaggerateSmallParticles = true; // display objects larger than they actually are by this factor
    public static double minDisplayScale = 1e-10;

    public volatile GameObject gameObject;
    private volatile Transform transform;

    public string name;
    public int id;

    public double x;
    public double y;
    public double vx;
    public double vy;
    public double mass; // units of 1e27 kg
    public bool isBlackHole;

    public double temperature;
    public double lifeLeft;
    public double radius;
    public double luminosity;
    public double surfaceTemperature;
    public double albedo = 0.0;
    public string objectClass;
    public Color color;
    
    public bool removed;
    public bool massDirty = false; // dirty flag for mass
    private double lastRenderX = double.PositiveInfinity;
    private double lastRenderY = double.PositiveInfinity;
    private double lastRenderDX = 0;
    private double lastRenderDY = 0;
    public Vector3 lastRenderDifference = new Vector3();

    public Particle(string name) {
        this.name = name;
        this.removed = true;
    }
    
    public Particle(string name, int id, double x, double y, double vx, double vy, double mass, Mesh meshToUse, bool isBlackHole = false) {
        this.name = name;
        this.id = id;
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.mass = mass;

        this.gameObject = new GameObject(name);
        this.gameObject.AddComponent<MeshFilter>();
        this.gameObject.GetComponent<MeshFilter>().mesh = meshToUse;
        this.gameObject.AddComponent<MeshRenderer>();
        this.gameObject.transform.position = new Vector2((float)(x / Main.distanceScale),(float)(y / Main.distanceScale));
        this.gameObject.GetComponent<MeshRenderer>().sharedMaterial = Main.yellowDwarfMaterial;

        this.transform = this.gameObject.transform;
        this.isBlackHole = isBlackHole;
        this.lifeLeft = 1;

        updateRadius();
        updateLuminosity();
        updateTemperature();
        this.surfaceTemperature = this.temperature;
        updateColor();
        updateStarClass();
        updateGameObject(true);
        updateLighting();
        updateScale();
    }

    private static double[] starClassMinimumTemperatures = {
          200, // < 200 = P
          700, // < 700 = Y
         1300, // < 1300 = T
         2000, // < 2000 = L
         3700, // < 3700 = M
         5200, // < 5200 = K
         6000, // < 6000 = G
         7500, // < 7500 = F
        10000, // < 10000 = A
        30000, // < 30000 = B
        60000, // < 60000 = O
        // > 60000 = WR
    };

    private static string[] starClasses = {
        "P-class Brown Dwarf",
        "Y-class Brown Dwarf",
        "T-class Brown Dwarf",
        "L-class Brown Dwarf",
        "M-class Star",
        "K-class Star",
        "G-class Star",
        "F-class Star",
        "A-class Star",
        "B-class Star",
        "O-class Star",
        "WR-class Star"
    };

    public void updateNonPhysics(double stepSize) {
        if (massDirty) {
            string oldStarClass = objectClass;
            updateRadius();
            updateLuminosity();
            updateTemperature();
            updateColor();
            updateStarClass();
            updateGameObject(!objectClass.Equals(oldStarClass));
            massDirty = false;
            updateScale();
        }
        updateLighting();
        updateLife(stepSize);
        
    }

    public void updateScale() {
        double diameter = 2 * radius / Main.distanceScale;

        if (exaggerateSmallParticles && diameter < 15 * minDisplayScale) {
            diameter = 15 * minDisplayScale;
        }

        transform.localScale = (float)diameter * new Vector3(1, 1, 1);
    }

    public bool updatePhysics(double stepSize) {
        x += stepSize * vx;
        y += stepSize * vy;
        double r2FromLastRender = (lastRenderX - x) * (lastRenderX - x) + (lastRenderY - y ) * (lastRenderY - y);
        if (r2FromLastRender > minDisplayScale * minDisplayScale) {
            updateTransform();
            return true;
        }

        return false;
        
    }

    public void updateTransform() {
        Vector3 oldPosition = transform.position;
        transform.position = new Vector2((float)(x / Main.distanceScale), (float)(y / Main.distanceScale));
        lastRenderDX = x - lastRenderX;
        lastRenderDY = y - lastRenderY;
        lastRenderX = x;
        lastRenderY = y;
        lastRenderDifference = transform.position - oldPosition;
}

    private void updateGameObject(bool classChanged) {
        if (gameObject != null) {
            if (classChanged) {
                switch (objectClass) {
                    case "O-class Star":
                    case "B-class Star":
                    case "A-class Star":
                        this.gameObject.GetComponent<MeshRenderer>().sharedMaterial = Main.blueDwarfMaterial;
                        break;
                    case "F-class Star":
                    case "G-class Star":
                    case "K-class Star":
                        this.gameObject.GetComponent<MeshRenderer>().sharedMaterial = Main.yellowDwarfMaterial;
                        break;
                    case "M-class Star":
                        this.gameObject.GetComponent<MeshRenderer>().sharedMaterial = Main.redDwarfMaterial;
                        break;
                    case "L-class Brown Dwarf":
                    case "T-class Brown Dwarf":
                    case "Y-class Brown Dwarf":
                        this.gameObject.GetComponent<MeshRenderer>().sharedMaterial = Main.brownDwarfMaterial;
                        break;
                    case "Gas Planet":
                    case "Rocky Planet":
                        this.gameObject.GetComponent<MeshRenderer>().sharedMaterial = Main.plainThermalMaterial;
                        break;
                    case "Black Hole":
                    default:
                        gameObject.GetComponent<MeshRenderer>().sharedMaterial = Main.blankMaterial;
                        break;
                }
            }
        }
    }

    private void updateLighting() {
        if (gameObject != null) {
            MaterialPropertyBlock newBlock = new MaterialPropertyBlock();
            newBlock.SetFloat("_BaseTemperature", (float)surfaceTemperature);
            gameObject.GetComponent<MeshRenderer>().SetPropertyBlock(newBlock);
        }
    }

    private void updateStarClass() {
        if (isBlackHole) {
            objectClass = "Black Hole";
        }
        if (this.mass < 2 * Main.earthMass) {
            objectClass = "Rocky Planet";
        } else if (this.mass < 5 * Main.jupiterMass) {
            objectClass = "Gas Planet";
        } else {
            for (int i = 0; i < starClassMinimumTemperatures.Length; i++) {
                if (temperature < starClassMinimumTemperatures[i]) {
                    objectClass = starClasses[i];
                    return;
                }
            }
            objectClass = starClasses[starClassMinimumTemperatures.Length];
        }
    }

    private void updateTemperature() {
        /* L = SB * 4pi * r^2 * T^4
         * T^4 = L / ( SB * 4 * pi * r^2)
         * T = pow(L / SB / 4 / pi / r^2, 0.25)
         */
        double Rt = this.radius;
        this.temperature = Pow(this.luminosity / 4 / PI / Rt / Rt / SB, 0.25);
    }

    private void updateLuminosity() {
        double mFactor = this.mass / Main.solarMass;
        if (mFactor > 20) {
            this.luminosity = 58.8 * Pow(mFactor, 2.3) * L_Sun;
        }
        else if (mFactor > 2) {
            this.luminosity = 1.82 * Pow(mFactor, 3.46) * L_Sun;
            //returns 1.5*2^3.5 at 2
        }
        else if (mFactor > 1) {
            this.luminosity = Pow(mFactor, 4.33) * L_Sun;
        }
        else if (mFactor > 0.4) {
            this.luminosity = Pow(mFactor, 3.75) * L_Sun;
        }
        else {
            this.luminosity = Pow(mFactor, 4) * L_Sun;
        }
    }


    private void updateRadius() {
        if (this.isBlackHole) {
            this.radius = 2 * Main.G * this.mass / (Main.c * Main.c);
        }
        // above about 150 solar masses i.e. 300 000 e27 kg, further growth becomes impossible as extreme temperatures expel the outer layers of a star.
        // But we don't model that here :)
        else if (this.mass > 4000e3) { // approx 2 solar masses; larger stars, where additional mass is less effective at creating enough fusion to overcome gravity
            this.radius = 2.230e3 * Pow(this.mass, 0.60);
        }
        else if (this.mass > 159e3) { // 84 jupiter masses i.e. 0.08 solar masses: stars, heat from fusion gradually dominates gravitational effects
            this.radius = 4.5695 * Pow(this.mass, 0.88); // 1.2873e9 at upper end
            // solar radius set to be actual 6.957e8m using this scale
        }
        //jupiter mass ~1.898
        else if (this.mass > 0.778e3) { // 130 earth masses i.e. 0.41 jupiter masses: jupiters and brown dwarfs, gravity slowly begins to shrink the object as mass increases
            this.radius = 1.603e8 * Pow(this.mass, -0.04); // 7.535e7 at upper end
            // jupiter radius 1.31077 with scaling
        }
        else if (this.mass > 0.012e3) { // 2 earth masses: gas/ice dwarfs such as neptune, runaway increase in size due to rapid accumulation of low-density volatiles
            this.radius = 3.118e4 * Pow(mass, 0.59); // 9.322e7 at upper end
        }
        else { // less than 2 earth masses; radius grows approximately with cube root of mass, with only minor corrections due to gravitational compression
            this.radius = 5.733e5 * Pow(mass, 0.28); // 7.953e6 at upper end
        }
        // below about 1e-6 units, objects begin to deviate significantly from round, and mean radius can't be predicted from mass even with known composition
        // but we don't model that here either
    }

    private void updateColor() {
        double red, green, blue, temp;
        red = green = blue = 0;
        temp = this.temperature / 100;
        if (Main.colorscheme == ColorScheme.NATURAL) {
            if (this.isBlackHole) {
                this.color = new Color(0, 0, 0);
            }
            else {
                if (temp < 66) {
                    red = 255;
                }
                else {
                    red = temp - 60;
                    red = 329.6987 * Pow(red, -0.1332);
                    if (red < 0) {
                        red = 0;
                    }
                    if (red > 255) {
                        red = 255;
                    }
                }
                if (temp <= 66) {
                    green = temp;
                    green = 99.47 * Log(green) - 161.1196;
                }
                else {
                    green = temp - 60;
                    green = 288.122 * Pow(green, -0.0755);
                }
                if (green < 0) {
                    green = 0;
                }
                if (green > 255) {
                    green = 255;
                }
                if (temp >= 66) {
                    blue = 255;
                }
                if (temp <= 19 && temp > 10) {
                    blue = 50 - temp * temp;
                }
                if (temp < 10) {
                    blue = 100 - temp * temp;
                }
                if (temp < 66 && temp > 19) {
                    blue = temp - 10;
                    blue = 138.5177 * Log(blue) - 305.0448;
                }
                if (blue < 0) {
                    blue = 0;
                }
                if (blue > 255) {
                    blue = 255;
                }
                if (temp < 10) {
                    this.color = new Color((float)(red * temp * temp * temp / 1e3), (float)(green * temp * temp * temp / 1e3), (float)(blue));
                }
                else {
                    this.color = new Color((float)(red), (float)(green), (float)(blue));
                }
            }
        }
        else if (Main.colorscheme == ColorScheme.VELOCITY) {
            double v = Sqrt(vx * vx + vy * vy);
            red = green = blue = 1000 * v;
            if (red > 255) {
                red = 255;
            }
            if (green > 255) {
                green = 255;
            }
            if (blue > 255) {
                blue = 255;
            }
            this.color = new Color((float)red, (float)green, (float)blue);
        }
        else if (Main.colorscheme == ColorScheme.LIFETIME) {
            red = green = blue = lifeLeft / 4000000;
            if (red > 255) {
                red = 255;
            }
            if (green > 255) {
                green = 255;
            }
            if (blue > 255) {
                blue = 255;
            }
            this.color = new Color((float)red, (float)green, (float)blue);
        }
        else {
            this.color = new Color(0, 255, 0);
        }
    }

    private void updateLife(double stepSize) {
        if (this.isBlackHole) {
            return;
        }
        double massScaled = mass * 1e-11;
        double penalty = massScaled * massScaled * massScaled * stepSize;
        this.lifeLeft -= penalty;
        if (lifeLeft < 0) {
            splinter();
        }
    }

    public void remove() {
        this.removed = true;
        Object.DestroyImmediate(this.gameObject);
        this.gameObject = null;
    }

    private void splinter() {
    }
}
