using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static System.Math;

public class DoubleVector2
{
    public double x;
    public double y;

    public DoubleVector2(double x, double y)
    {
        this.x = x;
        this.y = y;
    }

    public DoubleVector2(Vector2 v)
    {
        this.x = v.x;
        this.y = v.y;
    }

    public static implicit operator Vector2(DoubleVector2 v)
    {
        return new Vector2((float)v.x, (float)v.y);
    }

    public static implicit operator Vector3(DoubleVector2 v)
    {
        return new Vector3((float)v.x, (float)v.y, 0);
    }

    public static DoubleVector2 operator *(double scalar, DoubleVector2 v) {
        return new DoubleVector2(scalar * v.x, scalar * v.y);
    }

    public static DoubleVector2 operator /(DoubleVector2 v, double scalar)
    {
        return new DoubleVector2(v.x / scalar, v.y / scalar);
    }

    public static DoubleVector2 operator +(DoubleVector2 v1, DoubleVector2 v2)
    {
        return new DoubleVector2(v1.x + v2.x, v1.y + v2.y);
    }

    public static DoubleVector2 operator -(DoubleVector2 v1, DoubleVector2 v2)
    {
        return new DoubleVector2(v1.x - v2.x, v1.y - v2.y);
    }

    public double Length()
    {
        return Sqrt(x * x + y * y);
    }
}
