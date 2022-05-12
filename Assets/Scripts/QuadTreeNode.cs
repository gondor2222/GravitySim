using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using static System.Math;

public class QuadTreeNode
{
    
    public int particleId = -1;
    // index of next node in depth-first search, ignoring children
    // intended to be used by GPU for memoryless traversal of non-ignored nodes;
    // if this node is a leaf or we are treating it as a leaf the next node is nextNode,
    // otherwise the next node is the first child.
    public int nextNode = -1; 
    public double totalMass = 0; // total mass contained within the node;
    public double centerOfMassX = 0; // center of mass of the node along the x axis
    public double centerOfMassY = 0; // center of mass of the node along the y axis
    public double vx; // velocity of center of mass along x axis
    public double vy; // velocity of center of mass along y axis
    public double width = 0;
    public readonly int[] children = new int[4]; // { indices for upper-left, upper-right, lower-left, lower-right

    public QuadTreeNode() {
        this.children = new int[] { -1, -1, -1, -1 };
    }

    public static QuadTreeNode placeholder = new QuadTreeNode();

    private static async Task<int> fillNodeArray(Particle[] allParticles, int[] particleIdxs, int count, QuadTreeNode[] array, int addIndex, int depth)
    {
        if (addIndex % Main.numParticleChecksPerYieldCheck == 0)
        {
            await Main.yieldCpuIfFrameTooLong();
        }
        if (count == 0)
        {
            throw new System.Exception("Empty node constructed");
        }
        if (array[addIndex] == null)
        {
            array[addIndex] = new QuadTreeNode();
        }
        if (count == 1)
        {  // leaf node
            int particleIdx = particleIdxs[0];
            array[addIndex].particleId = allParticles[particleIdx].id;
            array[addIndex].width = allParticles[particleIdx].radius;
            array[addIndex].totalMass = allParticles[particleIdx].mass;
            array[addIndex].centerOfMassX = allParticles[particleIdx].x;
            array[addIndex].centerOfMassY = allParticles[particleIdx].y;
            array[addIndex].vx = allParticles[particleIdx].vx;
            array[addIndex].vy = allParticles[particleIdx].vy;
            for (int i = 0; i < 4; i++)
            {
                array[addIndex].children[i] = -1;
            }
            return addIndex + 1;
        }
        else
        { // internal node
            int[][] subTreeParticles = {
                new int[count - 1],
                new int[count - 1],
                new int[count - 1],
                new int[count - 1]
            };
            array[addIndex].centerOfMassX = 0;
            array[addIndex].centerOfMassY = 0;
            array[addIndex].totalMass = 0;
            array[addIndex].particleId = -1;
            array[addIndex].vx = 0;
            array[addIndex].vy = 0;
            for (int i = 0; i < 4; i++)
            {
                array[addIndex].children[i] = -1;
            }

            double minX = double.PositiveInfinity;
            double maxX = double.NegativeInfinity;
            double minY = double.PositiveInfinity;
            double maxY = double.NegativeInfinity;
            Particle p1;
            for (int i = 0; i < count; i++)
            { // calculate bounds of this node 
                p1 = allParticles[particleIdxs[i]];
                minX = p1.x < minX ? p1.x : minX;
                maxX = p1.x > maxX ? p1.x : maxX;
                minY = p1.y < minY ? p1.y : minY;
                maxY = p1.y > maxY ? p1.y : maxY;
            }
            double centerX = (maxX + minX) / 2;
            double centerY = (maxY + minY) / 2;
            array[addIndex].width = Sqrt((maxX - minX) * (maxX - minX) + (maxY - minY) * (maxY - minY));
            int idx;
            int[] subcounts = { 0, 0, 0, 0 };
            int sublistIdx;
            for (int i = 0; i < count; i++)
            { // sort particles into sublists by quadrant
                idx = particleIdxs[i];
                p1 = allParticles[idx];
                int xChildIndex = p1.x > centerX ? 1 : 0;
                int yChildIndex = p1.y > centerY ? 1 : 0;
                sublistIdx = xChildIndex * 2 + yChildIndex;
                subTreeParticles[sublistIdx][subcounts[sublistIdx]] = idx;
                subcounts[sublistIdx]++;
                if (i % Main.numParticleChecksPerYieldCheck == 0)
                {
                    await Main.yieldCpuIfFrameTooLong();
                }
            }
            int childIdx = 0;
            int firstFreeIdx = addIndex + 1;
            int lastChildAdded = addIndex + 1;
            for (int i = 0; i < 2; i++)
            { // initialize subtrees
                for (int j = 0; j < 2; j++)
                {
                    if (subcounts[2 * i + j] > 0) {
                        lastChildAdded = await fillNodeArray(allParticles, subTreeParticles[2 * i + j], subcounts[2 * i + j], array, firstFreeIdx, depth + 1);
                        array[addIndex].children[childIdx++] = firstFreeIdx;
                        array[addIndex].totalMass = array[addIndex].totalMass + array[firstFreeIdx].totalMass;
                        array[addIndex].centerOfMassX += array[firstFreeIdx].totalMass * array[firstFreeIdx].centerOfMassX; // weighted sum, will be normalized by total mass
                        array[addIndex].centerOfMassY += array[firstFreeIdx].totalMass * array[firstFreeIdx].centerOfMassY;
                        array[addIndex].vx += array[firstFreeIdx].totalMass * array[firstFreeIdx].vx;
                        array[addIndex].vy += array[firstFreeIdx].totalMass * array[firstFreeIdx].vy;
                        firstFreeIdx = lastChildAdded;
                    }
                }
            }
            array[addIndex].centerOfMassX /= array[addIndex].totalMass; // normalize centerOfMass by total mass
            array[addIndex].centerOfMassY /= array[addIndex].totalMass;
            array[addIndex].vx /= array[addIndex].totalMass;
            array[addIndex].vy /= array[addIndex].totalMass;
            return firstFreeIdx;
        }
    }

    public static void initializeNextNodes(int nodeId, QuadTreeNode[] nodeArray, int nextNode, int depth) {
        nodeArray[nodeId].nextNode = nextNode;
        for (int i = 0; i < 4; i++)
        {
            if (nodeArray[nodeId].children[i] != -1) {
                // all children receive the next child as nextNode, except for the last child which inherits nextNode from its parent
                initializeNextNodes(nodeArray[nodeId].children[i], nodeArray, i < 3 && nodeArray[nodeId].children[i + 1] != -1 ? nodeArray[nodeId].children[i + 1] : nextNode, depth + 1);
            }
        }
    }

    public static async Task initializeNodeArray(QuadTreeNode[] array, Particle[] particles)
    {
        int[] particleIds = new int[particles.Length];
        int idx = 0;
        for (int i = 0; i < particles.Length; i++)
        {
            if (!particles[i].removed)
            {
                particleIds[idx++] = i;
            }
        }

        int addIndex = await fillNodeArray(particles, particleIds, idx, array, 0, 0);
        await Main.yieldCpuIfFrameTooLong();
        while (addIndex < array.Length)
        {
            if (array[addIndex] == null)
            {
                array[addIndex] = new QuadTreeNode();
            }
            else
            {
                array[addIndex].particleId = -1;
                for (int i = 0; i < 4; i++)
                {
                    array[addIndex].children[i] = -1;
                }
            }
            addIndex++;
        }
        await Main.yieldCpuIfFrameTooLong();
        initializeNextNodes(0, array, -1, 0);
    }
}
