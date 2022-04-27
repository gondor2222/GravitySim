using System.Collections;
using System.Collections.Generic;
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
    public double width = 0;
    public readonly int[] children = new int[4]; // { indices for upper-left, upper-right, lower-left, lower-right

    public QuadTreeNode() {
        this.children = new int[] { -1, -1, -1, -1 };
    }

    public static int fillNodeArray(Particle[] allParticles, List<int> particleIdxs, ref QuadTreeNode[] nodes, ref int addIndex)
    {
        int addedIndex = addIndex;
        addIndex++;
        if (particleIdxs.Count == 0)
        {
            throw new System.Exception("Empty node constructed");
        }
        if (nodes[addedIndex] == null)
        {
            nodes[addedIndex] = new QuadTreeNode();
        }
        if (particleIdxs.Count == 1)
        {  // leaf node
            int particleIdx = particleIdxs[0];
            nodes[addedIndex].particleId = allParticles[particleIdx].id;
            nodes[addedIndex].width = allParticles[particleIdx].radius;
            nodes[addedIndex].totalMass = allParticles[particleIdx].mass;
            nodes[addedIndex].centerOfMassX = allParticles[particleIdx].x;
            nodes[addedIndex].centerOfMassY = allParticles[particleIdx].y;
            for (int i = 0; i < 4; i++)
            {
                nodes[addedIndex].children[i] = -1;
            }
        }
        else
        { // internal node
            List<int>[] subTreeParticles = {
                new List<int>(particleIdxs.Count / 2 + 1),
                new List<int>(particleIdxs.Count / 2 + 1),
                new List<int>(particleIdxs.Count / 2 + 1),
                new List<int>(particleIdxs.Count / 2 + 1)
            };
            nodes[addedIndex].centerOfMassX = 0;
            nodes[addedIndex].centerOfMassY = 0;
            nodes[addedIndex].totalMass = 0;
            for (int i = 0; i < 4; i++)
            {
                nodes[addedIndex].children[i] = -1;
            }

            double minX = double.PositiveInfinity;
            double maxX = double.NegativeInfinity;
            double minY = double.PositiveInfinity;
            double maxY = double.NegativeInfinity;
            foreach (int idx in particleIdxs)
            { // calculate bounds of this node 
                minX = allParticles[idx].x < minX ? allParticles[idx].x : minX;
                maxX = allParticles[idx].x > maxX ? allParticles[idx].x : maxX;
                minY = allParticles[idx].y < minY ? allParticles[idx].y : minY;
                maxY = allParticles[idx].y > maxY ? allParticles[idx].y : maxY;
            }
            double centerX = (maxX + minX) / 2;
            double centerY = (maxY + minY) / 2;
            nodes[addedIndex].width = Sqrt((maxX - minX) * (maxX - minX) + (maxY - minY) * (maxY - minY));
            foreach (int idx in particleIdxs)
            { // sort particles into sublists by quadrant
                int xChildIndex = allParticles[idx].x > centerX ? 1 : 0;
                int yChildIndex = allParticles[idx].y > centerY ? 1 : 0;
                subTreeParticles[xChildIndex * 2 + yChildIndex].Add(idx);
            }
            int childIdx = 0;
            for (int i = 0; i < 2; i++)
            { // initialize subtrees
                for (int j = 0; j < 2; j++)
                {
                    if (subTreeParticles[2 * i + j].Count > 0)
                    {
                        int childNodeIdx = fillNodeArray(allParticles, subTreeParticles[2 * i + j], ref nodes, ref addIndex);
                        nodes[addedIndex].children[childIdx++] = childNodeIdx;
                        nodes[addedIndex].totalMass += nodes[childNodeIdx].totalMass;
                        nodes[addedIndex].centerOfMassX += nodes[childNodeIdx].totalMass * nodes[childNodeIdx].centerOfMassX; // weighted sum, will be normalized by total mass
                        nodes[addedIndex].centerOfMassY += nodes[childNodeIdx].totalMass * nodes[childNodeIdx].centerOfMassY;
                    }
                }
            }
            nodes[addedIndex].centerOfMassX /= nodes[addedIndex].totalMass; // normalize centerOfMass by total mass
            nodes[addedIndex].centerOfMassY /= nodes[addedIndex].totalMass;
        }
        return addedIndex;
    }

    public static void initializeNextNodes(int nodeId, ref QuadTreeNode[] nodeArray, int nextNode, int depth) {
        nodeArray[nodeId].nextNode = nextNode;
        for (int i = 0; i < 4; i++)
        {
            if (nodeArray[nodeId].children[i] != -1) {
                // all children receive the next child as nextNode, except for the last child which inherits nextNode from its parent
                initializeNextNodes(nodeArray[nodeId].children[i], ref nodeArray, i < 3 && nodeArray[nodeId].children[i + 1] != -1 ? nodeArray[nodeId].children[i + 1] : nextNode, depth + 1);
            }
        }
    }

    public static void fillNodeArray(QuadTreeNode[] array, Particle[] particles)
    {
        int addIndex = 0;
        List<int> particleIds = new List<int>();
        for (int i = 0; i < particles.Length; i++)
        {
            if (!particles[i].removed)
            {
                particleIds.Add(i);
            }
        }
        fillNodeArray(particles, particleIds, ref array, ref addIndex);
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
        initializeNextNodes(0, ref array, -1, 0);
    }
}
