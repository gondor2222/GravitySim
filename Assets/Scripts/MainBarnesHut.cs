using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Collections;
using UnityEngine.Rendering;
using static System.Math;

public struct PhysicsShaderQuadTreeNode
{
    public int particleId;
    public int child; // first child
    public int nextNode; // see QuadTreeNode
    public double totalMass; // total mass contained within the node;
    public double centerOfMassX; // center of mass of the node along the x axis
    public double centerOfMassY; // center of mass of the node along the y axis
    public double width;
}; // size 48 (actually 44 but will be padded to multiple of 8)

public class MainBarnesHut : Main
{
    private QuadTreeNode[] nodeList;
    // If the width of a node region divided by the distance between a particle and the node's
    // center of mass is smaller than this, that node will be ignored.
    private static double barnesHutThreshold = 0.5;
    // Start is called before the first frame update
    protected override async Task Start()
    {
        await base.Start();
    }

    protected override void updatePhysicsOutput(Particle p1, ref PhysicsShaderOutputType outputPixel, bool printDebug=false)
    {
        double dx, dy, r, a, distanceToNode;
        int collisionIndex = 0, nodeToCheck = 0;
        QuadTreeNode node;
        while (nodeToCheck != -1)
        {
            node = nodeList[nodeToCheck];
            if (node.children[0] == -1) // leaf node
            {
                if (node.particleId != p1.id) // ignore self-interaction
                {
                    if (printDebug)
                    {
                        Debug.Log(string.Format("Particle {0} checking leaf particle {1}", p1.id, node.particleId));
                    }
                    dx = node.centerOfMassX - p1.x;
                    dy = node.centerOfMassY - p1.y;
                    r = Sqrt(dx * dx + dy * dy);
                    a = gStepSize * node.totalMass / (r * r);
                    outputPixel.ax += a * dx / r;
                    outputPixel.ay += a * dy / r;
                    if (collisionIndex < 4 && p1.radius + node.width > r)
                    {
                        switch (collisionIndex)
                        {
                            case 0:
                                outputPixel.collisions0 = node.particleId;
                                break;
                            case 1:
                                outputPixel.collisions1 = node.particleId;
                                break;
                            case 2:
                                outputPixel.collisions2 = node.particleId;
                                break;
                            case 3:
                                outputPixel.collisions3 = node.particleId;
                                break;
                        }
                        collisionIndex++;
                    }
                }
                nodeToCheck = node.nextNode;
            }
            else // internal node
            {
                distanceToNode = Sqrt((p1.x - node.centerOfMassX) * (p1.x - node.centerOfMassX) + (p1.y - node.centerOfMassY) * (p1.y - node.centerOfMassY));
                if (node.width / distanceToNode < barnesHutThreshold) // treat as leaf, i.e. skip over children
                {
                    dx = node.centerOfMassX - p1.x;
                    dy = node.centerOfMassY - p1.y;
                    r = Sqrt(dx * dx + dy * dy);
                    a = gStepSize * node.totalMass / (r * r);
                    outputPixel.ax += a * dx / r;
                    outputPixel.ay += a * dy / r;
                    nodeToCheck = node.nextNode;
                    
                }
                else
                {
                    if (printDebug)
                    {
                        Debug.Log(string.Format("Particle {0} descending into node {1}'s children", p1.id, nodeToCheck));
                    }
                    nodeToCheck = node.children[0];
                }
            }
        }
    }

    private async Task initializeShaderBarnesHut(ComputeBuffer nodeListBuffer)
    {
        PhysicsShaderQuadTreeNode node;

        NativeArray<PhysicsShaderQuadTreeNode> nodeListInput = nodeListBuffer.BeginWrite<PhysicsShaderQuadTreeNode>(0, nodeList.Length);
        for (int i = 0; i < nodeList.Length; i++)
        {
            if (i % Main.numParticleChecksPerYieldCheck == 0)
            {
                await Main.yieldCpuIfFrameTooLong();
            }
            await Main.yieldCpuIfFrameTooLong();
            node.particleId = nodeList[i].particleId;
            node.child = nodeList[i].children[0];
            node.nextNode = nodeList[i].nextNode;
            node.totalMass = nodeList[i].totalMass;
            node.centerOfMassX = nodeList[i].centerOfMassX;
            node.centerOfMassY = nodeList[i].centerOfMassY;
            node.width = nodeList[i].width;
            nodeListInput[i] = node;
        }

        nodeListBuffer.EndWrite<PhysicsShaderQuadTreeNode>(nodeList.Length);
        computeShader.SetBuffer(computeShaderKernelIndex, "inputDataQuadTreeNodes", nodeListBuffer);
        computeShader.SetFloat("barnesHutThreshold", (float)barnesHutThreshold);
    }

    override protected async Task initVariables() {
        await base.initVariables();
        Debug.Log("Initializing node list");
        nodeList = new QuadTreeNode[(int)(2 * particles.Length)];
    }

    override protected async Task gravitate()
    {
        await QuadTreeNode.initializeNodeArray(nodeList, particles);

        ComputeBuffer nodeListBuffer = new ComputeBuffer(nodeList.Length, 48, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
        await initializeShaderBarnesHut(nodeListBuffer);
        await base.gravitate();
        nodeListBuffer.Release();
    }

    private List<int> findParticleInNodes(int id, QuadTreeNode[] nodes, int nodeIdx)
    {
        List<int> pathToNode = new List<int>();
        if (nodes[nodeIdx].particleId == id)
        {
            pathToNode.Add(nodeIdx);
            return pathToNode;
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                int childIdx = nodes[nodeIdx].children[i];
                if (childIdx != -1) {
                    List<int> childPath = findParticleInNodes(id, nodes, childIdx);
                    if (childPath.Count > 0)
                    {
                        pathToNode.Add(nodeIdx);
                        pathToNode.AddRange(childPath);
                        return pathToNode;
                    }
                }
            }
        }
        return pathToNode;
    }

    override protected void reduceParticleCount(int newParticleArraySize)
    {
        base.reduceParticleCount(newParticleArraySize);
        nodeList = new QuadTreeNode[(int)(2 * particles.Length)];
    }
}
