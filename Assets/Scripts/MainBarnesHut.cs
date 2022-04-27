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
    new protected void Start()
    {
        base.Start();
        nodeList = new QuadTreeNode[(int)(2 * particles.Length)];
    }

    
    private int[] nodesToCheck;

    protected override void updatePhysicsOutput(Particle p1, ref PhysicsShaderOutputType outputPixel)
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
                                outputPixel.collisions0 = node.particleId;
                                break;
                            case 2:
                                outputPixel.collisions0 = node.particleId;
                                break;
                            case 3:
                                outputPixel.collisions0 = node.particleId;
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
                    nodeToCheck = node.children[0];
                }
            }
        }
    }

    private void initializeShaderBarnesHut(ComputeBuffer nodeListBuffer)
    {
        PhysicsShaderQuadTreeNode node;

        NativeArray<PhysicsShaderQuadTreeNode> nodeListInput = nodeListBuffer.BeginWrite<PhysicsShaderQuadTreeNode>(0, nodeList.Length);
        for (int i = 0; i < nodeList.Length; i++)
        {
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

    override protected async Task gravitate()
    {
        QuadTreeNode.fillNodeArray(nodeList, particles);

        ComputeBuffer nodeListBuffer = new ComputeBuffer(nodeList.Length, 48, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
        initializeShaderBarnesHut(nodeListBuffer);
        await base.gravitate();

        nodeListBuffer.Release();
    }

    override protected void reduceParticleCount(int newParticleArraySize)
    {
        base.reduceParticleCount(newParticleArraySize);
        nodeList = new QuadTreeNode[(int)(2 * particles.Length)];
    }
}
