#define DEBUG_PHYSICS
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using System.Threading.Tasks;
using static System.Math;

public struct PhysicsShaderInputType
{
    public int removed;
    public double x;
    public double y;
    public double vx;
    public double vy;
    public double mass;
    public double radius;
}; // size 56 (raw size 52 but will be padded by GPU framework to multiple of 8)

public struct PhysicsShaderOutputType
{
    public int idx;
    public int threadId;
    public double ax;
    public double ay;
    public int collisions0;
    public int collisions1;
    public int collisions2;
    public int collisions3;
}; // size 40


public class Main : MonoBehaviour
{

    // particle array
    protected Particle[] particles;
    // initial number of particles and initial capacity of particle array
    public static int initNumParticles = 1024 * 256;
    // smallest length of particle array, i.e. guaranteed minimum capacity for particle array at all times
    public static int minParticleBufferSize = 512;

    // Number of particles the GPU will attempt to calculate interactions for before synchronizing with the other
    protected static int gpuParticleGroupSize = 512;

    // Number of particles the CPU will attempt to calculate interactions for before synchronizing with the other
    protected static int cpuParticleGroupSize = 64;

    protected static int numParticleChecksPerYieldCheck = 256;

    // number of threads used by the UpdatePhysics compute shader
    protected static readonly int numGpuThreads = 512;

    // Initial mass of particles in 1e27 kg units
    // 1988 = sun
    // 243 = proxima centauri
    // 119 = lower bound of stellar fusion
    // 1.898 = jupiter
    // 0.102 = neptune
    // 0.00597 = earth
    // 0.000642 = mars
    // 0.0000735 = moon
    public static double initialMass = 0.002f;

    // How to color the particles; see ColorScheme for options
    public static ColorScheme colorscheme = ColorScheme.NATURAL;

    // Shader used to render the particles
    public Shader starShader;

    // Materials for various types of particle
    public Material brownDwarfMaterial;
    public Material redDwarfMaterial;
    public Material yellowDwarfMaterial;
    public Material blueDwarfMaterial;
    public Material blankMaterial;

    // Compute shader used to calculate interactions on GPU
    public ComputeShader computeShader;

    public string computeShaderKernelName;

    // Kernel of the compute shader's physics calculation kernel
    protected int computeShaderKernelIndex;

    // Command buffer used to dispatch compute requests to the GPU
    protected CommandBuffer commandBuffer;

    // Number of simulation timesteps completed
    protected int numPhysicsFrames = 0;

    // store timestamp for last time the main thread yielded, for use by the fps handler
    protected double lastYieldTime = 0;

    // Prevent update from trying to call a new physics step if calculations for the previous step are ongoing
    protected volatile bool physicsOngoing = false;

    // All interactions for the current simulation timestep
    protected PhysicsShaderOutputType[] physicsOutput2 = new PhysicsShaderOutputType[initNumParticles];

    /* Stores a maximum of two batches of GPU calculations.
     * Need two arrays here since at any given time we need a non-writable copy of the previous batch and a
     * writable copy of the  incoming chunk, so that the incoming writes can be interleaved with outgoing
     * reads for thread flexibility
     */
    protected PhysicsShaderOutputType[][] currentOutputArrays2 = { new PhysicsShaderOutputType[gpuParticleGroupSize],
            new PhysicsShaderOutputType[gpuParticleGroupSize] };

    // Array with i-th element equal to localization for error code i returned from shader.
    // This code may not be used in versions where the shader doesn't return error codes to increase performance
    private static string[] errorMessages = { "No Error", "Particle position i was NaN",
        "Particle position j was NaN", "Distance was NaN",
        "gStepSize invalid", "particle i has invalid mass",
        "particle j has invalid mass", "distance was 0" };

    /// multiplier for displayed object sizes
    public static double displayScale = 1.0f;

    // multiplier for simulation timespeed; multiplier for both velocity increments to position and
    // acceleration increments to velocity.
    protected static double stepSize = 50.0f;

    

    // gravitational constant
    protected static double G = 1.992e-6f;

    protected static double gStepSize = G * stepSize;

    // precomputed mass * position, mass * velocity variables
    private double[] mx = new double[initNumParticles];
    private double[] my = new double[initNumParticles];
    private double[] mvx = new double[initNumParticles];
    private double[] mvy = new double[initNumParticles];

    protected Particle placeholderParticle;

    // Start is called before the first frame update
    protected async void Start()
    {
        Application.runInBackground = true;
        Random.InitState(0);
        initParticles();

        computeShaderKernelIndex = computeShader.FindKernel(computeShaderKernelName);
        #if DEBUG_GPU
        Debug.Log(string.Format("Kernel index is {0}", computeShaderKernelIndex));
        #endif
        commandBuffer = new CommandBuffer();
        commandBuffer.name = "Physics Compute Renderer Buffer";
        

        if (gpuParticleGroupSize % numGpuThreads != 0)
        {
            Debug.LogError(string.Format("Particle group size {0} must be a multiple of number of GPU threads {1}", gpuParticleGroupSize, numGpuThreads));
            UnityEditor.EditorApplication.ExitPlaymode();
            await Task.Yield();
        }


    }

    // Update is called once per frame
    protected async void Update()
    {
        lastYieldTime = Time.realtimeSinceStartupAsDouble;
        if (physicsOngoing)
        {
            return;
        }
        physicsOngoing = true;
        for (int i = 0; i < particles.Length; i++)
        {
            if (i % numParticleChecksPerYieldCheck == 0)
            {
                await yieldCpuIfFrameTooLong();
            }
            if (!particles[i].removed)
            {
                particles[i].updateNonPhysics(stepSize);
            }
        }
        await yieldCpuIfFrameTooLong();
        await gravitate();
        await updateParticles();
        numPhysicsFrames++;
        physicsOngoing = false;
    }

    protected void initParticles()
    {
        placeholderParticle = new Particle("placeholder", -1, 0, 0, 0, 0, 0, this);
        placeholderParticle.remove();
        particles = new Particle[initNumParticles];
        particles[0] = new Particle("Star", 0, 0, 0, 0, 0, 4000, this);
        int numInFirstRing = 1024;
        int maxThisRing = numInFirstRing;
        int counter = 0;
        int numRings = 0;

        for (int i = 1; i < initNumParticles; i++)
        {
            if (counter == maxThisRing)
            {
                counter = 0;
                numRings++;
                maxThisRing = (int)(numInFirstRing * Pow(1.004f, numRings));
            }
            int numLeft = initNumParticles - i;
            if (counter + numLeft < maxThisRing)
            {
                maxThisRing = numLeft;
                #if DEBUG_PHYSICS
                Debug.Log(string.Format("Final ring contains {0} particles", numLeft));
                #endif
            }
            double theta = (counter % maxThisRing) * 2 * PI / maxThisRing;


            // pos = 5 * Random.insideUnitCircle;
            double x = (350 + numRings * 2.0) * Sin(theta);
            double y = (350 + numRings * 2.0) * Cos(theta);
            double rsqr = Sqrt(Sqrt(x * x + y * y));

            // float xrot = (float)(-pos.y / 400 / rsqr / Exp(rsqr / 5));
            // float yrot = (float)(pos.x / 400 / rsqr / Exp(rsqr / 5));
            double xrot = -y;
            double yrot = x;

            Vector2 rand = 0.002f * Random.insideUnitCircle;

            double vx = xrot / 11.3 / rsqr / rsqr / rsqr + rand.x;
            double vy = yrot / 11.3 / rsqr / rsqr / rsqr + rand.y;

            particles[i] = new Particle(string.Format("Particle {0}", i), i, x, y, vx, vy, initialMass, this);
            counter++;
        }
    }

    protected void initializeShader(ComputeBuffer shaderInputBuffer, ComputeBuffer shaderOutputBuffer)
    {
        Particle p1;
        PhysicsShaderInputType currentInput;


        NativeArray<PhysicsShaderInputType> gpuArrayInput = shaderInputBuffer.BeginWrite<PhysicsShaderInputType>(0, particles.Length);

        for (int i = 0; i < particles.Length; i++)
        {
            p1 = particles[i];
            currentInput.x = p1.x;
            currentInput.y = p1.y;
            currentInput.vx = p1.vx;
            currentInput.vy = p1.vy;
            currentInput.removed = p1.removed ? 1 : 0;
            currentInput.mass = p1.mass;
            currentInput.radius = p1.radius;
            gpuArrayInput[i] = currentInput;
        }

        shaderInputBuffer.EndWrite<PhysicsShaderInputType>(particles.Length);
        computeShader.SetFloat("g", (float)G);
        computeShader.SetFloat("stepSize", (float)stepSize);
        computeShader.SetInt("numParticles", particles.Length);
        computeShader.SetBuffer(computeShaderKernelIndex, "inputData", shaderInputBuffer);
        computeShader.SetBuffer(computeShaderKernelIndex, "outputData", shaderOutputBuffer);
    }

    protected virtual async Task gravitate()
    {
        

        int particlesPerThread = gpuParticleGroupSize / numGpuThreads;
        int numGroups = particles.Length / gpuParticleGroupSize;
        bool dataReady = false;
        PhysicsShaderOutputType cpuOutputPixel;

        ComputeBuffer shaderInputBuffer = new ComputeBuffer(particles.Length, 56, ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
        ComputeBuffer shaderOutputBuffer = new ComputeBuffer(gpuParticleGroupSize, 40);

        initializeShader(shaderInputBuffer, shaderOutputBuffer);

        // index of the writable array in the above
        int freeCurrentOutputArray = 0;

        int i, u, n = particles.Length, startIndex, cpuCalculateIndex = particles.Length;

        computeShader.SetInt("particlesPerThread", particlesPerThread);

        int currentGroup = 0;

        #if DEBUG_GPU
        Debug.Log(string.Format("Dispatching particles {0}-{1} to gpu", currentGroup * gpuParticleGroupSize, (currentGroup + 1) * gpuParticleGroupSize - 1));
        #endif
        dispatchCompute(currentGroup);
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(shaderOutputBuffer, (AsyncGPUReadbackRequest requestCallback) => {
            // has to be copied to cpu-side memory because Unity throws away the cache after one frame (which happens if we yield)
            requestCallback.GetData<PhysicsShaderOutputType>().CopyTo(currentOutputArrays2[freeCurrentOutputArray]);
            dataReady = true;
        });

        // Exit when maximum index already calculated by GPU is >= minimum index already calculated by cpu
        while (currentGroup * gpuParticleGroupSize < cpuCalculateIndex)
        {
            // GPU has results for us
            if (dataReady)
            {
                #if DEBUG_GPU
                Debug.Log(string.Format("GPU ready with particles {0} to {1}", currentGroup * gpuParticleGroupSize, (currentGroup + 1) * gpuParticleGroupSize - 1));
                #endif
                dataReady = false;

                // redirect writes to the current readable outputArray, while the previous writable array becomes the read array
                freeCurrentOutputArray = 1 - freeCurrentOutputArray;
                // If next group is valid, i.e. gpu has not already calculated everything
                if (currentGroup < numGroups - 1)
                {
                    // Only dispatch next batch to GPU if the CPU has not calculated all the way down to the min index for this group
                    if ((currentGroup + 1) * gpuParticleGroupSize < cpuCalculateIndex)
                    {
                        #if DEBUG_GPU
                        Debug.Log(string.Format("Dispatching particles {0}-{1} to gpu", (currentGroup + 1) * gpuParticleGroupSize, (currentGroup + 2) * gpuParticleGroupSize - 1));
                        #endif
                        dispatchCompute(currentGroup + 1);
                        request = AsyncGPUReadback.Request(shaderOutputBuffer, (AsyncGPUReadbackRequest requestCallback) =>
                        {
                            // has to be copied to cpu-side memory because Unity throws away the cache after one frame (which happens if we yield)
                            requestCallback.GetData<PhysicsShaderOutputType>().CopyTo(currentOutputArrays2[freeCurrentOutputArray]);
                            dataReady = true;
                        });
                    }
                    #if DEBUG_GPU
                    else
                    {
                        Debug.Log(string.Format("GPU Standing by with particles {0}-{1} because CPU is already at index {2}", (currentGroup + 1) * gpuParticleGroupSize, (currentGroup + 2) * gpuParticleGroupSize - 1, cpuCalculateIndex));
                    }
                    #endif
                }
                startIndex = currentGroup * gpuParticleGroupSize;
                // copy the copy of the current GPU batch to cpu-side memory
                for (i = 0; i < gpuParticleGroupSize; i++)
                {
                    if (i % numParticleChecksPerYieldCheck == 0)
                    {
                        await yieldCpuIfFrameTooLong();
                    }
                    physicsOutput2[i + startIndex] = currentOutputArrays2[1 - freeCurrentOutputArray][i];
                        #if DEBUG_GPU
                        if (physicsOutput2[i + startIndex].idx != i + startIndex) {
                            Debug.Log(string.Format("GPU reported particle {0} as having id {1} (thread {2}). ax = {3}",
                            i + startIndex,
                            currentOutputArrays2[1 - freeCurrentOutputArray][i].idx,
                            currentOutputArrays2[1 - freeCurrentOutputArray][i].threadId,
                            currentOutputArrays2[1 - freeCurrentOutputArray][i].ax));
                            UnityEditor.EditorApplication.ExitPlaymode();
                            await Task.Yield();
                        }
                        #endif
                }
                currentGroup++;
            }
            // do more work on the CPU while we wait for GPU
            else
            {
                double halfStepSize = 0.5 * stepSize;
                #if DEBUG_CPU
                Debug.Log(string.Format("CPU attempting to calculate particles {0}-{1}", cpuCalculateIndex - cpuParticleGroupSize, cpuCalculateIndex - 1));
                #endif
                for (u = 0; u < cpuParticleGroupSize; u++)
                {
                    if (u % numParticleChecksPerYieldCheck == 0)
                    {
                        await yieldCpuIfFrameTooLong();
                    }
                    if (cpuCalculateIndex < (currentGroup) * gpuParticleGroupSize)
                    {
                        #if DEBUG_CPU
                        Debug.Log(string.Format("CPU standing by because it is on particle {0} and current GPU group runs from {1} to {2}", cpuCalculateIndex, currentGroup * gpuParticleGroupSize, (currentGroup + 1) * gpuParticleGroupSize - 1));
                        #endif
                        break;
                    }
                    cpuCalculateIndex--;
                    i = cpuCalculateIndex;

                    cpuOutputPixel.idx = i;
                    cpuOutputPixel.ax = 0;
                    cpuOutputPixel.ay = 0;
                    cpuOutputPixel.threadId = -1;
                    cpuOutputPixel.collisions0 = -1;
                    cpuOutputPixel.collisions1 = -1;
                    cpuOutputPixel.collisions2 = -1;
                    cpuOutputPixel.collisions3 = -1;

                    updatePhysicsOutput(particles[i], ref cpuOutputPixel);
                    
                    physicsOutput2[i] = cpuOutputPixel;
                }
            }
            await yieldCpuIfFrameTooLong();
        }
        if (!request.done)
        {
            request.WaitForCompletion();
        }
        shaderInputBuffer.Release();
        shaderOutputBuffer.Release();
    }

    protected virtual void updatePhysicsOutput(Particle p1, ref PhysicsShaderOutputType cpuOutputPixel)
    {
        double halfStepSize = 0.5 * stepSize;
        if (!p1.removed)
        {
            int j, v = 0;
            double dx, dy, massRatio, r, a;
            Particle p2;
            for (j = 0; j < particles.Length; j++)
            {

                p2 = particles[j];
                if (p2.removed || p1.id == p2.id)
                {
                    continue;
                }
                massRatio = p1.mass / p2.mass;
                dx = p2.x + halfStepSize * p2.vx - p1.x - halfStepSize * p1.vx;
                dy = p2.y + halfStepSize * p2.vy - p1.y - halfStepSize * p1.vy;
                r = Sqrt(dx * dx + dy * dy);
                a = gStepSize * p2.mass / (r * r);
                cpuOutputPixel.ax += a * dx / r;
                cpuOutputPixel.ay += a * dy / r;

                if (p1.radius + p2.radius > r && v < 4)
                {
                    switch (v)
                    {
                        case 0:
                            cpuOutputPixel.collisions0 = j;
                            break;
                        case 1:
                            cpuOutputPixel.collisions0 = j;
                            break;
                        case 2:
                            cpuOutputPixel.collisions0 = j;
                            break;
                        case 3:
                            cpuOutputPixel.collisions0 = j;
                            break;
                    }
                    v++;
                }
            }
        }
    }

    protected void dispatchCompute(int groupNumber)
    {
        int startIndex = groupNumber * gpuParticleGroupSize;
        #if DEBUG_GPU
        Debug.Log(string.Format("This dispatch starts at particle {0}", startIndex));
        #endif
        computeShader.SetInt("startIndex", startIndex);
        computeShader.Dispatch(computeShaderKernelIndex, 1, 1, 1);
    }

    protected async Task yieldCpuIfFrameTooLong()
    {
        double currentTime = Time.realtimeSinceStartupAsDouble;
        if (currentTime * 1000 > lastYieldTime * 1000 + 50)
        { // if more than 50 ms have passed since last yield
            await Task.Yield();
            lastYieldTime = Time.realtimeSinceStartupAsDouble;
        }
    }

    protected async Task updateParticles()
    {
        int i, j, k, n = particles.Length, numLivingParticles = 0;
        Particle p1, p2;
        PhysicsShaderOutputType outputPixel;
        double newm;
        
        bool canPrintError = true;

        #if DEBUG_PHYSICS

        int numWritten = 0;
        int numAlive = 0;
        double centerOfMassVX = 0;
        double centerOfMassVY = 0;
        double totalMass = 0;
        for (i = 0; i < n; i++)
        {
            if (i % numParticleChecksPerYieldCheck == 0)
            {
                await yieldCpuIfFrameTooLong();
            }
            if (physicsOutput2[i].idx == i)
            {
                numWritten++;
                if (!particles[i].removed)
                {
                    centerOfMassVX += particles[i].mass * particles[i].vx;
                    centerOfMassVY += particles[i].mass * particles[i].vy;
                    totalMass += particles[i].mass;
                    numAlive++;
                }
            }
        }
        centerOfMassVX /= totalMass;
        centerOfMassVY /= totalMass;
        double centerOfMassSpeed = Sqrt(centerOfMassVX * centerOfMassVX + centerOfMassVY * centerOfMassVY);
        Debug.Log(string.Format("{0} particles were calculated, {1} are alive", numWritten, numAlive));
        Debug.Log(string.Format("Center of mass moving with velocity {0}", centerOfMassSpeed));

        #endif

        for (i = 0; i < n; i++)
        {
            if (i % numParticleChecksPerYieldCheck == 0)
            {
                await yieldCpuIfFrameTooLong();
            }
            if (!particles[i].removed)
            {
                numLivingParticles++;
                mx[i] = particles[i].mass * particles[i].x;
                my[i] = particles[i].mass * particles[i].y;
                mvx[i] = particles[i].mass * particles[i].vx;
                mvy[i] = particles[i].mass * particles[i].vy;
            }
        }

        for (i = 0; i < n; i++)
        {
            if (i % numParticleChecksPerYieldCheck == 0)
            {
                await yieldCpuIfFrameTooLong();
            }

            if (particles[i].removed)
            {
                continue;
            }
            p1 = particles[i];
            outputPixel = physicsOutput2[i];
            if (outputPixel.idx != i)
            {
                if (canPrintError)
                {
                    Debug.LogError(string.Format("Particle {0} was reported to have id {1} by shader", i, outputPixel.idx));
                    canPrintError = false;
                }
                UnityEditor.EditorApplication.ExitPlaymode();
                await Task.Yield();
            }
            if (outputPixel.ax != outputPixel.ax || outputPixel.ax == 0)
            {
                if (canPrintError)
                {
                    Debug.LogError(string.Format("Particle {0} has x-acceleration {1}", i, outputPixel.ax));
                    canPrintError = false;
                }
                UnityEditor.EditorApplication.ExitPlaymode();
                await Task.Yield();
            }
            p1.vx += outputPixel.ax;
            p1.vy += outputPixel.ay;
            for (k = 0; k < 4; k++)
            {
                int collisionIndex = -1;
                switch (k)
                {
                    case 0:
                        collisionIndex = outputPixel.collisions0;
                        break;
                    case 1:
                        collisionIndex = outputPixel.collisions1;
                        break;
                    case 2:
                        collisionIndex = outputPixel.collisions2;
                        break;
                    case 3:
                        collisionIndex = outputPixel.collisions3;
                        break;
                }
                if (collisionIndex < -1 || collisionIndex >= particles.Length)
                {
                    throw new System.Exception(string.Format("Bad collision index: {0}", collisionIndex));
                }
                if (collisionIndex != -1 && !particles[collisionIndex].removed)
                {
                    j = collisionIndex;
                    p2 = particles[j];
                    newm = p1.mass + p2.mass;

                    p1.mass = newm;
                    p1.massDirty = true;
                    p1.x = (mx[i] + mx[j]) / newm;
                    p1.y = (my[i] + my[j]) / newm;

                    p1.vx = (mvx[i] + mvx[j]) / newm;
                    p1.vy = (mvy[i] + mvy[j]) / newm;

                    mx[i] = p1.mass * p1.x;
                    my[i] = p1.mass * p1.y;
                    mvx[i] = p1.mass * p1.vx;
                    mvy[i] = p1.mass * p1.vy;
                    p2.remove();
                    await yieldCpuIfFrameTooLong();
                }
            }
        }
        for (i = 0; i < particles.Length; i++)
        {
            if (i % numParticleChecksPerYieldCheck == 0)
            {
                await yieldCpuIfFrameTooLong();
            }
            if (!particles[i].removed)
            {
                particles[i].updatePhysics(stepSize);
            }
        }
        int newParticleArraySize = particles.Length / 2;

        if (numLivingParticles <= newParticleArraySize && newParticleArraySize >= minParticleBufferSize)
        {
            reduceParticleCount(newParticleArraySize);
        }
    }

    protected virtual void reduceParticleCount(int newParticleArraySize)
    {
        Particle[] newParticles = new Particle[newParticleArraySize];
        int idx = 0;
        for (int i = 0; i < particles.Length; i++)
        {
            if (!particles[i].removed)
            {
                newParticles[idx] = particles[i];
                particles[i].id = idx;
                idx++;
            }
        }
        while (idx < newParticleArraySize)
        {
            newParticles[idx++] = placeholderParticle;
        }

        particles = newParticles;
        physicsOutput2 = new PhysicsShaderOutputType[particles.Length];
        mx = new double[particles.Length];
        my = new double[particles.Length];
        mvx = new double[particles.Length];
        mvy = new double[particles.Length];
    }

    public void addParticle(Particle p)
    {
        
    }
}
