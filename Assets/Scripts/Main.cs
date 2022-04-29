#define DEBUG_PHYSICS
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using UnityEngine.UI;
using System.Threading.Tasks;
using static System.Math;


// Information about particles to be used by the GPU
[StructLayout(LayoutKind.Explicit, Size=52)]
public struct PhysicsShaderInputType
{
    
    [FieldOffset( 0)]public double x;
    [FieldOffset( 8)]public double y;
    [FieldOffset(16)]public double vx;
    [FieldOffset(24)]public double vy;
    [FieldOffset(32)]public double mass;
    [FieldOffset(40)]public double radius;
    [FieldOffset(48)]public int removed;
}; // size 56 (raw size 52 but will be padded by GPU framework to multiple of 8)

// Information about particle updates calculated by either the GPU or CPU

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

/*
 * Base MonoBehaviour; if attached as run as is, will do calculations using the naive method
 * 
 * Some large data structures only required in specific methods with sizes that change infrequently are
 * instead assigned here to avoid the performance penalty of frequent memory allocations of constant size.
 */
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

    // Number of particles to check in O(N) loops before checking if we need to yield
    public static readonly int numParticleChecksPerYieldCheck = 256;

    // number of threads used by the UpdatePhysics compute shader
    protected static readonly int numGpuThreads = 512;


    // Mass of various objects in 1e27 kg units
    // 243 = proxima centauri
    // 119 = lower bound of stellar fusion
    public static double solarMass = 1988;

    public static double jupiterMass = 1.898;

    public static double neptuneMass = 0.102;

    public static double earthMass = 5.97e-3;

    public static double marsMass = 0.642e-3;

    public static double moonMass = 0.0735e-3;
    
    // initial mass of particles (also 1e27 kg units)
    public static double initialMass = 1.0e-3;

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
    protected static double lastYieldTime = 0;

    private readonly static double targetFps = 30;

    private readonly static double targetSecondsPerFrame = 1.0 / targetFps;

    // Prevent update from trying to call a new physics step if calculations for the previous step are ongoing
    protected volatile bool physicsOngoing = false;

    // All interactions for the current simulation timestep
    protected PhysicsShaderOutputType[] physicsOutput2 = new PhysicsShaderOutputType[initNumParticles];

    /* Stores a maximum of two batches of GPU calculations.
     * Need two arrays here since at any given time we need a non-writable copy of the previous batch and a
     * writable copy of the incoming chunk, so that the incoming writes can be interleaved with outgoing
     * reads and GPU can therefore write at the same time as we read with no synchronization needed.
     */
    protected PhysicsShaderOutputType[][] currentOutputArrays2 = { new PhysicsShaderOutputType[gpuParticleGroupSize],
            new PhysicsShaderOutputType[gpuParticleGroupSize] };

    // Array with i-th element equal to localization for error code i returned from shader.
    // This code may not be used in versions where the shader doesn't return error codes to increase performance
    private static string[] errorMessages = { "No Error", "Particle position i was NaN",
        "Particle position j was NaN", "Distance was NaN",
        "gStepSize invalid", "particle i has invalid mass",
        "particle j has invalid mass", "distance was 0" };

    // multiplier for displayed object sizes
    public static double displayScale = 1.0f;

    // multiplier for simulation timespeed; multiplier for both velocity increments to position and
    // acceleration increments to velocity.
    // note that this is baked into the acceleration calculations and therefore not used directly
    // when updating velocities
    protected static readonly double stepSize = 50.0f;

    // gravitational constant
    public static readonly double G = 1.992e-6f;

    // total static modifier for acceleration strength, used in all interaction calculations
    protected static readonly double gStepSize = G * stepSize;

    // precomputed mass * position, mass * velocity variables
    private double[] mx = new double[initNumParticles];
    private double[] my = new double[initNumParticles];
    private double[] mvx = new double[initNumParticles];
    private double[] mvy = new double[initNumParticles];

    // Dummy particle stored at the end of the particle array to fill it out to a size of 2^n
    // always marked as deleted
    protected Particle placeholderParticle;

    protected int selectedParticle = -1;

    public Camera focusCamera;

    public Text particleInformationText;

    // Start is called before the first frame update
    protected async void Start()
    {
        Application.runInBackground = true;
        Random.InitState(0);
        await initParticles();

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

        await Task.Yield();


    }

    protected string buildParticleInformation(Particle particle)
    {
        List<string> rows = new List<string>();
        rows.Add(string.Format("Name: {0}", particle.name));
        rows.Add(string.Format("Class: {0}", particle.starClass));
        rows.Add(string.Format("Mass: {0:E3} kg", particle.mass * 1e27));
        rows.Add(string.Format("Temperature: {0:F3} K", particle.temperature));
        rows.Add(string.Format("Radius: {0:F3} units", particle.radius));
        rows.Add(string.Format("ID: {0}", particle.id));

        return string.Join("\n", rows);
    }

    protected void deselectCurrentParticle()
    {
        particles[selectedParticle].gameObject.layer = 0;
        focusCamera.enabled = false;
        particleInformationText.text = "";
    }

    protected void selectNewParticle(int particleIdx)
    {
        Debug.Log(string.Format("Selecting particle {0}", particleIdx));
        if (selectedParticle != -1 && particleIdx == selectedParticle)
        {
            deselectCurrentParticle();
        }
        else if (selectedParticle != particleIdx)
        {
            selectedParticle = particleIdx;
            particles[selectedParticle].gameObject.layer = 3;
            focusCamera.transform.parent = particles[selectedParticle].gameObject.transform;
            focusCamera.transform.localPosition = new Vector3(0, 0, -10);
            focusCamera.orthographicSize = 2 * (float)particles[selectedParticle].radius; 
            particleInformationText.text = buildParticleInformation(particles[selectedParticle]);
            focusCamera.enabled = true;
        }
    }

    protected void handleInput()
    {
        float xMovement = Input.GetAxis("Horizontal");
        float yMovement = Input.GetAxis("Vertical");
        float xMovementMouse = Input.GetAxis("Mouse X");
        float yMovementMouse = Input.GetAxis("Mouse Y");
        float zoomChange = Input.GetAxis("Mouse ScrollWheel");
        bool isMouseHeldDown = Input.GetMouseButton(0);
        bool leftClicked = Input.GetMouseButtonDown(0);

        Camera camera = GetComponent<Camera>();

        Vector3 movement = isMouseHeldDown ? new Vector3(xMovementMouse, yMovementMouse, 0) :
            new Vector3(xMovement, yMovement);

        float cameraScale = camera.orthographicSize;

        movement *= camera.orthographicSize;

        gameObject.transform.position -= movement / (float)targetFps;

        zoomChange *= -5;

        if (leftClicked && xMovement == 0 && yMovement == 0) // try to select a particle
        {
            Particle p1; float r;
            float bestDistance = float.PositiveInfinity;
            int toSelect = -1;
            for (int i = 0; i < particles.Length; i++)
            {
                p1 = particles[i];
                if (p1.removed)
                {
                    continue;
                }
                Vector3 particleScreenPos = camera.WorldToScreenPoint(new Vector3((float)p1.x, (float)p1.y, 0));
                particleScreenPos.z = 0;
                r = (particleScreenPos - Input.mousePosition).magnitude;
                if (r < 20 && r < bestDistance)
                {
                    toSelect = i;
                    bestDistance = r;
                }
            }
            if (toSelect != -1)
            {
                selectNewParticle(toSelect);
            }
        }

        if (zoomChange != 0)
        {
            camera.orthographicSize = Mathf.Clamp((float)(camera.orthographicSize * Pow(1.2, zoomChange)), 5, 500);
        }

        

    }

    // Update is called once per frame
    protected async void Update()
    {
        lastYieldTime = Time.realtimeSinceStartupAsDouble;
        handleInput();
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

    protected async Task initParticles()
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

    protected async Task initializeShader(ComputeBuffer shaderInputBuffer, ComputeBuffer shaderOutputBuffer)
    {
        Particle p1;
        PhysicsShaderInputType currentInput;

        NativeArray<PhysicsShaderInputType> gpuArrayInput = shaderInputBuffer.BeginWrite<PhysicsShaderInputType>(0, particles.Length);

        for (int i = 0; i < particles.Length; i++)
        {
            if (i % numParticleChecksPerYieldCheck == 0)
            {
                await yieldCpuIfFrameTooLong();
            }
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

       await initializeShader(shaderInputBuffer, shaderOutputBuffer);

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

    protected virtual void updatePhysicsOutput(Particle p1, ref PhysicsShaderOutputType cpuOutputPixel, bool printDebug=false)
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

    public static async Task yieldCpuIfFrameTooLong()
    {
        double currentTime = Time.realtimeSinceStartupAsDouble;
        if (currentTime > lastYieldTime + targetSecondsPerFrame)
        { // if more than 50 ms have passed since last yield
            await Task.Yield();
            lastYieldTime = Time.realtimeSinceStartupAsDouble;
        }
    }

    protected async Task updateParticles()
    {
        int i, j, k, n = particles.Length;
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
        // Debug.Log(string.Format("{0} particles were calculated, {1} are alive", numWritten, numAlive));
        // Debug.Log(string.Format("Center of mass moving with velocity {0}", centerOfMassSpeed));

        #endif

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
            mx[i] = particles[i].mass * particles[i].x;
            my[i] = particles[i].mass * particles[i].y;
            mvx[i] = particles[i].mass * particles[i].vx;
            mvy[i] = particles[i].mass * particles[i].vy;
        }
        for (i = 0; i < n; i++) {
            if (i % numParticleChecksPerYieldCheck == 0)
            {
                await yieldCpuIfFrameTooLong();
            }
            p1 = particles[i];
            outputPixel = physicsOutput2[i];
            if (particles[i].removed) {
                continue;
            }
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

        if (numAlive <= newParticleArraySize && newParticleArraySize >= minParticleBufferSize)
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
