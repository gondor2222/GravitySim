#define DEBUG_PHYSICS
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using Unity.Collections;
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

public enum InitMode {
    CLOUD,
    DISK,
    SYSTEM
};

/*
 * Base MonoBehaviour; if attached as run as is, will do calculations using the naive method
 * 
 * Some large data structures only required in specific methods with sizes that change infrequently are
 * instead assigned here to avoid the performance penalty of frequent memory allocations of constant size.
 */
public class Main : MonoBehaviour {

    // particle array
    protected volatile Particle[] particles;
    // initial number of particles and initial capacity of particle array
    public static int initNumParticles = 1024 * 512 / 4;
    public static InitMode defaultInitMode = InitMode.SYSTEM;
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

    public static readonly int maxHeatSources = 5;

    // mass in kg of 1 mass unit in the particle class
    public static readonly double massUnits = 1e21;

    // Mass of various objects in massUnits
    // 0.06 M_sun ~ lower bound of stellar fusion
    public static readonly double solarMass = 1.988e9;

    public static readonly double jupiterMass = 1.898e6;

    public static readonly double neptuneMass = 1.02e5;

    public static readonly double earthMass = 5.97e3;

    public static readonly double marsMass = 6.42e2;

    public static readonly double moonMass = 7.35e1;

    // initial mass of particles (also 1e27 kg units)
    public static double initialMass = 0.3e0;

    public static long simulationStartTime;

    // How to color the particles; see ColorScheme for options
    public static ColorScheme colorscheme = ColorScheme.NATURAL;

    public string computeShaderKernelName;

    // Kernel of the compute shader's physics calculation kernel
    protected int computeShaderKernelIndex;

    // Command buffer used to dispatch compute requests to the GPU
    protected CommandBuffer commandBuffer;

    // Number of simulation timesteps completed
    protected long numPhysicsFrames = 0;

    protected double totalSimulationTime = 0;

    // store timestamp for last time the main thread yielded, for use by the fps handler
    protected static double lastYieldTime = 0;

    public double lastMenuChange = 0;

    private static double targetFps = 100;

    private readonly static double targetSecondsPerFrame = 1.0 / targetFps;

    protected bool paused = true;

    // Prevent update from trying to call a new physics step if calculations for the previous step are ongoing
    protected volatile bool physicsOngoing = false;

    protected volatile bool handlingInput = false;

    protected volatile bool isInitializing = false;

    protected bool draggedDuringClick = false;

    // All interactions for the current simulation timestep
    protected PhysicsShaderOutputType[] physicsOutput2;

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

    // multiplier for simulation timespeed; multiplier for both velocity increments to position and
    // acceleration increments to velocity.
    // note that this is baked into the acceleration calculations and therefore not used directly
    // when updating velocities
    public static int currentStepSize = 3;

    protected static readonly double[] stepSizes = {
        1e+1, 3e+1,
        1e+2, 3e+2,
        1e+3, 3e+3,
        1e+4, 3e+4,
        1e+5, 3e+5,
        1e+6, 3e+6
    };

    // gravitational constant in m^3 / s^2 / mass unit
    public static readonly double G = 6.674e-11 * massUnits;

    // total modifier for acceleration strength, used in all interaction calculations
    protected static double gStepSize = G * stepSizes[currentStepSize];

    //speed of light
    public static readonly double c = 2.998e8;

    public static readonly double distanceScale = 1e9f; // 1 unit of gameobject transform space in meters

    // precomputed mass * position, mass * velocity variables
    private double[] mx;
    private double[] my;
    private double[] mvx;
    private double[] mvy;

    // Dummy particle stored at the end of the particle array to fill it out to a size of 2^n
    // always marked as deleted
    protected Particle placeholderParticle;

    protected int selectedParticle = -1;

    public bool isReady = false;

    public bool cancelCalculation = false;

    public static Camera focusCamera;
    public static Camera mainCamera;

    public static Text particleInformationText;
    public static Text systemInformationText;
    public static Text speedText;
    public static Canvas informationCanvas;
    public static Canvas mainMenuCanvas;
    public static Image selectionIcon;
    public static Image pauseButtonImage;
    public static EventSystem eventSystem;
    public static Button initButton;
    public static Button speedUpButton;
    public static Button slowDownButton;

    // Shader used to render the particles
    public static Shader starShader;

    // Materials for various types of particle
    public static Material brownDwarfMaterial;
    public static Material redDwarfMaterial;
    public static Material yellowDwarfMaterial;
    public static Material blueDwarfMaterial;
    public static Material plainThermalMaterial;
    public static Material blankMaterial;

    public static Sprite playSprite;
    public static Sprite pauseSprite;

    // Compute shader used to calculate interactions on GPU
    public static ComputeShader computeShader;

    public static Mesh sphereMesh;

    // Start is called before the first frame update
    protected async virtual Task Start() {
        Application.runInBackground = true;
        Application.targetFrameRate = (int)targetFps;

        mainCamera = GameObject.Find("Main Camera").GetComponent<Camera>();
        focusCamera = GameObject.Find("Focus Camera").GetComponent<Camera>();

        particleInformationText = GameObject.Find("Particle Information").GetComponent<Text>();
        systemInformationText = GameObject.Find("System Information").GetComponent<Text>();
        speedText = GameObject.Find("Speed Text").GetComponent<Text>();
        informationCanvas = GameObject.Find("Information Canvas").GetComponent<Canvas>();
        mainMenuCanvas = GameObject.Find("Main Menu").GetComponent<Canvas>();
        selectionIcon = GameObject.Find("Selection Icon").GetComponent<Image>();
        pauseButtonImage = GameObject.Find("Pause Button Image").GetComponent<Image>();
        eventSystem = GameObject.Find("EventSystem").GetComponent<EventSystem>();
        initButton = GameObject.Find("Initialize Button").GetComponent<Button>();
        speedUpButton = GameObject.Find("Speed Up Button").GetComponent<Button>();
        slowDownButton = GameObject.Find("Slow Down Button").GetComponent<Button>();

        starShader = Resources.Load<Shader>("Shaders/StarSurfaceShader");
        brownDwarfMaterial = Resources.Load<Material>("Materials/BrownDwarfMaterial");
        redDwarfMaterial = Resources.Load<Material>("Materials/RedDwarfMaterial");
        yellowDwarfMaterial = Resources.Load<Material>("Materials/YellowDwarfMaterial");
        blueDwarfMaterial = Resources.Load<Material>("Materials/BlueDwarfMaterial");
        blankMaterial = Resources.Load<Material>("Materials/BlankMaterial");
        plainThermalMaterial = Resources.Load<Material>("Materials/PlainThermalMaterial");
        computeShader = Resources.Load<ComputeShader>("Shaders/UpdatePhysics");

        playSprite = Resources.Load<Sprite>("Sprites/play");
        pauseSprite = Resources.Load<Sprite>("Sprites/pause");

        UnityEngine.Random.InitState(0);
        placeholderParticle = new Particle("placeholder");
        
        initializeComponents();

        computeShaderKernelIndex = computeShader.FindKernel(computeShaderKernelName);
        #if DEBUG_GPU
        Debug.Log(string.Format("Kernel index is {0}", computeShaderKernelIndex));
        #endif
        commandBuffer = new CommandBuffer();
        commandBuffer.name = "Physics Compute Renderer Buffer";

#if UNITY_EDITOR
        if (gpuParticleGroupSize % numGpuThreads != 0) {
            Debug.LogError(string.Format("Particle group size {0} must be a multiple of number of GPU threads {1}", gpuParticleGroupSize, numGpuThreads));
            UnityEditor.EditorApplication.ExitPlaymode();
            await Task.Yield();
        }
#endif

        GameObject sphereObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphereMesh = sphereObject.GetComponent<MeshFilter>().mesh;

        GameObject.DestroyImmediate(sphereObject);

        await Task.Yield();
    }

    protected void initializeComponents() {
        setPaused(paused);
        toggleMainMenu(true);
        updateSpeedText();
    }

    // Update is called once per frame
    protected async void Update() {
        
        lastYieldTime = Time.realtimeSinceStartupAsDouble;
        if (!handlingInput) {
            handlingInput = true;
            await handleInput();
            handlingInput = false;
        }
        if (!isReady) {
            return;
        }
        updateHUD();
        if (!paused) {
            if (physicsOngoing) {
                return;
            }
            cancelCalculation = false;
            physicsOngoing = true;
            double physicsFrameStart = Time.realtimeSinceStartupAsDouble;
            for (int i = 0; i < particles.Length; i++) {
                if (i % numParticleChecksPerYieldCheck == 0) {
                    await yieldCpuIfFrameTooLong();
                    if (cancelCalculation) {
                        physicsOngoing = false;
                        return;
                    }
                }
                if (!particles[i].removed) {
                    particles[i].updateNonPhysics(stepSizes[currentStepSize]);
                }
            }
            await gravitate();
            int numAlive = await updateParticles();
            updateParticleDisplay();
            numPhysicsFrames++;
            double timePassed = Time.realtimeSinceStartupAsDouble - physicsFrameStart;
            List<String> globalInformationRows = new List<string>();
            globalInformationRows.Add(string.Format("{0} particles / {1} capacity", numAlive, particles.Length));
            globalInformationRows.Add(string.Format("Simulation started: {0}", new DateTime(simulationStartTime).ToString()));
            globalInformationRows.Add(string.Format("Total simulation time: {0} s", totalSimulationTime.ToString(".00")));
            globalInformationRows.Add(string.Format("Number of timesteps: {0}", numPhysicsFrames));
            globalInformationRows.Add(string.Format("Last timestep: {0} ms", (timePassed * 1000).ToString("00.00")));
            totalSimulationTime += timePassed;
            systemInformationText.text = string.Join("\n", globalInformationRows);
            physicsOngoing = false;
        }
    }

    public virtual void onPausePressed() {
        setPaused(!paused);

    }

    public virtual void setPaused(bool newPaused) {
        this.paused = newPaused;
        pauseButtonImage.sprite = newPaused ? playSprite : pauseSprite;
    }

    public virtual async void slowDown() {
        if (currentStepSize > 0) {
            while (physicsOngoing) {
                // Debug.Log("Waiting for current physics frame to end before saving...");
                await Task.Delay(200);
            }
            currentStepSize--;
            updateSpeedText();
            gStepSize = G * stepSizes[currentStepSize];
            if (currentStepSize == 0) {
                slowDownButton.interactable = false;
            }
            speedUpButton.interactable = true;
        }
    }

    public virtual async void speedUp() {
        if (currentStepSize < stepSizes.Length - 1) {
            while (physicsOngoing) {
                // Debug.Log("Waiting for current physics frame to end before saving...");
                await Task.Delay(200);
            }
            currentStepSize++;
            updateSpeedText();
            gStepSize = G * stepSizes[currentStepSize];
            if (currentStepSize == stepSizes.Length - 1) {
                speedUpButton.interactable = false;
            }
            slowDownButton.interactable = true;
        }
    }

    public virtual void updateSpeedText() {
        double inGameSecondsPerFrame = stepSizes[currentStepSize];
        double inGameSecondsPerSecond = inGameSecondsPerFrame * targetFps;
        speedText.text = string.Format("timestep {0} / frame, {1} / second @ {2} fps",
            secondsToTimeUnit(inGameSecondsPerFrame),
            secondsToTimeUnit(inGameSecondsPerSecond),
            targetFps); 
    }

    public String secondsToTimeUnit(double seconds) {
        if (seconds > 60 * 60) {
            return string.Format("{0:F2} h", seconds / 3600);
        } else if (seconds > 60) {
            return string.Format("{0:F2} min", seconds / 60);
        } else {
            return string.Format("{0:F2} s", seconds);
        }
    }

    public virtual void onQuit() {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.ExitPlaymode();
#endif
        Application.Quit();
    }

    public static async Task yieldCpuIfFrameTooLong() {
        double currentTime = Time.realtimeSinceStartupAsDouble;
        if (currentTime > lastYieldTime + targetSecondsPerFrame) { // if more than 50 ms have passed since last yield
            await Task.Yield();
            lastYieldTime = Time.realtimeSinceStartupAsDouble;
        }
    }

    protected string buildParticleInformation(Particle particle) {
        List<string> rows = new List<string>();
        rows.Add(string.Format("Name: {0}", particle.name));
        rows.Add(string.Format("Class: {0}", particle.objectClass));
        rows.Add(string.Format("Mass: {0:E3} kg" + (particle.mass > 0.5 * jupiterMass ? " ({1:F3} M\u2609)": ""), particle.mass * massUnits, particle.mass / solarMass));
        rows.Add(string.Format("Temperature: {0:F1} K internal, {1:F1} K external", particle.temperature, particle.surfaceTemperature));
        rows.Add(string.Format("Radius: {0:E3} m" + (particle.mass > 0.5 * jupiterMass ? " ({1:F2} R\u2609)" : ""), particle.radius, particle.radius / Particle.r_Sun));
        rows.Add(string.Format("ID: {0}", particle.id));

        return string.Join("\n", rows);
    }

    protected void deselectCurrentParticle() {
        particles[selectedParticle].gameObject.layer = 0;
        focusCamera.enabled = false;
        selectionIcon.GetComponent<Image>().enabled = false;
        particleInformationText.text = "";
        selectedParticle = -1;
    }

    protected void updateParticleDisplay() {
        if (selectedParticle != -1 && !particles[selectedParticle].removed) {
            particles[selectedParticle].gameObject.layer = 3;
            focusCamera.transform.parent = particles[selectedParticle].gameObject.transform;
            focusCamera.transform.localPosition = new Vector3(0, 0, -50);
            focusCamera.orthographicSize =(float)particles[selectedParticle].gameObject.transform.localScale.x;
            particleInformationText.text = buildParticleInformation(particles[selectedParticle]);

            float newSize = Mathf.Max((float)(0.03 * mainCamera.orthographicSize), (float)(2 * particles[selectedParticle].gameObject.transform.localScale.x));
            selectionIcon.transform.localScale = new Vector3(newSize, newSize, newSize);
        }
    }

    protected async Task selectNewParticle(int particleIdx) {
        // Debug.Log(string.Format("Selecting particle {0}", particleIdx));
        bool selectionChanged = selectedParticle != particleIdx;
        if (selectedParticle != -1) {
            deselectCurrentParticle();
        }
        if (selectionChanged) {
            selectedParticle = particleIdx;
            updateParticleDisplay();

            await centerReferenceFrame(particleIdx);

            focusCamera.enabled = true;
            selectionIcon.GetComponent<Image>().enabled = true;
            selectionIcon.GetComponent<RectTransform>().position = particles[selectedParticle].gameObject.transform.position;
            
        }
    }

    protected virtual void updateHUD() {
        selectionIcon.GetComponent<RectTransform>().Rotate(new Vector3(0, 0, (float)(-100.0 / targetFps)));
    }

    protected async Task handleInput() {
        
        bool menuPressed = Input.GetAxis("Menu") != 0;

        bool pausePressed = Input.GetAxis("Pause") != 0;

        double currentTime = Time.realtimeSinceStartupAsDouble;

        if (menuPressed && currentTime - lastMenuChange > 1) {
            lastMenuChange = currentTime;
            toggleMainMenu(!mainMenuCanvas.enabled);
        }

        if (pausePressed) {
            onPausePressed();
        }

        if (!mainMenuCanvas.enabled) {
            float xMovement = Input.GetAxis("Horizontal");
            float yMovement = Input.GetAxis("Vertical");
            float xMovementMouse = Input.GetAxis("Mouse X");
            float yMovementMouse = Input.GetAxis("Mouse Y");
            float zoomChange = Input.GetAxis("Mouse ScrollWheel");
            bool isMouseHeldDown = Input.GetMouseButton(0);
            bool leftClickReleased = Input.GetMouseButtonUp(0);

            PointerEventData eventData = new PointerEventData(eventSystem);
            //Set the Pointer Event Position to that of the mouse position
            eventData.position = Input.mousePosition;

            List<RaycastResult> infoPanelRaycastTargets = new List<RaycastResult>();

            informationCanvas.GetComponent<GraphicRaycaster>().Raycast(eventData, infoPanelRaycastTargets);

            if (infoPanelRaycastTargets.Count > 0) {
                return;
            }

            if (Input.GetMouseButtonDown(0)) {
                draggedDuringClick = false;
            }
            Vector3 movement = isMouseHeldDown ? new Vector3(xMovementMouse, yMovementMouse, 0) :
                new Vector3(xMovement, yMovement);

            if (isMouseHeldDown && Abs(xMovementMouse) + Abs(yMovementMouse) > 0.3) {
                draggedDuringClick = true;
            }

            float cameraScale = mainCamera.orthographicSize;

            movement *= mainCamera.orthographicSize;

            gameObject.transform.position -= 0.05f * movement;

            zoomChange *= -5;

            Vector3 mousePosInWorld = mainCamera.ScreenToWorldPoint(Input.mousePosition);

            // only select a particle if simulation is ready, left click was released during this frame, and the screen wasn't moved at all during the click
            if (isReady && leftClickReleased && !draggedDuringClick) // try to select a particle
            {
                Particle p1; float rScreen, rWorld;
                float bestDistance = float.PositiveInfinity;
                int toSelect = -1;
                
                mousePosInWorld.z = 0;
                // Debug.Log("Mouse position of click is " + mousePosInWorld);
                // Debug.Log(string.Format("Particle 0 is at {0}, {1}", particles[0].x, particles[0].y));
                for (int i = 0; i < particles.Length; i++) {
                    p1 = particles[i];
                    if (p1.removed) {
                        continue;
                    }
                    Vector3 particleScreenPos = mainCamera.WorldToScreenPoint(p1.gameObject.transform.position);
                    particleScreenPos.z = 0;
                    
                    rScreen = (particleScreenPos - Input.mousePosition).magnitude;
                    rWorld = (p1.gameObject.transform.position - mousePosInWorld).magnitude;
                    if ((rScreen < 20 || rWorld < p1.gameObject.transform.localScale.x / 2) && rScreen < bestDistance) {
                        toSelect = i;
                        bestDistance = rScreen;
                    }
                }
                if (toSelect != -1) {
                    await selectNewParticle(toSelect);
                }
            }

            if (zoomChange != 0) {
                double oldSize = mainCamera.orthographicSize;
                float newSize = Mathf.Clamp((float)(mainCamera.orthographicSize * Pow(1.2, zoomChange)), 1e-2f, 1e3f);
                mainCamera.orthographicSize = newSize;
                if (selectedParticle != -1 && isReady) {
                    float newSelectionSize = Mathf.Max((float)(0.03 * mainCamera.orthographicSize), (float)(2 * particles[selectedParticle].gameObject.transform.localScale.x));
                    selectionIcon.transform.localScale = new Vector3(newSelectionSize, newSelectionSize, newSelectionSize);
                    Vector3 cameraToMousePosInWorld = mousePosInWorld - mainCamera.transform.position;
                    if (zoomChange < 0) {
                        mainCamera.transform.position += cameraToMousePosInWorld * (float)(1 - newSize / oldSize);
                    }
                }

                if (isReady && Particle.exaggerateSmallParticles) {
                    updateParticleDisplayScale();
                }
            }
        }
    }

    protected void updateParticleDisplayScale() {
        double targetMinDisplayScale = 0.5 * mainCamera.orthographicSize / Screen.height; // half a pixel
        if (Abs(Log(targetMinDisplayScale / Particle.minDisplayScale)) > 0.2) {
            Particle.minDisplayScale = targetMinDisplayScale;
            for (int i = 0; i < particles.Length; i++) {
                if (!particles[i].removed) {
                    particles[i].updateScale();
                }
            }
        }
    }

    protected void toggleMainMenu(bool show) {
        mainMenuCanvas.enabled = show;
        focusCamera.enabled = !show;
        informationCanvas.enabled = !show;
    }

    public async void doInit() {
        await doInitInternal();
    }

    public async Task doInitInternal() {
        initButton.interactable = false;
        isInitializing = true;
        await initParticles(defaultInitMode);
        initVariables();
        await updateParticleSurfaceTemperatures();
        for (int i = 0; i < particles.Length; i++) {
            if (i % numParticleChecksPerYieldCheck == 0) {
                await yieldCpuIfFrameTooLong();
                if (cancelCalculation) {
                    return;
                }
            }
            if (!particles[i].removed) {
                particles[i].updateNonPhysics(stepSizes[currentStepSize]);
            }
        }
        isReady = true;
        isInitializing = false;
    }

    public async void saveState() {
        string filename = "SaveData/" + GameObject.Find("Save InputField Text").GetComponent<Text>().text;
        bool oldIsReady = isReady;
        try {
            if (!isReady) {
                GameObject.Find("Save Error Text").GetComponent<Text>().text = "Nothing to save; simulation hasn't started yet";
                return;
            }
            isReady = false;
            while (physicsOngoing) {
                // Debug.Log("Waiting for current physics frame to end before saving...");
                await Task.Delay(500);
            }
            using (BinaryWriter writer = new BinaryWriter(File.Open(filename, FileMode.Create))) {
                int numLivingParticles = 0;
                for (int i = 0; i < particles.Length; i++) {
                    if (!particles[i].removed) {
                        numLivingParticles++;
                    }
                }
                // cast to long just to make the alignment prettier :)
                writer.Write((long)numLivingParticles);
                writer.Write((long)selectedParticle);
                writer.Write(numPhysicsFrames);
                writer.Write(simulationStartTime);
                writer.Write(totalSimulationTime);
                for (int i = 0; i < particles.Length; i++) {
                    if (!particles[i].removed) {
                        writer.Write(particles[i].x);
                        writer.Write(particles[i].y);
                        writer.Write(particles[i].vx);
                        writer.Write(particles[i].vy);
                        writer.Write(particles[i].mass);
                        writer.Write(particles[i].isBlackHole);
                    }
                }
            }
            GameObject.Find("Save Error Text").GetComponent<Text>().text = "";
            GameObject.Find("Save InputField Text").GetComponent<Text>().text = "";
        } catch (Exception e) {
            GameObject.Find("Save Error Text").GetComponent<Text>().text = e.Message;
        } finally {
            isReady = oldIsReady;
        }
    }

    public async void loadState() {
        string filename = "SaveData/" + GameObject.Find("Load InputField Text").GetComponent<Text>().text;

        if (isInitializing) {
            GameObject.Find("Load Error Text").GetComponent<Text>().text = "Wait for current load to complete first";
            return;
        }

        try {
            isReady = false;
            isInitializing = true;
            selectedParticle = -1;
            using (BinaryReader reader = new BinaryReader(File.Open(filename, FileMode.Open))) {
                int numToRead = (int)reader.ReadInt64();
                int selectedParticleFromFile = (int)reader.ReadInt64();
                numPhysicsFrames = reader.ReadInt64();
                simulationStartTime = reader.ReadInt64();
                totalSimulationTime = reader.ReadDouble();
                int newParticleArraySize = minParticleBufferSize;
                while (newParticleArraySize < numToRead) {
                    newParticleArraySize *= 2;
                }
                cancelCalculation = true;
                isReady = false;
                if (particles != null) {
                    for (int i = 0; i < particles.Length; i++) {
                        if (particles[i] != null && !particles[i].removed) {
                            particles[i].remove();
                        }
                    }
                }
                particles = new Particle[newParticleArraySize];
                double x, y, vx, vy, mass;
                bool isBlackHole;
                // Debug.Log(string.Format("File stated that we have {0} particles to read", numToRead));
                for (int i = 0; i < numToRead; i++) {
                    if (i % numParticleChecksPerYieldCheck == 0) {
                        await yieldCpuIfFrameTooLong();
                        GameObject.Find("Load Error Text").GetComponent<Text>().text = "Reading data" + new string('.', (int)(2 * Time.realtimeSinceStartupAsDouble) % 3);
#if UNITY_EDITOR
                        if (!UnityEditor.EditorApplication.isPlaying) {
                            return;
                        }
#endif
                    }
                    x = reader.ReadDouble();
                    y = reader.ReadDouble();
                    vx = reader.ReadDouble();
                    vy = reader.ReadDouble();
                    mass = reader.ReadDouble();
                    isBlackHole = reader.ReadByte() == 1;
                    
                    particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh, isBlackHole);
                    if (i == selectedParticleFromFile) {
                        await selectNewParticle(selectedParticle);
                    }
                }

                while (numToRead < newParticleArraySize) {
                    particles[numToRead++] = placeholderParticle;
                }
                await doInitInternal();
            }
            GameObject.Find("Load Error Text").GetComponent<Text>().text = "";
            GameObject.Find("Load InputField Text").GetComponent<Text>().text = "";
            isReady = true;
        } catch (Exception e) {
            GameObject.Find("Load Error Text").GetComponent<Text>().text = e.Message;
            isReady = false;
        } finally {
            isInitializing = false;
        }
    }

    protected virtual void initVariables() {
        mx = new double[particles.Length];
        my = new double[particles.Length];
        mvx = new double[particles.Length];
        mvy = new double[particles.Length];
        for (int i = 0; i < particles.Length; i++) {
            if (particles[i] == null) {
                particles[i] = placeholderParticle;
            }
        }
        updateParticleDisplayScale();
        physicsOutput2 = new PhysicsShaderOutputType[particles.Length];
    }

    protected async virtual Task initParticles(InitMode initMode) {
        double starMass = 1.0 * solarMass;

        double baseDistance = 5.5e8;
        double baseDistanceGrowth = 0.7;
        double baseMass = 0.01e5;
        double baseMassGrowth = 0.63;


        if (initMode == InitMode.SYSTEM) {
            particles = new Particle[512];
            particles[0] = new Particle("Star", 0, 0, 0, 0, 0, starMass, sphereMesh);
            double mass, x, y, vx, vy;
            int i;
            for (i = 1; i < 20; i++) {
                mass = baseMass * Pow(2, i * baseMassGrowth);
                x = baseDistance * Pow(2, i * baseDistanceGrowth);
                y = 0;
                vx = 0;
                vy = x / (x * Sqrt(x)) * Sqrt(G * starMass);
                particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
            }
            y = 0;
            vx = 0;

            mass = 0.0001 * particles[4].mass;
            x = particles[4].x + 0.015 * baseDistance;
            vy = particles[4].vy + Sqrt(G * particles[4].mass) / Sqrt(0.014 * baseDistance);
            particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
            i++;

            mass = 0.001 * particles[9].mass;
            x = particles[9].x + 0.3 * baseDistance;
            vy = particles[9].vy + Sqrt(G * particles[9].mass) / Sqrt(0.3 * baseDistance);
            particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
            i++;
            x = particles[9].x + 0.476 * baseDistance;
            vy = particles[9].vy + Sqrt(G * particles[9].mass) / Sqrt(0.476 * baseDistance);
            particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
            i++;
            x = particles[9].x + 0.756 * baseDistance;
            vy = particles[9].vy + Sqrt(G * particles[9].mass) / Sqrt(0.756 * baseDistance);
            particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
            i++;
            x = particles[9].x + 1.2 * baseDistance;
            vy = particles[9].vy + Sqrt(G * particles[9].mass) / Sqrt(1.2 * baseDistance);
            particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
            i++;

            mass = 0.5 * particles[15].mass;
            x = particles[15].x + 0.4 * baseDistance;
            vy = particles[15].vy + 0.75 * Sqrt(G * particles[15].mass) / Sqrt(0.30 * baseDistance);
            particles[15].vy -= 0.5 * Sqrt(G * particles[15].mass) / Sqrt(0.30 * baseDistance);
            particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
            i++;

            for (int n = 0; n < 3; n++) {
                mass = 1e-4 * particles[19].mass;
                x = particles[19].x + 0.2 * baseDistance * (n + 1);
                vy = particles[19].vy + Sqrt(G * particles[19].mass) / Sqrt(0.2 * baseDistance * (n + 1));
                particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
                i++;
            }
            for (int n = 0; n < 5; n++) {
                mass = 1e-8 * particles[19].mass;
                x = particles[19].x + 1.5 * baseDistance * ( 1 + 0.3 * n) ;
                vy = particles[19].vy + Sqrt(G * particles[19].mass) * 1.35 / Sqrt(1.5 * baseDistance * ( 1 + 0.3 * n));
                particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, mass, sphereMesh);
                i++;
            }

            for (; i < particles.Length; i++) {
                particles[i] = placeholderParticle;
            }
        } else {
            particles = new Particle[initNumParticles];
            if (initMode == InitMode.DISK) {
                particles[0] = new Particle("Star", 0, 0, 0, 0, 0, starMass, sphereMesh);
            }
            int numInFirstRing = 1024;
            int maxThisRing = numInFirstRing;
            int counter = 0;
            int numRings = 0;
            int startIndex = initMode == InitMode.DISK ? 1 : 0;

            for (int i = startIndex; i < initNumParticles; i++) {
                if (i % numParticleChecksPerYieldCheck == 0) {
                    await yieldCpuIfFrameTooLong();
                    if (cancelCalculation) {
                        return;
                    }
#if UNITY_EDITOR
                if (!UnityEditor.EditorApplication.isPlaying) {
                    return;
                }
#endif
                }
                if (counter == maxThisRing) {
                    counter = 0;
                    numRings++;
                    maxThisRing = (int)(numInFirstRing * Pow(1.004f, numRings));
                }
                int numLeft = initNumParticles - i;
                if (counter + numLeft < maxThisRing) {
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

                Vector2 rand = 0.002f * UnityEngine.Random.insideUnitCircle;

                double vx = xrot * Sqrt(G * starMass) / rsqr / rsqr / rsqr + rand.x;
                double vy = yrot * Sqrt(G * starMass) / rsqr / rsqr / rsqr + rand.y;

                particles[i] = new Particle("Particle " + i, i, x, y, vx, vy, initialMass, sphereMesh);
                counter++;
            }
        }
        await centerReferenceFrame();
        simulationStartTime = DateTime.Now.Ticks;
    }

    protected async Task initializeShader(ComputeBuffer shaderInputBuffer, ComputeBuffer shaderOutputBuffer) {
        Particle p1;
        PhysicsShaderInputType currentInput;

        NativeArray<PhysicsShaderInputType> gpuArrayInput = shaderInputBuffer.BeginWrite<PhysicsShaderInputType>(0, particles.Length);

        for (int i = 0; i < particles.Length; i++) {
            if (i % numParticleChecksPerYieldCheck == 0) {
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
        computeShader.SetFloat("stepSize", (float)stepSizes[currentStepSize]);
        computeShader.SetInt("numParticles", particles.Length);
        computeShader.SetBuffer(computeShaderKernelIndex, "inputData", shaderInputBuffer);
        computeShader.SetBuffer(computeShaderKernelIndex, "outputData", shaderOutputBuffer);
    }

    protected virtual async Task centerReferenceFrame(int particleIdx = -1) {
        double centerOfMassX = 0;
        double centerOfMassY = 0;
        double centerOfMassVX = 0;
        double centerOfMassVY = 0;
        int n = particles.Length;
        if (particleIdx == -1 || particles[particleIdx].removed) {
            double totalMass = 0;
            for (int i = 0; i < n; i++) {
                if (i % numParticleChecksPerYieldCheck == 0) {
                    await yieldCpuIfFrameTooLong();
                }
                if (!particles[i].removed) {
                    centerOfMassX += particles[i].mass * particles[i].x;
                    centerOfMassY += particles[i].mass * particles[i].y;
                    centerOfMassVX += particles[i].mass * particles[i].vx;
                    centerOfMassVY += particles[i].mass * particles[i].vy;
                    totalMass += particles[i].mass;
                }
            }
            centerOfMassX /= totalMass;
            centerOfMassY /= totalMass;
            centerOfMassVX /= totalMass;
            centerOfMassVY /= totalMass;
        } else {
            centerOfMassX = particles[particleIdx].x;
            centerOfMassY = particles[particleIdx].y;
            centerOfMassVX = particles[particleIdx].vx;
            centerOfMassVY = particles[particleIdx].vy;
        }

        for (int i = 0; i < n; i++) {
            if (i % numParticleChecksPerYieldCheck == 0) {
                await yieldCpuIfFrameTooLong();
            }
            if (!particles[i].removed) {
                particles[i].x -= centerOfMassX;
                particles[i].y -= centerOfMassY;
                particles[i].vx -= centerOfMassVX;
                particles[i].vy -= centerOfMassVY;
                particles[i].updateTransform();
            }
        }

        mainCamera.transform.position -= new Vector3((float)centerOfMassX, (float)centerOfMassY, 0) / (float)distanceScale;
    }

    protected virtual async Task gravitate() {
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
        while (currentGroup * gpuParticleGroupSize < cpuCalculateIndex && !cancelCalculation) {
            // GPU has results for us
            if (dataReady) {
                #if DEBUG_GPU
                Debug.Log(string.Format("GPU ready with particles {0} to {1}", currentGroup * gpuParticleGroupSize, (currentGroup + 1) * gpuParticleGroupSize - 1));
                #endif
                dataReady = false;

                // redirect writes to the current readable outputArray, while the previous writable array becomes the read array
                freeCurrentOutputArray = 1 - freeCurrentOutputArray;
                // If next group is valid, i.e. gpu has not already calculated everything
                if (currentGroup < numGroups - 1) {
                    // Only dispatch next batch to GPU if the CPU has not calculated all the way down to the min index for this group
                    if ((currentGroup + 1) * gpuParticleGroupSize < cpuCalculateIndex) {
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
                for (i = 0; i < gpuParticleGroupSize; i++) {
                    if (i % numParticleChecksPerYieldCheck == 0) {
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
                #if DEBUG_CPU
                Debug.Log(string.Format("CPU attempting to calculate particles {0}-{1}", cpuCalculateIndex - cpuParticleGroupSize, cpuCalculateIndex - 1));
                #endif
                for (u = 0; u < cpuParticleGroupSize; u++) {
                    if (u % numParticleChecksPerYieldCheck == 0) {
                        await yieldCpuIfFrameTooLong();
                        if (cancelCalculation) {
                            break;
                        }
                    }
                    if (cpuCalculateIndex < (currentGroup) * gpuParticleGroupSize) {
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
        if (!request.done) {
            request.WaitForCompletion();
        }
        shaderInputBuffer.Release();
        shaderOutputBuffer.Release();
    }

    protected virtual void updatePhysicsOutput(Particle p1, ref PhysicsShaderOutputType cpuOutputPixel, bool printDebug=false) {
        if (!p1.removed) {
            int j, v = 0;
            double dx, dy, massRatio, r, a;
            Particle p2;
            for (j = 0; j < particles.Length; j++) {

                p2 = particles[j];
                if (p2.removed || p1.id == p2.id) {
                    continue;
                }
                massRatio = p1.mass / p2.mass;
                dx = p2.x - p1.x;
                dy = p2.y - p1.y;
                r = Sqrt(dx * dx + dy * dy);
                a = gStepSize * p2.mass / (r * r);
                cpuOutputPixel.ax += a * dx / r;
                cpuOutputPixel.ay += a * dy / r;

                if (p1.radius + p2.radius > r && v < 4) {
                    switch (v) {
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

    protected void dispatchCompute(int groupNumber) {
        int startIndex = groupNumber * gpuParticleGroupSize;
        #if DEBUG_GPU
        Debug.Log(string.Format("This dispatch starts at particle {0}", startIndex));
        #endif
        computeShader.SetInt("startIndex", startIndex);
        computeShader.Dispatch(computeShaderKernelIndex, 1, 1, 1);
    }

    

    protected async Task<int> updateParticles() {
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
        for (i = 0; i < n; i++) {
            if (i % numParticleChecksPerYieldCheck == 0) {
                await yieldCpuIfFrameTooLong();
                if (cancelCalculation) {
                    return numAlive;
                }
            }
            if (physicsOutput2[i].idx == i) {
                numWritten++;
                if (!particles[i].removed) {
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

        for (i = 0; i < n; i++) {
            if (i % numParticleChecksPerYieldCheck == 0) {
                await yieldCpuIfFrameTooLong();
                if (cancelCalculation) {
                    return numAlive;
                }
            }

            if (particles[i].removed) {
                continue;
            }
            p1 = particles[i];
            outputPixel = physicsOutput2[i];
            #if UNITY_EDITOR
            if (outputPixel.idx != i) {
                if (canPrintError) {
                    Debug.LogError(string.Format("Particle {0} was reported to have id {1} by shader", i, outputPixel.idx));
                    canPrintError = false;
                }
                UnityEditor.EditorApplication.ExitPlaymode();
                await Task.Yield();
            }
            if (double.IsNaN(outputPixel.ax) || outputPixel.ax == 0) {
                if (canPrintError) {
                    Debug.LogError(string.Format("Particle {0} has x-acceleration {1}", i, outputPixel.ax));
                    canPrintError = false;
                }
                UnityEditor.EditorApplication.ExitPlaymode();
                await Task.Yield();
            }
            #endif
            p1.vx += outputPixel.ax / (numPhysicsFrames == 0 ? 2 : 1); // initial frame applies only a half step to v
            p1.vy += outputPixel.ay / (numPhysicsFrames == 0 ? 2 : 1); // initial frame applies only a half step to v; leapfrog integration
            mx[i] = particles[i].mass * particles[i].x;
            my[i] = particles[i].mass * particles[i].y;
            mvx[i] = particles[i].mass * particles[i].vx;
            mvy[i] = particles[i].mass * particles[i].vy;
        }
        for (i = 0; i < n; i++) {
            if (i % numParticleChecksPerYieldCheck == 0) {
                await yieldCpuIfFrameTooLong();
                if (cancelCalculation) {
                    return numAlive;
                }
            }
            p1 = particles[i];
            outputPixel = physicsOutput2[i];
            if (particles[i].removed) {
                continue;
            }
            for (k = 0; k < 4; k++) {
                int collisionIndex = -1;
                switch (k) {
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
                if (collisionIndex < -1 || collisionIndex >= particles.Length) {
                    throw new System.Exception(string.Format("Bad collision index: {0}", collisionIndex));
                }
                if (collisionIndex != -1 && !particles[collisionIndex].removed) {
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
                    if (selectedParticle == j) {
                        await selectNewParticle(i);
                    }
                    p2.remove();
                    await yieldCpuIfFrameTooLong();
                    if (cancelCalculation) {
                        return numAlive;
                    }
                }
            }
        }

        bool transformUpdated;
        for (i = 0; i < particles.Length; i++) {
            if (i % numParticleChecksPerYieldCheck == 0) {
                await yieldCpuIfFrameTooLong();
                if (cancelCalculation) {
                    return numAlive;
                }
            }
            if (!particles[i].removed) {
                transformUpdated = particles[i].updatePhysics(stepSizes[currentStepSize]);

                if (i == selectedParticle && transformUpdated) {
                    gameObject.transform.position += particles[i].lastRenderDifference;
                    selectionIcon.GetComponent<RectTransform>().position = particles[selectedParticle].gameObject.transform.position;
                }
            }
        }
        await updateParticleSurfaceTemperatures();
        int newParticleArraySize = 3 * particles.Length / 4;

        if (numAlive <= newParticleArraySize && newParticleArraySize >= minParticleBufferSize) {
            reduceParticleCount(newParticleArraySize);
        }

        return numAlive;
    }

    /**
     * Update particle surface temperatures by calculating total power flux at the surface
     * Flux at the surface is equal to internal temperature flux plus incoming flux of top n
     * most luminous light sources in the scene
     *
     * Some simple cases include:
     * 
     * Only internal heating: T = T_internal
     * 
     * No internal heating, single star as heat source with surfaces at 0 separation:
     *  T = T_star / (4*(1-albedo))^(1/4)
     *
     * No internal heating, single star as heat source at distance d >> r, where
     * r is radius of particle: T = (L_star * ( 1 - albedo) / (16 * pi * SB * d^2))^(1/4)
     * 
     * SB is the Stefan-Boltzmann constant.
     * 
     */
    protected virtual async Task updateParticleSurfaceTemperatures() {
        List<Particle> heatSources = new List<Particle>(maxHeatSources);
        for (int i = 0; i < particles.Length; i++) {
            if (i % numParticleChecksPerYieldCheck == 0) {
                await yieldCpuIfFrameTooLong();
            }
            if (!particles[i].removed && particles[i].luminosity > 1e12) {
                    heatSources.Add(particles[i]);
                    heatSources.Sort((p1, p2) => (p1.luminosity > p2.luminosity ? 1 : -1));
                if (heatSources.Count > maxHeatSources) {
                    heatSources.RemoveAt(0);
                    heatSources.Sort((p1, p2) => (p1.luminosity > p2.luminosity ? 1 : -1));
                }
            }
        }
        for (int i = 0; i < particles.Length; i++) {
            if (!particles[i].removed) {
                double r2 = particles[i].radius;
                particles[i].surfaceTemperature = 4 * PI * Particle.SB * r2 * Pow(particles[i].temperature, 4); // self-generated power
                for (int j = 0; j < heatSources.Count; j++) {
                    if (particles[i].id == heatSources[j].id) {
                        continue;
                    }
                    double dx = particles[i].x - heatSources[j].x;
                    double dy = particles[i].y - heatSources[j].y;
                    double h2 = (dx * dx + dy * dy);

                    double rSquaredOverHSquared = r2 / h2;

                    double solidAngleFraction = rSquaredOverHSquared < 1e-6
                        ? rSquaredOverHSquared / 4
                        : 0.5 * (1 - 1 / Sqrt(rSquaredOverHSquared + 1)); // calculation unstable for r/h < 1e-5, approximation accurate to 5 decimal places when (r/h)^2 = 1e-6
                    double numerator = heatSources[j].luminosity * (1 - particles[i].albedo);
                    particles[i].surfaceTemperature += numerator * solidAngleFraction; // intercepted power from considered heat source
                }
                particles[i].surfaceTemperature = Pow(particles[i].surfaceTemperature / (4 * PI * r2 * Particle.SB), 0.25); // final temperature
            }
        }
    }

    protected virtual void reduceParticleCount(int newParticleArraySize) {
        Particle[] newParticles = new Particle[newParticleArraySize];
        int idx = 0;
        for (int i = 0; i < particles.Length; i++) {
            if (!particles[i].removed) {
                newParticles[idx] = particles[i];
                particles[i].id = idx;
                if (selectedParticle == i) {
                    selectedParticle = idx;
                }
                idx++;
            }
        }
        while (idx < newParticleArraySize) {
            newParticles[idx++] = placeholderParticle;
        }

        particles = newParticles;
        physicsOutput2 = new PhysicsShaderOutputType[particles.Length];
        mx = new double[particles.Length];
        my = new double[particles.Length];
        mvx = new double[particles.Length];
        mvy = new double[particles.Length];
    }

    public void addParticle(Particle p) {
        
    }
}
