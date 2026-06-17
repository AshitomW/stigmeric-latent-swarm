// State management
let systemStatus = { data_exists: false, model_exists: false };
let isTraining = false;
let isExploring = false;
let exploreInterval = null;
let viewMode = 'generated'; // 'generated' or 'reference'
let originalData = null;
let currentFrame = 0;
let playbackSpeed = 1;
let currentTheme = 'dark';

// Swarm explorer visual variables
let swarmAgentsData = []; // Trajectories: [num_agents, T, N, 3]
let latentPositions = []; // Latent coords: [num_agents, 2]
let agentFitness = [];    // Fitness: [num_agents]

// Three.js instances
let scene, camera, renderer, controls;
let particlesGroup, edgesGroup, referenceGroup;
let agentMeshes = []; // List of agent particle groups
let agentEdgeLines = []; // List of agent connection lines
let agentTrails = []; // List of historical coordinate trails for each agent node
const MAX_TRAIL_POINTS = 50;

// Chart.js instances
let latentChart = null;

// Color palettes for visualization
const COLOR_PALETTES = {
    dark: {
        primary: '#3b82f6',
        secondary: '#9ca3af',
        gridColor: '#2d2d2d',
        textColor: '#f3f4f6',
        nodeBest: '#10b981',
        nodeNormal: '#3b82f6',
        edgeNormal: 'rgba(59, 130, 246, 0.25)',
        edgeBest: 'rgba(16, 185, 129, 0.8)'
    }
};

// API Base URL (local dev server)
const API_URL = '';

// On Document Load
document.addEventListener('DOMContentLoaded', () => {
    initUI();
    initThree();
    initChart();
    checkStatus();
    // Poll status every 3 seconds to update GUI
    setInterval(checkStatus, 3000);
});

// UI Setup & Bindings
function initUI() {
    // Slider value synchronization
    const sliders = [
        { id: 'train-epochs', valId: 'train-epochs-val' },
        { id: 'swarm-agents', valId: 'swarm-agents-val' },
        { id: 'param-inertia', valId: 'param-inertia-val' },
        { id: 'param-stigmergy', valId: 'param-stigmergy-val' },
        { id: 'param-mutual', valId: 'param-mutual-val' },
        { id: 'param-goal', valId: 'param-goal-val' }
    ];
    
    sliders.forEach(slider => {
        const input = document.getElementById(slider.id);
        const valueSpan = document.getElementById(slider.valId);
        input.addEventListener('input', () => {
            valueSpan.textContent = input.value;
        });
    });

    // Button Bindings
    document.getElementById('btn-train').addEventListener('click', startTraining);
    document.getElementById('btn-explore-start').addEventListener('click', startExplorer);
    document.getElementById('btn-explore-stop').addEventListener('click', stopExplorer);
    document.getElementById('btn-explore-reset').addEventListener('click', resetExplorer);
    
    // View mode toggle (Reference spring trajectories vs Generated choreography)
    document.getElementById('btn-view-mode').addEventListener('click', () => {
        const btn = document.getElementById('btn-view-mode');
        if (viewMode === 'generated') {
            viewMode = 'reference';
            btn.textContent = 'Show Generated Swarms';
            particlesGroup.visible = false;
            edgesGroup.visible = false;
            referenceGroup.visible = true;
            loadReferenceData();
        } else {
            viewMode = 'generated';
            btn.textContent = 'Show Reference Dynamics';
            particlesGroup.visible = true;
            edgesGroup.visible = true;
            referenceGroup.visible = false;
        }
    });
}

// Check Backend status (Weights and Datasets presence)
async function checkStatus() {
    try {
        const response = await fetch(`${API_URL}/api/status`);
        const status = await response.json();
        systemStatus = status;
        
        // Update indicators
        const dataBadge = document.getElementById('status-data');
        const modelBadge = document.getElementById('status-model');
        
        if (status.data_exists) {
            dataBadge.textContent = 'Available';
            dataBadge.className = 'badge badge-success';
        } else {
            dataBadge.textContent = 'Not Found';
            dataBadge.className = 'badge badge-error';
        }
        
        if (status.model_exists) {
            modelBadge.textContent = 'Trained';
            modelBadge.className = 'badge badge-success';
            document.getElementById('btn-explore-start').disabled = isExploring;
        } else {
            modelBadge.textContent = 'Untrained';
            modelBadge.className = 'badge badge-error';
            document.getElementById('btn-explore-start').disabled = true;
        }

        // Handle training progress updates
        const trainProgress = status.training;
        const trainBtn = document.getElementById('btn-train');
        const progressContainer = document.getElementById('training-progress-container');
        const progressBar = document.getElementById('training-progress-bar');
        const progressMsg = document.getElementById('training-status-msg');

        if (trainProgress.status === 'running') {
            isTraining = true;
            trainBtn.disabled = true;
            progressContainer.classList.remove('hidden');
            const pct = (trainProgress.epoch / trainProgress.max_epochs) * 100;
            progressBar.style.width = `${pct}%`;
            progressMsg.textContent = `${trainProgress.message} (${Math.round(pct)}%) - Loss: ${trainProgress.loss.toFixed(5)}`;
        } else {
            if (isTraining && trainProgress.status === 'completed') {
                isTraining = false;
                trainBtn.disabled = false;
                progressContainer.classList.add('hidden');
                
                // Reload matplotlib figure tags to show newly compiled curves
                document.getElementById('fig-loss-curve').src = `/backend/data/loss_curve.png?t=${Date.now()}`;
                document.getElementById('fig-latent-space').src = `/backend/data/latent_space.png?t=${Date.now()}`;
                document.getElementById('fig-loss-curve').style.display = 'block';
                document.getElementById('fig-latent-space').style.display = 'block';
                document.getElementById('fig-loss-placeholder').classList.add('hidden');
                document.getElementById('fig-latent-placeholder').classList.add('hidden');
            }
            
            // If already trained on load, make sure images are shown
            if (status.model_exists && !isTraining) {
                const imgLoss = document.getElementById('fig-loss-curve');
                const imgLatent = document.getElementById('fig-latent-space');
                if (imgLoss.style.display !== 'block') {
                    imgLoss.src = `/backend/data/loss_curve.png?t=${Date.now()}`;
                    imgLatent.src = `/backend/data/latent_space.png?t=${Date.now()}`;
                    imgLoss.style.display = 'block';
                    imgLatent.style.display = 'block';
                    document.getElementById('fig-loss-placeholder').classList.add('hidden');
                    document.getElementById('fig-latent-placeholder').classList.add('hidden');
                }
            }
        }
    } catch (error) {
        console.error("Error connecting to server:", error);
    }
}

// Trigger training
async function startTraining() {
    const epochs = document.getElementById('train-epochs').value;
    try {
        const response = await fetch(`${API_URL}/api/train?epochs=${epochs}`, { method: 'POST' });
        if (response.ok) {
            isTraining = true;
            document.getElementById('btn-train').disabled = true;
            document.getElementById('training-progress-container').classList.remove('hidden');
            document.getElementById('training-progress-bar').style.width = '0%';
            document.getElementById('training-status-msg').textContent = 'Initializing training loop...';
        }
    } catch (error) {
        console.error("Error starting training:", error);
    }
}

// 3D Scene Initialization (Three.js)
function initThree() {
    const container = document.getElementById('three-canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Create scene, camera, renderer
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 5, 12);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.1; // Don't go below floor

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(5, 10, 7);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x3b82f6, 0.4);
    dirLight2.position.set(-5, -5, -5);
    scene.add(dirLight2);

    // Grid Floor
    const gridHelper = new THREE.GridHelper(20, 20, 0x222222, 0x111111);
    gridHelper.position.y = -2;
    scene.add(gridHelper);

    // Container Groups
    particlesGroup = new THREE.Group();
    edgesGroup = new THREE.Group();
    referenceGroup = new THREE.Group();
    
    scene.add(particlesGroup);
    scene.add(edgesGroup);
    scene.add(referenceGroup);
    
    referenceGroup.visible = false;

    // Handle Window Resize
    window.addEventListener('resize', () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    // Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        
        // Update controls
        controls.update();
        
        // Spin reference group slightly for dynamic viewing
        if (referenceGroup.visible && originalData) {
            renderReferenceFrame();
        }
        
        renderer.render(scene, camera);
    }
    animate();
}

// Initialize latent space plot (Chart.js)
function initChart() {
    const ctx = document.getElementById('latent-space-chart').getContext('2d');
    
    const colors = COLOR_PALETTES[currentTheme];
    
    latentChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Swarm Explorer Agents',
                    data: [],
                    backgroundColor: '#10b981',
                    borderColor: '#047857',
                    borderWidth: 1,
                    pointRadius: 6,
                    pointHoverRadius: 8
                },
                {
                    label: 'Historical Exploration Paths',
                    data: [],
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    borderColor: 'rgba(59, 130, 246, 0.3)',
                    borderWidth: 1,
                    pointRadius: 2,
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: colors.gridColor },
                    ticks: { color: colors.textColor },
                    title: { display: true, text: 'Latent Component 1 (PCA)', color: colors.textColor }
                },
                y: {
                    grid: { color: colors.gridColor },
                    ticks: { color: colors.textColor },
                    title: { display: true, text: 'Latent Component 2 (PCA)', color: colors.textColor }
                }
            },
            plugins: {
                legend: {
                    labels: { color: colors.textColor }
                }
            }
        }
    });
}


// Load reference Spring data from backend
async function loadReferenceData() {
    if (originalData) return; // Already loaded
    try {
        const response = await fetch(`${API_URL}/api/trajectories`);
        if (response.ok) {
            originalData = await response.json();
            setupReferenceVisualizers();
        }
    } catch (e) {
        console.error("Failed to load reference trajectories:", e);
    }
}

// Build meshes for original reference dynamics
function setupReferenceVisualizers() {
    referenceGroup.clear();
    const trajectories = originalData.trajectories; // shape: (num_sims, steps, nodes, 3)
    const edgeIndex = originalData.edge_index;       // shape: (2, E)
    
    const numSimulations = trajectories.length;
    const numNodes = trajectories[0][0].length;
    
    // Create meshes for 1 simulation at a time (e.g. index 0)
    // Spheres
    const sphereGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const sphereMat = new THREE.MeshStandardMaterial({ color: 0xef4444 });
    
    const simGroup = new THREE.Group();
    const nodeMeshes = [];
    
    for (let i = 0; i < numNodes; i++) {
        const mesh = new THREE.Mesh(sphereGeo, sphereMat);
        simGroup.add(mesh);
        nodeMeshes.push(mesh);
    }
    
    // Lines
    const lines = [];
    const lineMat = new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 2, transparent: true, opacity: 0.6 });
    for (let e = 0; e < edgeIndex[0].length; e++) {
        const u = edgeIndex[0][e];
        const v = edgeIndex[1][e];
        
        // Unique pair only (since it is bidirectional in graph, we prevent duplicates)
        if (u < v) {
            const geom = new THREE.BufferGeometry();
            const line = new THREE.Line(geom, lineMat);
            simGroup.add(line);
            lines.push({ line, u, v });
        }
    }
    
    referenceGroup.add(simGroup);
    originalData.nodeMeshes = nodeMeshes;
    originalData.lines = lines;
    originalData.currentSimIndex = 0;
}

// Render one step of the reference dataset
function renderReferenceFrame() {
    const trajectories = originalData.trajectories;
    const simIdx = originalData.currentSimIndex;
    const simTraj = trajectories[simIdx]; // (steps, nodes, 3)
    const numSteps = simTraj.length;
    
    currentFrame = (currentFrame + 1) % numSteps;
    const frameData = simTraj[currentFrame]; // (nodes, 3)
    
    // Update spheres
    for (let i = 0; i < frameData.length; i++) {
        originalData.nodeMeshes[i].position.set(frameData[i][0], frameData[i][1], frameData[i][2]);
    }
    
    // Update spring lines
    originalData.lines.forEach(pair => {
        const p1 = frameData[pair.u];
        const p2 = frameData[pair.v];
        const vertices = new Float32Array([
            p1[0], p1[1], p1[2],
            p2[0], p2[1], p2[2]
        ]);
        pair.line.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        pair.line.geometry.computeBoundingSphere();
    });
}

// Clear all active explorer meshes
function clearGeneratedVisuals() {
    particlesGroup.clear();
    edgesGroup.clear();
    agentMeshes = [];
    agentEdgeLines = [];
    agentTrails = [];
}

// Swarm Explorer Setup
async function startExplorer() {
    isExploring = true;
    document.getElementById('btn-explore-start').disabled = true;
    document.getElementById('btn-explore-stop').disabled = false;
    document.getElementById('btn-explore-reset').disabled = false;
    document.getElementById('swarm-objective').disabled = true;
    document.getElementById('swarm-agents').disabled = true;
    
    const config = getExplorerConfig();
    
    // 1. Initialise backend explorer
    try {
        const response = await fetch(`${API_URL}/api/explore/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const res = await response.json();
        
        // 2. Setup Three.js meshes
        setupExplorerVisualizers(config.num_agents);
        
        // 3. Clear PCA path dataset
        latentChart.data.datasets[0].data = [];
        latentChart.data.datasets[1].data = [];
        latentChart.update();
        
        // 4. Run step loop
        exploreInterval = setInterval(runExplorerStep, 100);
        
    } catch (e) {
        console.error("Error starting explorer:", e);
        isExploring = false;
        resetExplorer();
    }
}

// Read parameters from UI
function getExplorerConfig() {
    return {
        objective: document.getElementById('swarm-objective').value,
        num_agents: parseInt(document.getElementById('swarm-agents').value),
        w_inertia: parseFloat(document.getElementById('param-inertia').value),
        c_stigmergy: parseFloat(document.getElementById('param-stigmergy').value),
        c_mutual: parseFloat(document.getElementById('param-mutual').value),
        c_goal: parseFloat(document.getElementById('param-goal').value),
        dt: 0.1
    };
}

// Setup meshes for each agent in the swarm
function setupExplorerVisualizers(numAgents) {
    clearGeneratedVisuals();
    
    // We assume fully connected graph of 5 particles
    const numNodes = 5;
    
    // Build edge list for fully connected 5 nodes
    const edgeIndex = [];
    for (let i = 0; i < numNodes; i++) {
        for (let j = i + 1; j < numNodes; j++) {
            edgeIndex.push([i, j]);
        }
    }
    
    const colors = COLOR_PALETTES[currentTheme];
    
    for (let a = 0; a < numAgents; a++) {
        // Group for this agent's multi-agent structure
        const agentGroup = new THREE.Group();
        particlesGroup.add(agentGroup);
        
        // Nodes
        const sphereGeo = new THREE.SphereGeometry(0.08, 12, 12);
        // Translucent materials for normal agents, solid glowing for best agent
        const sphereMat = new THREE.MeshStandardMaterial({
            color: colors.nodeNormal,
            transparent: true,
            opacity: 0.4
        });
        
        const nodeMeshes = [];
        for (let i = 0; i < numNodes; i++) {
            const mesh = new THREE.Mesh(sphereGeo, sphereMat);
            agentGroup.add(mesh);
            nodeMeshes.push(mesh);
        }
        agentMeshes.push(nodeMeshes);
        
        // Springs
        const lineMat = new THREE.LineBasicMaterial({
            color: colors.primary,
            transparent: true,
            opacity: 0.15
        });
        const lines = [];
        
        edgeIndex.forEach(pair => {
            const geom = new THREE.BufferGeometry();
            const line = new THREE.Line(geom, lineMat);
            agentGroup.add(line);
            lines.push({ line, u: pair[0], v: pair[1] });
        });
        agentEdgeLines.push(lines);
        
        // Trails
        // Keep a history buffer of coordinates for trail ribbons
        const trailsForAgent = [];
        for (let i = 0; i < numNodes; i++) {
            trailsForAgent.push([]);
        }
        agentTrails.push(trailsForAgent);
    }
}

// Run single step of swarm exploration
async function runExplorerStep() {
    if (!isExploring) return;
    
    const config = getExplorerConfig();
    
    try {
        const response = await fetch(`${API_URL}/api/explore/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) return;
        
        const result = await response.json();
        const trajectories = result.trajectories; // shape: [num_agents, T, N, 3]
        const z = result.latent_positions;       // shape: [num_agents, 2]
        const fitness = result.fitness;           // shape: [num_agents]
        
        // Find best agent index (max fitness)
        let bestAgentIdx = 0;
        let maxFit = -999999;
        for (let a = 0; a < fitness.length; a++) {
            if (fitness[a] > maxFit) {
                maxFit = fitness[a];
                bestAgentIdx = a;
            }
        }
        
        // Update Chart
        updateExplorerChart(z, bestAgentIdx);
        
        // Update 3D Visualizer
        // For each agent, we animate their generated trajectory over T steps
        // To show smooth motion, we step through the T frames generated by the ST-GVAE
        animateGeneratedTrajectories(trajectories, bestAgentIdx);
        
    } catch (e) {
        console.error("Error stepping explorer:", e);
    }
}

let animFrame = 0;
function animateGeneratedTrajectories(trajectories, bestAgentIdx) {
    const numAgents = trajectories.length;
    const seqLen = trajectories[0].length;
    const numNodes = trajectories[0][0].length;
    
    // Advance playback frame within the generated sequence length T
    animFrame = (animFrame + 1) % seqLen;
    
    const colors = COLOR_PALETTES[currentTheme];
    
    for (let a = 0; a < numAgents; a++) {
        const isBest = (a === bestAgentIdx);
        const frameData = trajectories[a][animFrame]; // (nodes, 3)
        
        // Update nodes positions and colors
        for (let i = 0; i < numNodes; i++) {
            const pos = frameData[i];
            const mesh = agentMeshes[a][i];
            mesh.position.set(pos[0], pos[1], pos[2]);
            
            // Adjust appearance of Best Agent vs Normal agents
            if (isBest) {
                mesh.material.color.set(colors.nodeBest);
                mesh.material.opacity = 0.95;
                mesh.scale.set(1.4, 1.4, 1.4);
            } else {
                mesh.material.color.set(colors.nodeNormal);
                mesh.material.opacity = 0.35;
                mesh.scale.set(1.0, 1.0, 1.0);
            }
        }
        
        // Update edges
        agentEdgeLines[a].forEach(pair => {
            const p1 = frameData[pair.u];
            const p2 = frameData[pair.v];
            const vertices = new Float32Array([
                p1[0], p1[1], p1[2],
                p2[0], p2[1], p2[2]
            ]);
            pair.line.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            pair.line.geometry.computeBoundingSphere();
            
            if (isBest) {
                pair.line.material.color.set(colors.nodeBest);
                pair.line.material.opacity = 0.7;
            } else {
                pair.line.material.color.set(colors.primary);
                pair.line.material.opacity = 0.1;
            }
        });
    }
}

// Update the 2D scatter plot with active agent coordinates and pheromone history
function updateExplorerChart(zData, bestAgentIdx) {
    if (!latentChart) return;
    
    // Active positions
    const activePoints = zData.map((pos, idx) => ({
        x: pos[0],
        y: pos[1]
    }));
    
    // Extract best agent position for visual highlight on chart if desired
    // (Here we treat all active agents as one dataset)
    latentChart.data.datasets[0].data = activePoints;
    
    // Append to historical paths (pheromone deposits)
    // To prevent overload, we limit historical plot length to 300 points
    const histData = latentChart.data.datasets[1].data;
    activePoints.forEach(pt => {
        histData.push(pt);
    });
    if (histData.length > 400) {
        latentChart.data.datasets[1].data = histData.slice(-400);
    }
    
    // Update chart without resetting view
    latentChart.update('none');
}

function stopExplorer() {
    isExploring = false;
    clearInterval(exploreInterval);
    document.getElementById('btn-explore-start').disabled = false;
    document.getElementById('btn-explore-stop').disabled = true;
}

async function resetExplorer() {
    stopExplorer();
    document.getElementById('btn-explore-stop').disabled = true;
    document.getElementById('btn-explore-reset').disabled = true;
    document.getElementById('swarm-objective').disabled = false;
    document.getElementById('swarm-agents').disabled = false;
    
    try {
        await fetch(`${API_URL}/api/explore/reset`, { method: 'POST' });
        clearGeneratedVisuals();
        latentChart.data.datasets[0].data = [];
        latentChart.data.datasets[1].data = [];
        latentChart.update();
    } catch (e) {
        console.error("Failed to reset explorer:", e);
    }
}
