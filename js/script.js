// DOM Elements
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const timerElement = document.getElementById("timer");
const randomLabelElement = document.getElementById("randomLabel");
const predictedLabelElement = document.getElementById("predictedLabel");

// State Variables
let drawing = false;
let timerId;
let countdown = 30;

// Utility Functions
const fetchJson = async(url) => await (await fetch(url)).json();

const getRandomNumber = (max) => Math.floor(Math.random() * (max + 1));

// Core Functions
const loadClassIndexMap = () => fetchJson('./assets/label_map.json');

const loadModel = async() => {
    const session = new onnx.InferenceSession();
    await session.loadModel("./assets/quick-draw.onnx");
    return session;
};

const runInference = async(session, inputTensor) => {
    const input = new onnx.Tensor(inputTensor, 'float32', [1, 1, 28, 28]);
    const outputMap = await session.run([input]);
    const outputData = outputMap.values().next().value.data;
    return outputData.reduce((maxIndex, item, index, array) =>
        array[maxIndex] < item ? index : maxIndex, 0);
};

const preprocessCanvas = () => {
    // Create a temporary canvas to resize the drawing
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    // Draw the content of the original canvas onto the temporary one, scaling it down
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);

    // Get the image data from the temporary canvas
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;

    let grayValues = [];

    // Convert the image to grayscale
    for (let i = 0; i < data.length; i += 4) {
        grayValues.push(data[i + 3]);
    }
    console.log(grayValues);

    return new Float32Array(grayValues);
};

// Event Handlers
const handleMouseDown = (event) => {
    drawing = true;
    draw(event);
};

const handleMouseUp = () => {
    drawing = false;
    ctx.beginPath();
};

const handleMouseMove = (event) => {
    if (drawing) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.strokeStyle = "black";

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }
};

// Main Function
const main = async() => {
    if (!timerElement || !randomLabelElement) {
        console.error("Required elements not found!");
        return;
    }

    const session = await loadModel();
    const labelMap = await loadClassIndexMap();
    const maxKey = Math.max(...Object.keys(labelMap).map(Number));

    const setRandomLabel = () => {
        const randomIndex = getRandomNumber(maxKey);
        const randomLabel = labelMap[randomIndex];
        randomLabelElement.textContent = randomLabel;
    };

    const resetTimer = () => {
        clearInterval(timerId);
        timerId = setInterval(() => {
            if (--countdown <= 0) {
                countdown = 30;
                setRandomLabel();
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            timerElement.textContent = countdown;
        }, 1000);
    };

    // Initialize
    setRandomLabel();
    resetTimer();

    // Continuous Prediction
    const runContinuousPrediction = async() => {
        const inputTensor = preprocessCanvas();
        const classIndex = await runInference(session, inputTensor);
        predictedLabelElement.textContent = labelMap[classIndex];
        setTimeout(() => requestAnimationFrame(runContinuousPrediction), 200);
    };

    runContinuousPrediction();
};

// Initialize
canvas.addEventListener("mousedown", handleMouseDown);
canvas.addEventListener("mouseup", handleMouseUp);
canvas.addEventListener("mousemove", handleMouseMove);

main().catch(console.error);