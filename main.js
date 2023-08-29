let canvas = document.getElementById("drawCanvas");
let ctx = canvas.getContext("2d");

let drawing = false;

async function loadModel() {
    const session = new onnx.InferenceSession();
    await session.loadModel("path/to/your/model.onnx");
    return session;
}

function preprocessImage(image) {
    // Your image preprocessing code here.
    // You can use canvas or other methods to resize and normalize the image.

    // Convert to Float32Array, reshape to 1x1x28x28 or whatever your input shape is.
    const inputTensor = new Float32Array(1 * 1 * 28 * 28);

    // Fill 'inputTensor' with your processed image data

    return inputTensor;
}

async function runInference(session, inputTensor) {
    const input = new onnx.Tensor(inputTensor, 'float32', [1, 1, 28, 28]);
    const outputMap = await session.run([input]);
    const outputData = outputMap.values().next().value.data;

    // Find the class index with the maximum score
    let maxScore = -Infinity;
    let classIndex = -1;
    for (let i = 0; i < outputData.length; i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            classIndex = i;
        }
    }

    return classIndex;
}

canvas.addEventListener("mousedown", function() {
    drawing = true;
    draw(event); // This will ensure a dot appears if the user just clicks the canvas
});

canvas.addEventListener("mouseup", function() {
    drawing = false;
    ctx.beginPath(); // This ensures the start of a new line after lifting the mouse
});

canvas.addEventListener("mousemove", draw);

function draw(event) {
    if (!drawing) return;

    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
}

async function main() {
    // Step 2: Load the model
    const session = await loadModel();

    // Assume you have an HTML image element or canvas for the image you want to use
    const image = document.getElementById("your-image-element");

    // Step 3: Preprocess the image into a tensor
    const inputTensor = preprocessImage(image);

    // Step 4: Run inference and get the class index
    const classIndex = await runInference(session, inputTensor);

    console.log(`Predicted class index: ${classIndex}`);
}

// Run the main function
main().catch(console.error);