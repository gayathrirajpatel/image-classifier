const classifier = knnClassifier.create();
const webcamElement = document.getElementById("webcam");

let net;

// Function to speak the sentence
function speakSentence(sentence) {
  const synth = window.speechSynthesis;
  const utterance = new SpeechSynthesisUtterance(sentence);
  synth.speak(utterance);
}

async function app() {
  console.log("Loading MobileNet...");

  net = await mobilenet.load();

  console.log("Loaded model");

  const webcam = await tf.data.webcam(webcamElement);

  const addExample = async (classId) => {
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);
    img.dispose();
  };

  document.getElementById("Hi").addEventListener("click", () => addExample(0));
  document.getElementById("I").addEventListener("click", () => addExample(1));
  document.getElementById("am").addEventListener("click", () => addExample(2));
  document.getElementById("G").addEventListener("click", () => addExample(3));

  let predictedClasses = [];

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();
      const activation = net.infer(img, "conv_preds");
      const result = await classifier.predictClass(activation);
      const classes = ["Hi", "I", "am", "G"];

      predictedClasses.push(classes[result.label]);

      const predictionText = `
          Prediction: ${classes[result.label]}\n
          Probability: ${result.confidences[result.label]}
      `;

      document.getElementById("console").innerText = predictionText;

      img.dispose();
    }

    await tf.nextFrame();
  }

  // Form a sentence with predicted classes and speak it after exiting the loop
  if (predictedClasses.length > 0) {
    const sentence = predictedClasses.join(" ");
    speakSentence(sentence);
    predictedClasses = []; // Reset for the next sentence
  }
}

app();
