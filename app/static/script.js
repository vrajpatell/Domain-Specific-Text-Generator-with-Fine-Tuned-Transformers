const promptInput = document.getElementById("prompt");
const maxLengthInput = document.getElementById("maxLength");
const generateButton = document.getElementById("generateBtn");
const output = document.getElementById("output");
const status = document.getElementById("status");

async function generateText() {
  const prompt = promptInput.value.trim();
  const maxLength = Number(maxLengthInput.value);

  if (!prompt) {
    status.textContent = "Please enter a prompt.";
    return;
  }

  generateButton.disabled = true;
  status.textContent = "Generating text...";
  output.textContent = "";

  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_length: maxLength }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Generation failed.");
    }

    output.textContent = data.generated_text;
    status.textContent = `Done using model source: ${data.model_source}`;
  } catch (error) {
    output.textContent = "";
    status.textContent = `Error: ${error.message}`;
  } finally {
    generateButton.disabled = false;
  }
}

generateButton.addEventListener("click", generateText);
