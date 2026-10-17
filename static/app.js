
const $ = (id) => document.getElementById(id);

let documentReady = false;
let busy = false;

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");

  window.setTimeout(() => {
    node.classList.remove("show");
  }, 3500);
}

function setBusy(value) {
  busy = value;

  $("send-button").disabled =
    !documentReady || busy || !$("question").value.trim();

  $("choose-file").disabled = busy;
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.error || `Request failed (${response.status})`
    );
  }

  return data;
}

function showOutput() {
  $("welcome").classList.add("hidden");
  $("output").classList.remove("hidden");
}

function addMessage(label, text, kind = "") {
  showOutput();

  const card = document.createElement("div");
  card.className = `message ${kind}`;

  const heading = document.createElement("div");
  heading.className = "message-label";
  heading.textContent = label;

  const content = document.createElement("div");
  content.textContent = text;

  card.append(heading, content);
  $("output").append(card);

  card.scrollIntoView({
    behavior: "smooth",
    block: "nearest"
  });

  return card;
}

function setDocumentReady(data) {
  documentReady = true;

  $("document-banner").classList.remove("hidden");

  $("doc-name").textContent =
    data.filename || "Uploaded PDF";

  $("doc-details").textContent =
    `${data.pages} pages · ${data.chunks} searchable sections`;

  $("question").disabled = false;
  $("question").placeholder =
    "Ask a question about your document...";

  setBusy(false);

  $("welcome").classList.remove("hidden");
  $("output").classList.add("hidden");
  $("output").replaceChildren();
}

async function uploadFile(file) {
  if (!file) return;

  if (
    file.type !== "application/pdf" &&
    !file.name.toLowerCase().endsWith(".pdf")
  ) {
    toast("Please choose a PDF file.");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  $("upload-status").textContent = "Processing PDF…";

  setBusy(true);

  try {
    const data = await api("/api/upload", {
      method: "POST",
      body: form
    });

    setDocumentReady(data);

    $("upload-status").textContent =
      "PDF processed successfully.";

    toast("Your document is ready.");
  } catch (error) {
    $("upload-status").textContent = error.message;
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

$("choose-file").addEventListener("click", () => {
  $("pdf-file").click();
});

$("pdf-file").addEventListener("change", (event) => {
  uploadFile(event.target.files[0]);
});

$("upload-form").addEventListener("submit", (event) => {
  event.preventDefault();
});

$("question").addEventListener("input", () => {
  $("char-count").textContent =
    `${$("question").value.length} / 2000`;

  $("send-button").disabled =
    !documentReady ||
    busy ||
    !$("question").value.trim();
});

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = $("question").value.trim();

  if (!question || !documentReady || busy) {
    return;
  }

  addMessage("You", question, "user");

  $("question").value = "";
  $("char-count").textContent = "0 / 2000";

  setBusy(true);

  const pending = addMessage(
    "DocuMind AI",
    "Searching your document and preparing an answer…",
    "loading"
  );

  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question })
    });

    pending.replaceChildren();

    const heading = document.createElement("div");
    heading.className = "message-label";
    heading.textContent = "DocuMind AI";

    const answer = document.createElement("div");
    answer.textContent = data.answer || "No answer returned.";

    pending.append(heading, answer);

    if (data.sources?.length) {
      const sourceTitle = document.createElement("div");
      sourceTitle.className = "message-label";
      sourceTitle.style.marginTop = "14px";
      sourceTitle.textContent = "Retrieved sources";

      const sources = document.createElement("div");
      sources.className = "sources";

      data.sources.forEach((item) => {
        const source = document.createElement("div");
        source.className = "source";

        source.textContent =
          `Page ${item.page}: ${item.text}`;

        sources.append(source);
      });

      pending.append(sourceTitle, sources);
    }
  } catch (error) {
    pending.textContent = error.message;
  } finally {
    setBusy(false);
    $("question").focus();
  }
});

async function runTool(endpoint, label, payload = {}) {
  if (!documentReady || busy) {
    toast("Upload a PDF first.");
    return;
  }

  $("output").replaceChildren();

  addMessage(
    label,
    "Working on your document…",
    "loading"
  );

  setBusy(true);

  try {
    const data = await api(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    $("output").replaceChildren();

    addMessage(
      label,
      data.result || data.answer || "No result returned."
    );
  } catch (error) {
    $("output").replaceChildren();
    addMessage("Notice", error.message);
  } finally {
    setBusy(false);
  }
}

$("nav-summary").addEventListener("click", () => {
  runTool("/api/summary", "Document summary");
});

$("nav-exam").addEventListener("click", () => {
  runTool(
    "/api/exam",
    "Practice exam",
    {
      count: 5,
      difficulty: "medium"
    }
  );
});

$("nav-chat").addEventListener("click", () => {
  $("welcome").classList.remove("hidden");
  $("output").classList.add("hidden");
});

$("clear-doc").addEventListener("click", async () => {
  if (busy) {
    toast("Please wait for the current operation to finish.");
    return;
  }

  try {
    await api("/api/reset", {
      method: "POST"
    });

    documentReady = false;

    $("document-banner").classList.add("hidden");

    $("question").disabled = true;
    $("question").value = "";
    $("question").placeholder =
      "Upload a PDF to start asking questions...";

    $("char-count").textContent = "0 / 2000";
    $("send-button").disabled = true;

    $("pdf-file").value = "";

    $("upload-status").textContent =
      "Current document cleared.";

    $("welcome").classList.remove("hidden");
    $("output").classList.add("hidden");
    $("output").replaceChildren();

    toast("Document cleared.");
  } catch (error) {
    toast(error.message);
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!documentReady) {
      toast("Upload a PDF first.");
      return;
    }

    $("question").value = button.dataset.question;

    $("char-count").textContent =
      `${$("question").value.length} / 2000`;

    $("send-button").disabled = busy;

    $("question").focus();
  });
});
