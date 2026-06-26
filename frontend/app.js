/**
 * Frontend logic for the Multi-Agent AI Research Platform.
 *
 * Pure vanilla JS — no frameworks, no build step.
 * Talks to the FastAPI backend at BASE_URL.
 */

const BASE_URL = "";  // same origin — served by FastAPI

// ── DOM handles ──────────────────────────────────────────────────────────────

const queryForm      = document.getElementById("query-form");
const queryInput      = document.getElementById("query-input");
const submitBtn       = document.getElementById("submit-btn");

const fileInput       = document.getElementById("file-upload");
const uploadBtn       = document.getElementById("upload-btn");
const uploadStatus    = document.getElementById("upload-status");

const pipelineSection = document.getElementById("pipeline-section");
const resultsSection  = document.getElementById("results-section");
const errorSection    = document.getElementById("error-section");
const errorMessage    = document.getElementById("error-message");

const resultMeta      = document.getElementById("result-meta");
const resultPlan      = document.getElementById("result-plan");
const resultResearch  = document.getElementById("result-research");
const resultTools     = document.getElementById("result-tools");
const resultReasoning = document.getElementById("result-reasoning");
const resultCritic    = document.getElementById("result-critic");
const resultAnswer    = document.getElementById("result-answer");

const STEP_IDS = [
    "step-router",
    "step-planner",
    "step-research",
    "step-tools",
    "step-reason",
    "step-critic",
    "step-synth",
];

// ── File upload enable / disable ─────────────────────────────────────────────

fileInput.addEventListener("change", () => {
    uploadBtn.disabled = !fileInput.files.length;
});

// ── Upload handler ───────────────────────────────────────────────────────────

uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    uploadStatus.textContent = "Uploading…";
    uploadStatus.className = "";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${BASE_URL}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Upload failed (${res.status})`);
        }

        const data = await res.json();
        uploadStatus.textContent = `✓ ${data.message}`;
        uploadStatus.className = "success";
    } catch (err) {
        uploadStatus.textContent = `✗ ${err.message}`;
        uploadStatus.className = "error";
    } finally {
        uploadBtn.disabled = false;
    }
});

// ── Query handler ────────────────────────────────────────────────────────────

queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const query = queryInput.value.trim();
    if (!query) return;

    // Reset UI
    resetUI();
    pipelineSection.classList.remove("hidden");
    submitBtn.disabled = true;
    submitBtn.textContent = "Processing…";

    // Simulate step progression while waiting for the response
    const stepTimer = simulateSteps();

    try {
        const res = await fetch(`${BASE_URL}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, session_id: "demo_session" }),
        });

        // Stop the step simulation
        clearInterval(stepTimer);

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed (${res.status})`);
        }

        const data = await res.json();
        markAllStepsDone(data);
        showResults(data);
    } catch (err) {
        clearInterval(stepTimer);
        showError(err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Send";
    }
});

// ── Step simulation ──────────────────────────────────────────────────────────

/**
 * Progressively mark pipeline steps as "active" one by one while
 * the real API call is in flight. Returns a setInterval ID.
 */
function simulateSteps() {
    let current = 0;

    // Mark the first step active immediately
    setStepState(STEP_IDS[0], "active");

    return setInterval(() => {
        if (current < STEP_IDS.length) {
            setStepState(STEP_IDS[current], "done");
        }
        current++;
        if (current < STEP_IDS.length) {
            setStepState(STEP_IDS[current], "active");
        }
    }, 2500);
}

function setStepState(id, state) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("active", "done", "skipped");
    if (state) el.classList.add(state);
}

/**
 * When the real response arrives, mark steps appropriately
 * based on what was actually skipped.
 */
function markAllStepsDone(data) {
    STEP_IDS.forEach((id) => setStepState(id, "done"));

    // If the router took the direct path, mark downstream as skipped
    if (data.plan === "Skipped by Router") {
        ["step-planner", "step-research", "step-tools", "step-reason", "step-critic", "step-synth"].forEach(
            (id) => setStepState(id, "skipped")
        );
    } else {
        // Mark individual agents skipped if they were bypassed
        if (data.research_context === "Skipped by Router") {
            setStepState("step-research", "skipped");
        }
        if (data.tool_results === "Skipped by Router") {
            setStepState("step-tools", "skipped");
        }
    }
}

// ── Display results ──────────────────────────────────────────────────────────

function showResults(data) {
    // Meta
    const parts = [];
    parts.push(`Time: ${data.processing_time_seconds}s`);
    parts.push(`Retries: ${data.retries_used}`);
    if (data.cached) parts.push("(cached)");
    resultMeta.textContent = parts.join("  ·  ");

    // Collapsible sections
    resultPlan.textContent      = data.plan      || "—";
    resultResearch.textContent  = data.research_context || "—";
    resultTools.textContent     = data.tool_results     || "—";
    resultReasoning.textContent = data.reasoning        || "—";
    resultCritic.textContent    = data.critic_feedback   || "—";

    // Final answer
    resultAnswer.textContent = data.answer || "No answer returned.";

    resultsSection.classList.remove("hidden");
    errorSection.classList.add("hidden");
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
}

function resetUI() {
    resultsSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    STEP_IDS.forEach((id) => setStepState(id, null));
}
