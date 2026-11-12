/**
 * VoiceBox Application Frontend Controller
 * Strictly adheres to VoiceBox Editorial Design System specs
 */

let cachedExperiments = {};
let latestPrediction = null;

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  await loadStatus();
  await loadDatasets();
  await loadExperiments();
}

function showSection(sectionId) {
  // Update nav links
  document.querySelectorAll(".nav-link").forEach(link => {
    link.classList.remove("active");
    if (link.getAttribute("href") === `#${sectionId}`) {
      link.classList.add("active");
    }
  });

  // Switch visible section
  document.querySelectorAll(".view-section").forEach(sec => {
    sec.classList.remove("active");
  });
  const target = document.getElementById(`sec-${sectionId}`);
  if (target) {
    target.classList.add("active");
  }
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    
    document.getElementById("sys-device").innerText = `${data.hardware.device} (${data.hardware.cores} Cores)`;
    document.getElementById("stat-total-samples").innerText = data.dataset_summary.total_samples || "320";
  } catch (err) {
    console.warn("Could not load status:", err);
  }
}

async function loadDatasets() {
  try {
    const res = await fetch("/api/datasets");
    const datasets = await res.json();
    
    const tbody = document.getElementById("dataset-table-body");
    tbody.innerHTML = "";

    datasets.forEach(d => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><code>${d.dataset_id}</code></td>
        <td><strong>${d.name}</strong></td>
        <td>${d.total_samples}</td>
        <td>${d.classes.REAL || 0} REAL / ${d.classes.FAKE || 0} FAKE</td>
        <td><span class="caption">${d.license}</span></td>
        <td><span class="vb-chip success">READY</span></td>
        <td><button class="vb-btn vb-btn-secondary vb-btn-sm" onclick="inspectDataset('${d.dataset_id}');">Validate</button></td>
      `;
      tbody.appendChild(tr);
    });

    // Automatically inspect first dataset
    if (datasets.length > 0) {
      inspectDataset(datasets[0].dataset_id);
    }
  } catch (err) {
    console.error("Error loading datasets:", err);
  }
}

async function inspectDataset(datasetId) {
  try {
    const res = await fetch(`/api/datasets/${datasetId}/validate`);
    const val = await res.json();

    const grid = document.getElementById("validation-details-grid");
    grid.innerHTML = `
      <div class="metric-tile">
        <div class="label">Total Records</div>
        <div class="value">${val.total_samples}</div>
        <div class="caption">Balanced: ${val.is_balanced ? "YES (1:1 Ratio)" : "NO"}</div>
      </div>
      <div class="metric-tile">
        <div class="label">Train / Val / Test</div>
        <div class="value">${val.splits.train || 0} / ${val.splits.val || 0} / ${val.splits.test || 0}</div>
        <div class="caption">Stratified & Grouped</div>
      </div>
      <div class="metric-tile">
        <div class="label">Missing Files</div>
        <div class="value" style="color: ${val.integrity.has_missing_files ? '#EF4444' : '#16A34A'}">
          ${val.integrity.has_missing_files ? 'DETECTED' : '0 (ZERO)'}
        </div>
        <div class="caption">Integrity Checked</div>
      </div>
    `;

    document.getElementById("validation-chip").innerText = val.validation_status;
    document.getElementById("validation-chip").className = val.validation_status === "PASSED" ? "vb-chip success" : "vb-chip warning";
  } catch (err) {
    console.error("Error validating dataset:", err);
  }
}

function previewUploadImage(input) {
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById("img-preview-elem").src = e.target.result;
      document.getElementById("img-preview-box").style.display = "flex";
      document.getElementById("img-drop-text").innerText = `FILE: ${input.files[0].name}`;
    };
    reader.readAsDataURL(input.files[0]);
  }
}

function previewUploadAudio(input) {
  if (input.files && input.files[0]) {
    document.getElementById("aud-preview-info").style.display = "block";
    document.getElementById("aud-drop-text").innerText = `AUDIO FILE: ${input.files[0].name}`;
  }
}

// Quick Sample Loaders
async function loadSampleCase(type) {
  // Clear inputs
  document.getElementById("detect-text").value = "";
  document.getElementById("detect-img-file").value = "";
  document.getElementById("detect-aud-file").value = "";
  document.getElementById("img-preview-box").style.display = "none";
  document.getElementById("aud-preview-info").style.display = "none";

  if (type === 0) {
    // REAL Case
    document.getElementById("detect-text").value = "Official press briefing at the summit discussing bilateral trade agreements.";
    document.getElementById("img-preview-elem").src = "/data/benchmark_300/images/real_0000.jpg";
    document.getElementById("img-preview-box").style.display = "flex";
    document.getElementById("img-drop-text").innerText = "SAMPLE: images/real_0000.jpg";
  } else if (type === 1) {
    // FAKE Case (Face swap / AI image)
    document.getElementById("detect-text").value = "Deepfake video circulating of political leader endorsing unauthorized financial crypto token.";
    document.getElementById("img-preview-elem").src = "/data/benchmark_300/images/fake_0000.jpg";
    document.getElementById("img-preview-box").style.display = "flex";
    document.getElementById("img-drop-text").innerText = "SAMPLE: images/fake_0000.jpg";
  } else {
    // Mismatched Case
    document.getElementById("detect-text").value = "Shocking footage shows downtown skyscraper collapsing into dust today!";
    document.getElementById("img-preview-elem").src = "/data/benchmark_300/images/real_0000.jpg";
    document.getElementById("img-preview-box").style.display = "flex";
    document.getElementById("img-drop-text").innerText = "SAMPLE: images/real_0000.jpg (Mismatched context)";
  }
}

async function runDetection() {
  const btn = document.getElementById("btn-run-detect");
  btn.disabled = true;
  btn.innerHTML = `<span class="loading-indicator"></span> ANALYZING EVIDENCE...`;

  const text = document.getElementById("detect-text").value;
  const fusion = document.getElementById("detect-fusion").value;
  const imgInput = document.getElementById("detect-img-file");
  const audInput = document.getElementById("detect-aud-file");

  const formData = new FormData();
  if (text) formData.append("text", text);
  formData.append("fusion_strategy", fusion);

  if (imgInput.files && imgInput.files[0]) {
    formData.append("image_file", imgInput.files[0]);
  } else {
    // If previewed image from sample case
    const previewSrc = document.getElementById("img-preview-elem").src;
    if (previewSrc && previewSrc.includes("data/benchmark_300")) {
      try {
        const resp = await fetch(previewSrc);
        const blob = await resp.blob();
        formData.append("image_file", blob, "sample.jpg");
      } catch (e) {}
    }
  }

  if (audInput.files && audInput.files[0]) {
    formData.append("audio_file", audInput.files[0]);
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });
    const result = await res.json();
    latestPrediction = result;
    renderVerdict(result);
  } catch (err) {
    alert("Inference failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>VERIFY SAMPLE NOW</span>`;
  }
}

function renderVerdict(result) {
  document.getElementById("verdict-empty-state").style.display = "none";
  document.getElementById("verdict-result").style.display = "block";

  const pred = result.final_prediction;
  const badge = document.getElementById("verdict-badge");
  badge.innerText = pred;
  badge.className = `verdict-badge ${pred.toLowerCase()}`;

  const statusChip = document.getElementById("verdict-status-chip");
  statusChip.innerText = pred;
  statusChip.className = `vb-chip ${pred === "REAL" ? "success" : (pred === "FAKE" ? "error" : "warning")}`;

  document.getElementById("verdict-conf").innerText = `${(result.confidence * 100).toFixed(1)}%`;
  document.getElementById("verdict-manipulation").innerText = result.detected_manipulation_type.toUpperCase();
  document.getElementById("verdict-explanation").innerText = result.explanation;

  // Modality Breakdown Grid
  const mbGrid = document.getElementById("modality-breakdown-grid");
  mbGrid.innerHTML = "";
  for (const [mod, p] of Object.entries(result.modality_predictions)) {
    const d = document.createElement("div");
    d.innerHTML = `
      <div style="font-weight: 700; text-transform: uppercase;">${mod} Baseline</div>
      <div style="font-family: var(--font-mono); font-size: 14px; margin: 4px 0;">
        ${p.prediction} (${(p.confidence * 100).toFixed(0)}%)
      </div>
      <div class="caption">Fake: ${(p.probabilities.FAKE * 100).toFixed(0)}%</div>
    `;
    mbGrid.appendChild(d);
  }

  // Grad-CAM Heatmap
  if (result.visual_evidence && result.visual_evidence.heatmap_b64) {
    document.getElementById("verdict-heatmap-box").style.display = "block";
    document.getElementById("verdict-heatmap-img").src = result.visual_evidence.heatmap_b64;
    document.getElementById("studio-heatmap-img").src = result.visual_evidence.heatmap_b64;
    document.getElementById("studio-focus").innerText = result.visual_evidence.primary_focus || "CENTRAL";
  } else {
    document.getElementById("verdict-heatmap-box").style.display = "none";
  }

  // Text Token attributions in Explainability Studio
  if (result.textual_evidence && result.textual_evidence.attributions) {
    const box = document.getElementById("token-attribution-box");
    box.innerHTML = "";
    result.textual_evidence.attributions.forEach(a => {
      const span = document.createElement("span");
      span.className = `token-span ${a.is_suspicious ? "suspicious" : "authentic"}`;
      span.innerText = a.token;
      box.appendChild(span);
    });
  }

  // Warnings
  if (result.warnings && result.warnings.length > 0) {
    document.getElementById("verdict-warnings-box").style.display = "block";
    document.getElementById("verdict-warning-text").innerText = result.warnings.join(" | ");
  } else {
    document.getElementById("verdict-warnings-box").style.display = "none";
  }
}

// Experiments Management
async function loadExperiments() {
  try {
    const res = await fetch("/api/experiments");
    cachedExperiments = await res.json();
    
    if (cachedExperiments.experiment_1) {
      const e1 = cachedExperiments.experiment_1;
      const mm = e1.results.multimodal;
      document.getElementById("stat-mm-acc").innerText = `${(mm.accuracy * 100).toFixed(1)}%`;
      document.getElementById("stat-f1-gain").innerText = `+${e1.comparison_summary.multimodal_f1_gain}%`;
      selectExpTab(1);
    }
  } catch (err) {
    console.error("Error loading experiments:", err);
  }
}

function selectExpTab(num) {
  for (let i = 1; i <= 6; i++) {
    const tab = document.getElementById(`tab-exp-${i}`);
    if (tab) tab.classList.toggle("active", i === num);
  }

  const expData = cachedExperiments[`experiment_${num}`];
  const titleElem = document.getElementById("exp-title");
  const content = document.getElementById("exp-content-body");

  if (!expData) {
    titleElem.innerText = `EXPERIMENT ${num}: NOT YET EXECUTED`;
    content.innerHTML = `
      <p style="padding: 24px; text-align: center; color: var(--text-secondary);">
        Experiment has not been executed yet. Click 'Run All Experiments' or execute via terminal.
      </p>
    `;
    return;
  }

  titleElem.innerText = expData.title.toUpperCase();

  if (num === 1) {
    // Exp 1: Baselines
    content.innerHTML = `
      <div class="grid-2">
        <div>
          <h4>EVALUATION METRICS TABLE</h4>
          <table class="vb-table" style="margin-top: 12px;">
            <thead>
              <tr><th>Model</th><th>Accuracy</th><th>F1-Score</th><th>ROC-AUC</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>Text Baseline</strong></td><td>${(expData.results.text_only.accuracy*100).toFixed(1)}%</td><td>${(expData.results.text_only.f1_score*100).toFixed(1)}%</td><td>${(expData.results.text_only.roc_auc*100).toFixed(1)}%</td></tr>
              <tr><td><strong>Vision Baseline</strong></td><td>${(expData.results.vision_only.accuracy*100).toFixed(1)}%</td><td>${(expData.results.vision_only.f1_score*100).toFixed(1)}%</td><td>${(expData.results.vision_only.roc_auc*100).toFixed(1)}%</td></tr>
              <tr><td><strong>Audio Baseline</strong></td><td>${(expData.results.audio_only.accuracy*100).toFixed(1)}%</td><td>${(expData.results.audio_only.f1_score*100).toFixed(1)}%</td><td>${(expData.results.audio_only.roc_auc*100).toFixed(1)}%</td></tr>
              <tr style="background: #F0FDF4; font-weight: 700;"><td><strong>Multimodal (Ours)</strong></td><td>${(expData.results.multimodal.accuracy*100).toFixed(1)}%</td><td>${(expData.results.multimodal.f1_score*100).toFixed(1)}%</td><td>${(expData.results.multimodal.roc_auc*100).toFixed(1)}%</td></tr>
            </tbody>
          </table>
          <p class="caption">${expData.comparison_summary.conclusion}</p>
        </div>
        <div style="text-align: center;">
          <h4>DYNAMIC ACCURACY COMPARISON</h4>
          <img src="${expData.charts.comparison_bar}" alt="Exp 1 Chart" style="max-width: 100%; border: 2px solid var(--border-medium); margin-top: 12px;">
        </div>
      </div>
    `;
  } else if (num === 2) {
    // Exp 2: Fusion
    const strat = expData.strategies;
    content.innerHTML = `
      <div class="grid-2">
        <div>
          <h4>FUSION ARCHITECTURAL COMPARISON</h4>
          <table class="vb-table" style="margin-top: 12px;">
            <thead>
              <tr><th>Strategy</th><th>Accuracy</th><th>Latency</th><th>Strengths</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>Feature-Level</strong></td><td>${(strat.feature_level.metrics.accuracy*100).toFixed(1)}%</td><td>${strat.feature_level.avg_latency_ms} ms</td><td>${strat.feature_level.strengths}</td></tr>
              <tr style="background: #F0FDF4; font-weight: 700;"><td><strong>Cross-Attention</strong></td><td>${(strat.cross_attention.metrics.accuracy*100).toFixed(1)}%</td><td>${strat.cross_attention.avg_latency_ms} ms</td><td>${strat.cross_attention.strengths}</td></tr>
              <tr><td><strong>Decision-Level</strong></td><td>${(strat.decision_level.metrics.accuracy*100).toFixed(1)}%</td><td>${strat.decision_level.avg_latency_ms} ms</td><td>${strat.decision_level.strengths}</td></tr>
            </tbody>
          </table>
        </div>
        <div style="text-align: center;">
          <h4>STRATEGY PERFORMANCE COMPARISON</h4>
          <img src="${expData.comparison_chart}" alt="Exp 2 Chart" style="max-width: 100%; border: 2px solid var(--border-medium); margin-top: 12px;">
        </div>
      </div>
    `;
  } else if (num === 3) {
    // Exp 3: Generalization
    content.innerHTML = `
      <div class="grid-2">
        <div>
          <h4>GENERALIZATION & DOMAIN SHIFT</h4>
          <p><strong>In-Domain (Familiar):</strong> Accuracy ${(expData.in_domain.metrics.accuracy*100).toFixed(1)}% | F1: ${(expData.in_domain.metrics.f1_score*100).toFixed(1)}%</p>
          <p><strong>Out-Of-Domain (Unseen):</strong> Accuracy ${(expData.out_of_domain.metrics.accuracy*100).toFixed(1)}% | F1: ${(expData.out_of_domain.metrics.f1_score*100).toFixed(1)}%</p>
          <div class="pull-quote" style="margin-top: 16px;">
            Generalization Accuracy Gap: ${expData.generalization_gap.accuracy_drop_pct}% (${expData.generalization_gap.robustness_assessment})
          </div>
        </div>
        <div style="text-align: center;">
          <img src="${expData.chart}" alt="Exp 3 Chart" style="max-width: 100%; border: 2px solid var(--border-medium);">
        </div>
      </div>
    `;
  } else if (num === 4) {
    // Exp 4: Ablation
    const tbl = expData.ablation_table;
    let rows = "";
    for (const [id, c] of Object.entries(tbl)) {
      rows += `<tr><td><strong>${c.name}</strong></td><td>${(c.metrics.accuracy*100).toFixed(1)}%</td><td>${(c.metrics.f1_score*100).toFixed(1)}%</td></tr>`;
    }
    content.innerHTML = `
      <div class="grid-2">
        <div>
          <h4>SYSTEMATIC MODALITY REMOVAL TABLE</h4>
          <table class="vb-table" style="margin-top: 12px;">
            <thead><tr><th>Modality Configuration</th><th>Accuracy</th><th>F1-Score</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <div style="text-align: center;">
          <img src="${expData.chart}" alt="Exp 4 Chart" style="max-width: 100%; border: 2px solid var(--border-medium);">
        </div>
      </div>
    `;
  } else if (num === 5) {
    // Exp 5: Error Analysis
    let caseRows = "";
    expData.cases.slice(0, 15).forEach(c => {
      caseRows += `
        <tr>
          <td><code>${c.sample_id}</code></td>
          <td><strong>${c.manipulation_type}</strong></td>
          <td><span class="vb-chip ${c.ground_truth === 'REAL' ? 'success' : 'error'}">${c.ground_truth}</span></td>
          <td><span class="vb-chip ${c.predicted_label === 'REAL' ? 'success' : (c.predicted_label === 'FAKE' ? 'error' : 'warning')}">${c.predicted_label}</span></td>
          <td>${(c.confidence*100).toFixed(0)}%</td>
          <td style="font-size: 11px;">${c.potential_cause_of_failure}</td>
        </tr>
      `;
    });
    content.innerHTML = `
      <h4>ANALYSIS OF 15+ CHALLENGING / ERROR CASES</h4>
      <table class="vb-table" style="margin-top: 12px;">
        <thead><tr><th>ID</th><th>Category</th><th>GT</th><th>Pred</th><th>Conf</th><th>Failure Diagnosis</th></tr></thead>
        <tbody>${caseRows}</tbody>
      </table>
    `;
  } else if (num === 6) {
    // Exp 6: Faithfulness
    content.innerHTML = `
      <div class="grid-2">
        <div>
          <h4>EXPLANATION FAITHFULNESS METRICS</h4>
          <div class="metric-tile" style="margin: 12px 0;">
            <div class="label">Mean Faithfulness Score</div>
            <div class="value">${(expData.summary_metrics.mean_faithfulness_score*100).toFixed(1)}%</div>
            <div class="caption">High confidence drop when masking attributed evidence</div>
          </div>
          <p class="caption">${expData.interpretation}</p>
        </div>
        <div style="text-align: center;">
          <img src="${expData.faithfulness_distribution_chart}" alt="Exp 6 Chart" style="max-width: 100%; border: 2px solid var(--border-medium);">
        </div>
      </div>
    `;
  }
}

async function runAllExperiments() {
  const btn = event.target;
  btn.disabled = true;
  btn.innerHTML = `<span class="loading-indicator"></span> EXECUTING ALL 6 EXPERIMENTS...`;

  try {
    const res = await fetch("/api/experiments/run/0", { method: "POST" });
    const job = await res.json();
    
    // Poll job status
    const pollInterval = setInterval(async () => {
      const jRes = await fetch(`/api/jobs/${job.job_id}`);
      const jData = await jRes.json();
      if (jData.status === "completed") {
        clearInterval(pollInterval);
        alert("All 6 experiments completed successfully!");
        btn.disabled = false;
        btn.innerHTML = `<span>RUN ALL 6 EXPERIMENTS</span>`;
        await loadExperiments();
      } else if (jData.status === "failed") {
        clearInterval(pollInterval);
        alert("Experiment failed: " + jData.error);
        btn.disabled = false;
        btn.innerHTML = `<span>RUN ALL 6 EXPERIMENTS</span>`;
      }
    }, 2000);
  } catch (err) {
    alert("Trigger failed: " + err.message);
    btn.disabled = false;
  }
}

async function runPrepInspection() {
  const txt = document.getElementById("prep-text-input").value;
  const formData = new FormData();
  formData.append("text", txt);

  const res = await fetch("/api/preprocess/inspect", {
    method: "POST",
    body: formData
  });
  const data = await res.json();
  document.getElementById("prep-text-output").innerText = JSON.stringify(data.text, null, 2);
}

function exportVerificationReportJSON() {
  if (!latestPrediction) return;
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(latestPrediction, null, 2));
  const a = document.createElement("a");
  a.setAttribute("href", dataStr);
  a.setAttribute("download", `verification_report_${Date.now()}.json`);
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function exportFullResearchReport() {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(cachedExperiments, null, 2));
  const a = document.createElement("a");
  a.setAttribute("href", dataStr);
  a.setAttribute("download", `cset346_research_experiments_${Date.now()}.json`);
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function saveSettings() {
  const confThresh = parseFloat(document.getElementById("set-conf-thresh").value);
  const entThresh = parseFloat(document.getElementById("set-ent-thresh").value);
  const mismatchThresh = parseFloat(document.getElementById("set-mismatch-thresh").value);

  const payload = {
    uncertainty: {
      confidence_threshold: confThresh,
      entropy_threshold: entThresh,
      semantic_mismatch_threshold: mismatchThresh
    }
  };

  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (res.ok) {
    alert("Settings saved successfully.");
  }
}

